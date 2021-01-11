/*
 * Copyright (c) 2020-2021, Intel Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Intel Corporation nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <ATen/record_function.h>
#include <ProcessGroupCCL.hpp>
#include <dispatch_stub.h>
#include <core/DPCPPUtils.h>
#include <core/Context.h>
#include <utils.h>

namespace torch_ccl
{

namespace {

// Check that all `tensors' have the same device and type and shape and
// are distributed across distinct GPUs if these are GPU tensors.
c10::DeviceType check_tensors_properties(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    throw std::runtime_error("Tensor list must be nonempty");
  }
  int device_count;
  at::dpcpp::dpcppGetDeviceCount(&device_count);
  if (tensors.size() > static_cast<size_t>(device_count)) {
    throw std::runtime_error(
      "Tensor list mustn't be larger than the number of available GPUs");
  }

  const auto& first = tensors.front();
  auto dev_type = first.device().type();

  // Set for ensuring that tensors are on separate devices.
  std::unordered_set<decltype(first.get_device())> usedDevices;
  usedDevices.reserve(tensors.size());

  for (const auto& t : tensors) {
    if (t.is_sparse()) {
      throw std::runtime_error("Tensors must be dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      throw std::runtime_error("Tensors must have identical type");
    }
    if (t.sizes() != first.sizes()) {
      throw std::runtime_error("Tensors must have identical size");
    }
    if (!t.is_contiguous()) {
      throw std::runtime_error("Tensors must be contiguous");
    }
    if (dev_type != t.device().type()) {
      throw std::runtime_error("Tensors must be on the same device type");
    }
    const auto inserted = usedDevices.insert(t.get_device()).second;
    if (!inserted) {
      throw std::runtime_error("Tensors must be on distinct devices");
    }
  }

  return dev_type;
}

Comms& get_ccl_comms(c10d::ProcessGroupCCL& pg_ccl, const std::string& devices_key, const std::vector<at::Device>& devices) {
  // Sanity check
  if (devices_key.empty()) {
    throw std::runtime_error(
            "Not able to create/get the CCL Communicator since "
            "the devices are empty ");
  }

  if (devices.size() != 1) {
    throw std::runtime_error("Torch CCL only support one device per process now");
  }

  if (pg_ccl.ccl_comms.find(devices_key) != pg_ccl.ccl_comms.end()) {
    // Reuse the cached communicator if there is one.
    return *pg_ccl.ccl_comms[devices_key];
  }

  // Create the gpu communicators
  // Only support the symmetric distributed communication
  int total_rank_size = pg_ccl.getSize() * devices.size();
  int local_base_rank = pg_ccl.getRank() * devices.size();
  ccl::vector_class<ccl::pair_class<int, cl::sycl::device>> devs_rank;
  for (size_t i = 0; i < devices.size(); ++i) {
    int rank = local_base_rank + i;
    auto sycl_dev = at::dpcpp::dpcppGetRawDevice(devices[i].index());
    devs_rank.emplace_back(rank, sycl_dev);
  }

  auto ctx = at::dpcpp::getDeviceContext(devices[0].index());
  auto dpcpp_comms = ccl::create_communicators(total_rank_size, devs_rank, ctx, pg_ccl.get_kvs());

  // Create the gpu streams
  std::vector<ccl::stream> ccl_streams;
  ccl_streams.reserve(devices.size());
  for(size_t i = 0; i < devices.size(); ++i)
  {
    // Use the same queue for computation and communication.
    // TODO: IPEX doesn't support multiple queue for now. Copy engine requires a dedicate queue
    auto q = at::dpcpp::getCurrentDPCPPStream(devices[i].index()).dpcpp_queue();
    ccl_streams.push_back(ccl::create_stream(q));
  }

  std::shared_ptr<Comms> dpcpp_comms_ptr = std::make_shared<Comms>(dpcpp_comms, ccl_streams);
  // Store the comms to cache
  pg_ccl.ccl_comms.emplace(devices_key, dpcpp_comms_ptr);

  return *dpcpp_comms_ptr.get();
}

} //namespace anonymous

class XPUCCLStubs final: public DispatchStub {

public:

  XPUCCLStubs() {}

  bool enabled() override {
    return true;
  }

  ~XPUCCLStubs() {}

protected:

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_(std::vector<at::Tensor>& tensors,
                                                            const AllreduceOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override;


  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce_(std::vector<at::Tensor>& tensors,
                                                         const ReduceOptions& opts,
                                                         ProcessGroupCCL& pg_ccl) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast_(std::vector<at::Tensor>& tensors,
                                                            const BroadcastOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const AllgatherOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override;

  void reset() override {
  }

private:
};

struct RegisterXPUMethods {
  RegisterXPUMethods() {
    static XPUCCLStubs methods;
    DispatchStub::register_ccl_stub(c10::DeviceType::XPU, &methods);
  }
};

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::allreduce_(std::vector<at::Tensor>& tensors,
                                                                       const AllreduceOptions& opts,
                                                                       ProcessGroupCCL& pg_ccl) {
  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms>(
    pg_ccl,
    tensors,
    tensors,
    [=](at::Tensor& input,
        at::Tensor& output,
        ccl::allreduce_attr attr,
        ccl::communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("torch_ccl::xpu::allreduce", std::vector<c10::IValue>({input}));

      ccl::event ret_evt;
      CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "torch_ccl::xpu::allreduce", [&] {
        call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
          CCL_CHECK(ret_evt = ccl::allreduce(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                                                  (size_t)input.numel(), cclOps.at(opts.reduceOp), comm, stream, attr););
        });
    });
    return ret_evt;
  });

  work->debugName = std::string("torch_ccl::xpu::allreduce::sz:") + std::to_string(tensors[0].numel());
  return work;
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::reduce_(std::vector<at::Tensor>& tensors,
                                                                    const ReduceOptions& opts,
                                                                    ProcessGroupCCL& pg_ccl) {
  const int root = opts.rootRank * tensors.size() + opts.rootTensor;
  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms>(
    pg_ccl,
    tensors,
    tensors,
    [=](at::Tensor& input,
        at::Tensor& output,
        ccl::reduce_attr attr,
        ccl::communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("torch_ccl::xpu::reduce", std::vector<c10::IValue>{input});

      ccl::event ret_evt;
      CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "torch_ccl::xpu::broadcast", [&] {
        call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
          CCL_CHECK(ret_evt = ccl::reduce(input.data_ptr<scalar_t>(),
                                  output.data_ptr<scalar_t>(),
                                  (size_t) input.numel(),
                                  cclOps.at(opts.reduceOp),
                                  root,
                                  comm,
                                  stream););
        });
      });
      return ret_evt;

  });
  work->debugName = std::string("torch_ccl::xpu::reduce::sz:") + std::to_string(tensors[0].numel());
  return work;
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::broadcast_(std::vector<at::Tensor>& tensors,
                                                                       const BroadcastOptions &opts,
                                                                       ProcessGroupCCL& pg_ccl) {
  const int root = opts.rootRank * tensors.size() + opts.rootTensor;
  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms>(
    pg_ccl,
    tensors,
    tensors,
    [=](at::Tensor& input,
        at::Tensor& output,
        ccl::broadcast_attr attr,
        ccl::communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("torch_ccl::xpu::broadcast", std::vector<c10::IValue>({input}));

      ccl::event ret_evt;
      CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "torch_ccl::xpu::broadcast", [&] {
        call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
          CCL_CHECK(ret_evt = ccl::broadcast(input.data_ptr<scalar_t>(), (size_t)input.numel(), root,
                                   comm, stream, attr););
        });
      });
      return ret_evt;
    });
  work->debugName = std::string("torch_ccl::xpu::broadcast::sz:") + std::to_string(tensors[0].numel());
  return work;
}


std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                       std::vector<at::Tensor>& inputTensors,
                                                                       const AllgatherOptions& opts,
                                                                       ProcessGroupCCL& pg_ccl) {
  const int rank = pg_ccl.getRank();
  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms>(
    pg_ccl,
    inputTensors,
    outputTensors,
    [=](at::Tensor input,
        std::vector<at::Tensor>& outputs,
        ccl::allgatherv_attr attr,
        ccl::communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("torch_ccl::xpu::allgather", std::vector<c10::IValue>({input}));

      ccl::event ret_evt;
      CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "torch_ccl::xpu::allgather", [&] {
        std::vector<size_t> recvCounts(outputs.size(), 0);
        std::transform(outputs.begin(), outputs.end(), recvCounts.begin(),
                       [](const at::Tensor& t) {
                            return t.numel();
                       });

        TORCH_CHECK((size_t)input.numel() == recvCounts[rank], "allgather: send and recv count doesn't match");
        std::vector<scalar_t*> recvBufs(outputs.size(), nullptr);
        std::transform(outputs.begin(), outputs.end(), recvBufs.begin(),
                       [](const at::Tensor& t) {
                          return t.data_ptr<scalar_t>();
                       });

        call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
          CCL_CHECK(ret_evt = ccl::allgatherv(input.data_ptr<scalar_t>(),
                                    (size_t) input.numel(),
                                    recvBufs,
                                    recvCounts,
                                    comm,
                                    stream););
        });
      });

      return ret_evt;
    });
  work->debugName = std::string("torch_ccl::xpu::allgather::sz:") +  std::to_string(inputTensors[0].numel());
  return work;
}

RegisterXPUMethods xpu_register;

}