//
// Created by johnlu on 2020/9/24.
//
#include <ATen/record_function.h>
#include <ProcessGroupCCL.hpp>
#include <dispatch_stub.h>
#include <core/DPCPPUtils.h>
#include <core/Context.h>

#define CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      /*AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, char, __VA_ARGS__)*/          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()


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

// Get the device type from list of tensors
at::Device get_dev_type(const std::vector<at::Tensor>& tensors) {
  return tensors[0].device();
}

// Get the list of devices from list of tensors
at::Device get_dev_type(const std::vector<std::vector<at::Tensor>>& tensor_lists) {
  return tensor_lists[0][0].device();
}

// Get the list of devices from list of tensors
std::vector<at::Device> get_device_list(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.device());
  }
  return res;
}

// Get the list of devices from list of tensors
std::vector<at::Device> get_device_list(const std::vector<std::vector<at::Tensor>>& tensor_lists) {
  std::vector<at::Device> res;
  res.reserve(tensor_lists.size());
  for (auto& tensors : tensor_lists) {
    res.push_back(tensors[0].device());
  }
  return res;
}

// Get the deviceList String from the list of devices
std::string get_key_from_devs(const std::vector<at::Device>& devices) {
  std::string deviceList;
  for (auto& device : devices) {
    if (deviceList.empty()) {
      deviceList = std::to_string(device.index());
    } else {
      deviceList += "," + std::to_string(device.index());
    }
  }
  return deviceList;
}

std::shared_ptr<CCLCommsCollector> get_ccl_comms(
  const std::string& devicesKey,
  const std::vector<at::Device>& devices,
  ProcessGroupCCL* pg_ccl) {
  // Sanity check
  if (devicesKey.empty()) {
    throw std::runtime_error(
      "Not able to create/get the CCL Communicator since "
      "the devices are empty ");
  }

  assert(devices.size() == 1 && "device size must be 1");
//
//  for (auto& device : devices) {
//    used_gpu_device_idxs.insert(device.index());
//  }

  if (pg_ccl->gpu_comms.find(devicesKey) != pg_ccl->gpu_comms.end()) {
    // Reuse the cached communicator if there is one.
    return pg_ccl->gpu_comms[devicesKey];
  }

  ccl::vector_class<ccl::pair_class<ccl::rank_t, cl::sycl::device>> devs_rank;

  for (size_t i = 0; i < devices.size(); ++i) {
    // GPU world size and GPU rank
    int rank = pg_ccl->getRank() * devices.size() + i;
    auto sycl_dev = at::dpcpp::dpcppGetRawDevice(devices[i].index());
    devs_rank.emplace_back( rank, sycl_dev );
  }

  auto ctx = at::dpcpp::getGlobalContext();
  auto communcators = ccl::environment::instance().create_device_communicators(
    pg_ccl->getSize(),
    devs_rank,
    ctx,
    pg_ccl->kvs);


  auto &comm = *communcators.begin();

  std::vector<ccl::stream> ccl_streams;
  ccl_streams.reserve(devices.size());

  for(size_t i = 0; i < devices.size(); ++i)
  {
    /* create SYCL stream */
    auto q = at::dpcpp::getCurrentDPCPPStream(devices[i].index()).dpcpp_queue();
    ccl_streams.push_back(ccl::environment::instance().create_stream(q));
  }

  auto comms_collector = std::make_shared<class CCLCommsCollector>(communcators, ccl_streams);

  // Move the CCL resource to cache
  pg_ccl->gpu_comms.emplace(devicesKey, std::move(comms_collector));

  return pg_ccl->gpu_comms[devicesKey];
}

template <class RunF>
class DPCPPWorkCCL : public ProcessGroupCCL::AsyncWorkCCL {
public:
  DPCPPWorkCCL(const std::vector<at::Tensor>& inputs,
               const std::vector<at::Tensor>& outputs,
               const RunF f,
               std::shared_ptr<CCLCommsCollector> gpu_comms) : AsyncWorkCCL(inputs, outputs), f(f), gpu_comms(gpu_comms) {}

  void run() override {
    req = f(inputs[0], outputs[0], gpu_comms->gpu_comms[0], gpu_comms->gpu_streams[0]);
  };

private:
  RunF f;
  std::shared_ptr<CCLCommsCollector> gpu_comms;
};

template <class RunF>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> make_work_ccl(const std::vector<at::Tensor>& inputs,
                                                             const std::vector<at::Tensor>& outputs,
                                                             RunF f,
                                                             std::shared_ptr<CCLCommsCollector> gpu_comms) {
  return std::make_shared<DPCPPWorkCCL<RunF>>(inputs, outputs, f, gpu_comms);
}

template <typename fn, typename pre_process, typename post_process, typename input_t, typename output_t>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> collective(
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun,
  pre_process pre,
  post_process post,
  ProcessGroupCCL* pg_ccl) {
  const auto devices = get_device_list(inputs);
  const auto key = get_key_from_devs(devices);
  auto comms_collector = get_ccl_comms(key, devices, pg_ccl);

  // First let CCL streams wait for computing kernel on the input tensors's finished.
//  syncStreams(devices, comms_collector->gpu_streams);

  // Work itself will create the CUDA events on all GPUs of tensors

//  pre(gpu_streams[key]);

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = make_work_ccl(inputs, outputs, fun, comms_collector);
//  for (size_t i = 0; i < inputs.size(); ++i) {
//    auto req = fun(inputs[i], outputs[i], comms_collector.gpu_comms[i], comms_collector.gpu_streams[i]);
//    work->requests_[i] = std::move(req);
//  }

//  post(gpu_streams[key]);

  return work;
}

template <typename fn, typename input_t, typename output_t>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> collective(
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun,
  ProcessGroupCCL* pg_ccl) {
  return collective(
    inputs,
    outputs,
    fun,
    [](std::vector<ccl::stream>&) {},
    [](std::vector<ccl::stream>&) {},
    pg_ccl);
}

} //namespace anonymous

class DPCPPCCLStubs final: public DispatchStub {

public:
  DPCPPCCLStubs() {}

  bool enabled() override {
    return true;
  }

  ~DPCPPCCLStubs() {}

protected:

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_(std::vector<at::Tensor>& tensors,
                                                            const AllreduceOptions& opts,
                                                            ProcessGroupCCL* pg_ccl) override;


  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce_(std::vector<at::Tensor>& tensors,
                                                         const ReduceOptions& opts,
                                                         ProcessGroupCCL* pg_ccl) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast_(std::vector<at::Tensor>& tensors,
                                                            const BroadcastOptions& opts,
                                                            ProcessGroupCCL* pg_ccl) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const AllgatherOptions& opts,
                                                            ProcessGroupCCL* pg_ccl) override;

};

struct RegisterDPCPPPMethods {
  RegisterDPCPPPMethods() {
    static DPCPPCCLStubs methods;
    DispatchStub::register_ccl_stub(c10::DeviceType::DPCPP, &methods);
    printf("register dpcpp backend\n");
  }
};


std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> DPCPPCCLStubs::allreduce_(std::vector<at::Tensor>& tensors,
                                                                         const AllreduceOptions& opts,
                                                                         ProcessGroupCCL* pg_ccl) {
  TORCH_CHECK(false, "not implemented");
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> DPCPPCCLStubs::reduce_(std::vector<at::Tensor>& tensors,
                                                                      const ReduceOptions& opts,
                                                                      ProcessGroupCCL* pg_ccl) {
  TORCH_CHECK(false, "not implemented");
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> DPCPPCCLStubs::broadcast_(std::vector<at::Tensor>& tensors,
                                                                         const BroadcastOptions &opts,
                                                                         ProcessGroupCCL* pg_ccl) {
  auto dev_type = check_tensors_properties(tensors);

  return collective(tensors, tensors,
                    [&](at::Tensor input,
                        at::Tensor output,
                        ccl::device_communicator& comm,
                        ccl::stream& stream) {
                      RECORD_FUNCTION("pg::allgather", std::vector<c10::IValue>({tensors[0]}));
                      auto count = input.numel();
                      auto attr = ccl::environment::instance().create_operation_attr<ccl::broadcast_attr>();

                      const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
                      ccl::communicator::coll_request_t ret_req;

                      CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "broadcast", [&] {
                          CCL_CHECK(ret_req = comm.broadcast(
                            input.data_ptr<scalar_t>(),
                            (size_t)count,
                            root,
                            stream,
                            attr));
                      });
                      return ret_req;
                    },
                    pg_ccl);
}


std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> DPCPPCCLStubs::allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                         std::vector<at::Tensor>& inputTensors,
                                                                         const AllgatherOptions& opts,
                                                                         ProcessGroupCCL* pg_ccl) {
  TORCH_CHECK(false, "not implemented");
}

RegisterDPCPPPMethods dpcpp_register;

}