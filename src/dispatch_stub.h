//
// Created by johnlu on 2020/9/24.
//

#ifndef TORCH_CCL_DISPATCH_STUBS_H
#define TORCH_CCL_DISPATCH_STUBS_H

#include "ProcessGroupCCL.hpp"

namespace torch_ccl {

using namespace c10d;

constexpr typename std::underlying_type<c10::DeviceType>::type to_int(c10::DeviceType dev_type) noexcept {
  return static_cast<typename std::underlying_type<c10::DeviceType>::type>(dev_type);
}

class DispatchStub {

public:
  DispatchStub() {}

  virtual bool enabled() {
    return false;
  }

  static std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce(std::vector<at::Tensor>& tensors,
                                                                  const AllreduceOptions& opts,
                                                                  ccl::communicator& comm) {
    c10::DeviceType dev_type = tensors[0].device().type();
    return stubs_[to_int(dev_type)]->allreduce_(tensors, opts, comm);
  }

  static std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce(std::vector<at::Tensor>& tensors,
                                                                const ReduceOptions& opts,
                                                                ccl::communicator& comm) {
    c10::DeviceType dev_type = tensors[0].device().type();
    return stubs_[to_int(dev_type)]->reduce_(tensors, opts, comm);
  }

  static std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast(std::vector<at::Tensor>& tensors,
                                                               const BroadcastOptions& opts,
                                                               ccl::communicator& comm) {
    c10::DeviceType dev_type = tensors[0].device().type();
    return stubs_[to_int(dev_type)]->broadcast_(tensors, opts, comm);
  }

  static std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                   std::vector<at::Tensor>& inputTensors,
                                                                   const AllgatherOptions& opts,
                                                                   ccl::communicator& comm) {
    c10::DeviceType dev_type = inputTensors[0].device().type();
    return stubs_[to_int(dev_type)]->allgather_(outputTensors, inputTensors, opts, comm);
  }

  virtual ~DispatchStub() {};

  static void register_ccl_stub(c10::DeviceType devoce_type, DispatchStub* stub);

protected:
  virtual std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_(std::vector<at::Tensor>& tensors,
                                                                    const AllreduceOptions& opts,
                                                                    ccl::communicator& comm) {
    fail(tensors[0].device().type(), "allreduce");
    return nullptr;
  }

  virtual std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce_(std::vector<at::Tensor>& tensors,
                                                                 const ReduceOptions& opts,
                                                                 ccl::communicator& comm) {
    fail(tensors[0].device().type(), "reduce");
    return nullptr;
  }

  virtual std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                    std::vector<at::Tensor>& inputTensors,
                                                                    const AllgatherOptions& opts,
                                                                    ccl::communicator& comm) {
    fail(inputTensors[0].device().type(), "allgather");
    return nullptr;
  }

  virtual std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast_(std::vector<at::Tensor>& tensors,
                                                                    const BroadcastOptions& opts,
                                                                    ccl::communicator& comm) {
    fail(tensors[0].device().type(), "broadcast");
    return nullptr;
  }

private:
  static DispatchStub* stubs_[static_cast<std::underlying_type<c10::DeviceType>::type>(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];
  static void fail(c10::DeviceType dev_type, const std::string method) {
    TORCH_CHECK(false, "torch_ccl: ", method, " isn't implementd on backend [", dev_type, "].");
  }
};

}

#endif //TORCH_CCL_DISPATCH_STUBS_H
