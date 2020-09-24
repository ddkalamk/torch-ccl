//
// Created by johnlu on 2020/9/24.
//
#include <ATen/record_function.h>
#include <ProcessGroupCCL.hpp>
#include <dispatch_stub.h>


namespace torch_ccl
{

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
                                                            ccl::communicator& comm) override;


  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce_(std::vector<at::Tensor>& tensors,
                                                         const ReduceOptions& opts,
                                                         ccl::communicator& comm) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast_(std::vector<at::Tensor>& tensors,
                                                            const BroadcastOptions& opts,
                                                            ccl::communicator& comm) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const AllgatherOptions& opts,
                                                            ccl::communicator& comm) override;

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
                                                                     ccl::communicator& comm) {
  TORCH_CHECK(false, "not implemented");
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> DPCPPCCLStubs::reduce_(std::vector<at::Tensor>& tensors,
                                                                  const ReduceOptions& opts,
                                                                  ccl::communicator& comm) {
  TORCH_CHECK(false, "not implemented");
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> DPCPPCCLStubs::broadcast_(std::vector<at::Tensor>& tensors,
                                                                     const BroadcastOptions &opts,
                                                                     ccl::communicator& comm) {
  TORCH_CHECK(false, "not implemented");
}


std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> DPCPPCCLStubs::allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                     std::vector<at::Tensor>& inputTensors,
                                                                     const AllgatherOptions& opts,
                                                                     ccl::communicator& comm) {
  TORCH_CHECK(false, "not implemented");
}

RegisterDPCPPPMethods dpcpp_register;

}