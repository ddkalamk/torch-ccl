//
// Created by johnlu on 2020/9/24.
//
#include <ATen/record_function.h>
#include <ProcessGroupCCL.hpp>
#include <dispatch_stub.h>
#include <core/DPCPPUtils.h>
#include <core/Context.h>
#include <utils.h>

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
std::vector<at::Device> get_device_list(const std::vector<std::vector<at::Tensor>>& tensor_lists) {
  std::vector<at::Device> res;
  res.reserve(tensor_lists.size());
  for (auto& tensors : tensor_lists) {
    res.push_back(tensors[0].device());
  }
  return res;
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
                                                                         ProcessGroupCCL& pg_ccl) {
  TORCH_CHECK(false, "not implemented");
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> DPCPPCCLStubs::reduce_(std::vector<at::Tensor>& tensors,
                                                                      const ReduceOptions& opts,
                                                                      ProcessGroupCCL& pg_ccl) {
  TORCH_CHECK(false, "not implemented");
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> DPCPPCCLStubs::broadcast_(std::vector<at::Tensor>& tensors,
                                                                         const BroadcastOptions &opts,
                                                                         ProcessGroupCCL& pg_ccl) {
  auto dev_type = check_tensors_properties(tensors);

  return collective(
    pg_ccl.ccl_comms,
    tensors,
    tensors,
    [&](at::Tensor input,
        at::Tensor output,
        ccl::device_communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("torch_ccl::dpcpp::broadcast", std::vector<c10::IValue>({input}));
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
    });
}


std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> DPCPPCCLStubs::allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                         std::vector<at::Tensor>& inputTensors,
                                                                         const AllgatherOptions& opts,
                                                                         ProcessGroupCCL& pg_ccl) {
  TORCH_CHECK(false, "not implemented");
}

RegisterDPCPPPMethods dpcpp_register;

}