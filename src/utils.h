//
// Created by johnlu on 2020/9/27.
//

#ifndef TORCH_CCL_UTILS_H
#define TORCH_CCL_UTILS_H

#include "ProcessGroupCCL.hpp"
#include <ATen/detail/FunctionTraits.h>
#include <c10d/Types.hpp>

#define CCL_CHECK(cmd)                                               \
  do {                                                               \
    try {                                                            \
        cmd;                                                         \
    }                                                                \
    catch (std::runtime_error& e) {                                  \
      throw e;                                                       \
    }                                                                \
    catch (...) {                                                    \
        throw std::runtime_error("unknown error in: " +              \
            std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
    }                                                                \
  } while (0)

#define CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      /*AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, char, __VA_ARGS__)*/      \
      /*AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int, __VA_ARGS__)*/        \
      /*AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)*/   \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

namespace torch_ccl {

using c10d::ProcessGroupCCL;

extern std::map<c10d::ReduceOp, ccl::reduction> cclOps;

// Get the deviceList String from the list of devices
std::string get_key_from_devs(const std::vector<at::Device>& devices);

// Get the list of devices from list of tensors
std::vector<at::Device> get_device_list(const std::vector<at::Tensor>& tensors);

static ccl::communicator::coll_request_t null_req;

template <typename RunF, typename CommType>
class AsyncWorkCCLWrap : public ProcessGroupCCL::AsyncWorkCCL {
public:
  using traits = function_traits<RunF>;
  static constexpr int num_params = traits::arity;

  AsyncWorkCCLWrap(const std::vector<at::Tensor>& inputs,
               const std::vector<at::Tensor>& outputs,
               const RunF f,
               CommType& comms) : AsyncWorkCCL(inputs, outputs), f(f), comms(comms) {}

  void run() override {
    using Indices = std::make_index_sequence<num_params - 3>;
      run_wrap_(Indices{});
  };

  ~AsyncWorkCCLWrap()
  {
    for(auto& req : reqs) {
      if (req != null_req)
      {
        std::cerr << "attempted destruction of WorkCCL before work has completed, "
                  << "terminating the program."
                  << std::endl;
        std::terminate();
      }
    }
  }

  bool isCompleted() override
  {
    for(auto& req : reqs) {
      if (req == null_req) {
        continue;
      }

      bool flag;

      CCL_CHECK(flag = req.test());

      if (flag)
      {
//      req.reset();
        req.operator=(ccl::communicator::coll_request_t());
      }
      else {
        return false;
      }
    }
    inputs.clear();
    // all request has been finished
    return true;
  }

  bool isSuccess() const override
  {
    for(auto& req : reqs) {
      if (req != null_req) {
        throw std::runtime_error(
          "invalid call to WorkCCL::isSuccess before work has completed");
      }
    }
    return true;
  }

  bool wait() override
  {
    for(auto& req : reqs) {
      if (req == null_req) {
        continue;
      }

      CCL_CHECK(req.wait());
      //    req.reset();
      req.operator=(ccl::communicator::coll_request_t());
    }
    inputs.clear();
    // Always return true, because abort API is not implemented.
    return true;
  }

  void abort() override
  {
    TORCH_CHECK(false, "ProcessGroupCCL::WorkCCL::abort not implemented");
  }

private:

  template <std::size_t...INDEX>
  void run_wrap_(std::index_sequence<INDEX...>) {
    for (size_t i = 0; i < inputs.size(); i++) {
      CCL_CHECK(reqs.push_back(f(inputs[i], outputs[i], comms.comms[i], comms.streams[i + INDEX]...)));
    }
  }

  std::vector<ccl::communicator::coll_request_t> reqs;
  RunF f;
  CommType& comms;
};

template <typename RunF, typename CommType>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> make_work_ccl(const std::vector<at::Tensor>& inputs,
                                                             const std::vector<at::Tensor>& outputs,
                                                             RunF f,
                                                             CommType& comms) {
  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> ret_ptr;
  ret_ptr.reset(new AsyncWorkCCLWrap<RunF, CommType>(inputs, outputs, f, comms));
  return ret_ptr;
}

template <typename comm_t, typename fn, typename pre_process, typename post_process, typename input_t, typename output_t>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> collective(
  CCLCommsCollector<comm_t>& ccl_comms,
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun,
  pre_process pre,
  post_process post) {
  using traits = function_traits<fn>;

  const auto devices = get_device_list(inputs);
  const auto key = get_key_from_devs(devices);
  auto& comms = ccl_comms.get_ccl_comms(key, devices);

  // First let CCL streams wait for computing kernel on the input tensors's finished.
//  syncStreams(devices, comms_collector->gpu_streams);

  // Work itself will create the CUDA events on all GPUs of tensors

//  pre(gpu_streams[key]);

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = make_work_ccl(inputs, outputs, fun, comms);
//  for (size_t i = 0; i < inputs.size(); ++i) {
//    auto req = fun(inputs[i], outputs[i], comms_collector.gpu_comms[i], comms_collector.gpu_streams[i]);
//    work->requests_[i] = std::move(req);
//  }

//  post(gpu_streams[key]);

  return work;
}

template <typename comm_t, typename fn, typename input_t, typename output_t>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> collective(
  CCLCommsCollector<comm_t>& ccl_comms,
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun) {
  return collective(
    ccl_comms,
    inputs,
    outputs,
    fun,
    [](std::vector<ccl::stream>&) {},
    [](std::vector<ccl::stream>&) {});
}

}

#endif //TORCH_CCL_UTILS_H
