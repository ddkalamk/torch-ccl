//
// Created by johnlu on 2020/9/27.
//

#ifndef TORCH_CCL_UTILS_H
#define TORCH_CCL_UTILS_H

#include "ProcessGroupCCL.hpp"
#include <ATen/detail/FunctionTraits.h>
#include <c10d/Types.hpp>
#include <oneapi/ccl/ccl_coll_attr_ids.hpp>

#define CCL_CHECK(cmd)                                               \
  do {                                                               \
    try {                                                            \
        cmd;                                                         \
    }                                                                \
    catch (std::runtime_error& e) {                                  \
      throw e;                                                       \
    }                                                                \
  }while(0)

#define CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      /*AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, char, __VA_ARGS__)  */    \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
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

template <typename RunF, typename CommType, typename InputType, typename OutputType, typename attr_t>
class AsyncWorkCCLWrap : public ProcessGroupCCL::AsyncWorkCCL {
public:
  using traits = function_traits<RunF>;
  static constexpr int num_params = traits::arity;

  AsyncWorkCCLWrap(const std::vector<InputType>& inputs,
               const std::vector<OutputType>& outputs,
               const RunF f,
               CommType& comms,
               attr_t& attr) : AsyncWorkCCL(), f(f), comms(comms), attr(attr), inputs(inputs), outputs(outputs), reqs{} {}

  void run() override {
    using Indices = std::make_index_sequence<num_params - 4>;
      run_wrap_(Indices{});
  };

  ~AsyncWorkCCLWrap()
  {
    if (!reqs.empty()) {
      std::cerr << "attempted destruction of WorkCCL before work has completed, "
                << "terminating the program."
                << std::endl;
      std::terminate();
    }
  }

  bool isCompleted() override
  {
    for(auto& req : reqs) {
      bool flag;

      CCL_CHECK(flag = req.test());

      if (!flag) {
        return false;
      }
    }
    // all request has been finished
    return true;
  }

  bool isSuccess() const override
  {
    throw std::runtime_error(
      "invalid call to ::isSuccess.");
  }

  bool wait() override
  {
    for(auto& req : reqs) {
      CCL_CHECK(req.wait());
    }
    reqs.clear();
    // Always return true, because abort API is not implemented.
    return true;
  }

  void abort() override
  {
    TORCH_CHECK(false, "ProcessGroupCCL::WorkCCL::abort not implemented");
  }

//  std::vector<OutputType>& getOutputTensors()
//  {
//    return outputs;
//  }
//
//  std::vector<InputType>& getInputTensors()
//  {
//    return inputs;
//  }

private:

  template <std::size_t...INDEX>
  void run_wrap_(std::index_sequence<INDEX...>) {
    if (reqs.empty()) {
      for (size_t i = 0; i < inputs.size(); i++) {
        CCL_CHECK(reqs.push_back(f(inputs[i], outputs[i], attr, comms.comms[i], comms.streams[i + INDEX]...)));
      }
    }
    else {
      // add warning for re run the ccl work
    }
  }

  RunF f;
  CommType& comms;
  attr_t attr;
  /*
      keep copy of tensors to increment tensor reference counters
      while CCL operation is in progress
  */
  std::vector<InputType> inputs;
  std::vector<OutputType> outputs;
  std::vector<ccl::communicator::coll_request_t> reqs;
};

template <typename RunF, typename CommType, typename InputType, typename OutputType, typename attr_t>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> make_work_ccl(const std::vector<InputType>& inputs,
                                                             const std::vector<OutputType>& outputs,
                                                             RunF f,
                                                             CommType& comms,
                                                             attr_t& attr) {
  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> ret_ptr;
  ret_ptr.reset(new AsyncWorkCCLWrap<RunF, CommType, InputType, OutputType, attr_t>(inputs, outputs, f, comms, attr));
  return ret_ptr;
}
#if 0
class callback_context {
  virtual void run_hook() = 0;
}

template<typename RunF>
class cpu_callback : public callback_context {
  virtual void run_hook() {
    actural_run();
  };

  private:
    void actural_run() {
      f();
    }

  RunF f;
}
#endif
extern void prologue_wrap(const void* in_buf,
                    size_t in_count,
                    ccl::datatype in_dtype,
                    void** out_buf,
                    size_t* out_count,
                    ccl::datatype* out_dtype,
                    const ccl::fn_context* context);

extern void epilogue_wrap(const void* in_buf,
                    size_t in_count,
                    ccl::datatype in_dtype,
                    void* out_buf,
                    size_t* out_count,
                    ccl::datatype* out_dtype,
                    const ccl::fn_context* context);

template <typename comm_t, typename fn, typename pre_process, typename post_process, typename input_t, typename output_t>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> collective(
  CCLCommsCollector<comm_t>& ccl_comms,
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun,
  pre_process pre,
  post_process post) {
  using traits = function_traits<fn>;
  using attr_t = typename traits::template arg<2>::type;
  attr_t attr = ccl::create_operation_attr<attr_t>();
//  attr.template set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn)prologue_wrap);
//  attr.template set<ccl::operation_attr_id::epilogue_fn>((ccl::epilogue_fn)epilogue_wrap);

  const auto devices = get_device_list(inputs);
  const auto key = get_key_from_devs(devices);
  auto& comms = ccl_comms.get_ccl_comms(key, devices);
  // First let CCL streams wait for computing kernel on the input tensors's finished.
  comms.sync_streams(devices);

//  pre(gpu_streams[key]);

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = make_work_ccl(inputs, outputs, fun, comms, attr);

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
