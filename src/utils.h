//
// Created by johnlu on 2020/9/27.
//

#ifndef TORCH_CCL_UTILS_H
#define TORCH_CCL_UTILS_H

#include "ProcessGroupCCL.hpp"
#include <ATen/detail/FunctionTraits.h>

namespace torch_ccl {

using c10d::ProcessGroupCCL;

// Get the deviceList String from the list of devices
std::string get_key_from_devs(const std::vector<at::Device>& devices);

// Get the list of devices from list of tensors
std::vector<at::Device> get_device_list(const std::vector<at::Tensor>& tensors);

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
    run_wrap_<num_params>();
  };

private:
  template <int num_params>
  void run_wrap_() {}

  template <>
  void run_wrap_<3>() {
    req = f(inputs[0], outputs[0], comms.comms[0]);
  }

  template <>
  void run_wrap_<4>() {
    req = f(inputs[0], outputs[0], comms.comms[0], comms.streams[0]);
  }

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

template <typename fn, typename pre_process, typename post_process, typename input_t, typename output_t>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> collective(
  CCLCommsCollector& ccl_comms,
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun,
  pre_process pre,
  post_process post) {
  using traits = function_traits<fn>;
  // The function input is input, output comm, and stream optional
  using CommType = Comms<std::remove_reference_t<typename traits::template arg<2>::type>>;

  const auto devices = get_device_list(inputs);
  const auto key = get_key_from_devs(devices);
  auto& comms = ccl_comms.get_ccl_comms<CommType>(key, devices);

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

template <typename fn, typename input_t, typename output_t>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> collective(
  CCLCommsCollector& ccl_comms,
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
