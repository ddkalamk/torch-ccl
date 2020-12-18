//
// Created by johnlu on 2019/12/10.
//
#include <init.h>
#include <pybind11/chrono.h>
#include "ProcessGroupCCL.hpp"

namespace py = pybind11;

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

TORCH_CCL_CPP_API void torch_ccl_python_init(pybind11::module &m) {

  m.def("oneCCL_version", []() {
      return "1.0";
  });

  py::object module = py::module::import("torch.distributed");
  py::object register_backend = module.attr("Backend").attr("register_backend");

  register_backend("ccl", py::cpp_function(&c10d::ProcessGroupCCL::createProcessGroupCCL,
                                           py::arg("store"),
                                           py::arg("rank"),
                                           py::arg("size"),
                                           py::arg("timeout") = std::chrono::milliseconds(
                                                   ::c10d::ProcessGroupCCL::OP_TIMEOUT_MILLIS)));

  auto processGroup = module.attr("ProcessGroup");
  auto processGroupOCCL = shared_ptr_class_<::c10d::ProcessGroupCCL>(
          module, "ProcessGroupOCCL", processGroup);

  processGroupOCCL.def(
    py::init([](const std::shared_ptr<::c10d::Store>& store,
                int rank,
                int size,
                std::chrono::milliseconds timeout) {
      return std::make_shared<::c10d::ProcessGroupCCL>(store, rank, size, timeout);
    }),
    py::arg("store"),
    py::arg("rank"),
    py::arg("size"),
    py::arg("timeout") = std::chrono::milliseconds(10 * 1000));

}
