//
// Created by johnlu on 2019/12/10.
//


#include <torch/extension.h>
#include <pybind11/chrono.h>
#include "ProcessGroupCCL.hpp"

#ifndef OCCL_LIBNAME
#define OCCL_LIBNAME liboccl
#endif

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PYBIND11_MODULE(OCCL_LIBNAME, m) {
  m.def("create_pg_occl", &c10d::ProcessGroupCCL::createProcessGroupCCL, "Create One CCL Backend");
//  m.def("_occl_version", );
//
//{"_nccl_version", (PyCFunction)THCPModule_nccl_version, METH_NOARGS, nullptr},
//{"_nccl_unique_id", (PyCFunction)THCPModule_nccl_unique_id, METH_NOARGS, nullptr},
//{"_nccl_init_rank", (PyCFunction)THCPModule_nccl_init_rank, METH_VARARGS, nullptr},
//{"_nccl_reduce", (PyCFunction)THCPModule_nccl_reduce, METH_VARARGS, nullptr},
//{"_nccl_all_reduce", (PyCFunction)THCPModule_nccl_all_reduce, METH_VARARGS, nullptr},
//{"_nccl_broadcast", (PyCFunction)THCPModule_nccl_broadcast, METH_VARARGS, nullptr},
//{"_nccl_all_gather", (PyCFunction)THCPModule_nccl_all_gather, METH_VARARGS, nullptr},
//{"_nccl_reduce_scatter", (PyCFunction)THCPModule_nccl_reduce_scatter, METH_VARARGS, nullptr},
  py::object module = py::module::import("torch.distributed");
  py::object register_backend = module.attr("Backend").attr("register_backend");
  // The first parameter is the backend name used by user in invoking
  // torch.distributed.init_process_group().
  // Note it could be different with module name. For example, the module
  // name is "torch_test" but the backend name is "test".
  // The second parameter is the instantiation function.
  register_backend("occl", py::cpp_function(&c10d::ProcessGroupCCL::createProcessGroupCCL,
                                            py::arg("store"),
                                            py::arg("rank"),
                                            py::arg("size"),
                                            py::arg("timeout") = std::chrono::milliseconds(
                                              ::c10d::ProcessGroupCCL::OP_TIMEOUT_MILLIS)));

  auto processGroup = module.attr("ProcessGroup");
  auto processGroupOCCL = shared_ptr_class_<::c10d::ProcessGroupCCL>(
    module, "ProcessGroupOCCL", processGroup);

//  shared_ptr_class_<::gloo::transport::Device>(processGroupGloo, "Device");

//  shared_ptr_class_<::c10d::ProcessGroupGloo::Options>(
//    processGroupGloo, "Options")
//    .def(py::init<>())
//    .def_readwrite("devices", &::c10d::ProcessGroupGloo::Options::devices)
//    .def_readwrite("timeout", &::c10d::ProcessGroupGloo::Options::timeout)
//    .def_readwrite("threads", &::c10d::ProcessGroupGloo::Options::threads);

//  processGroupOCCL.def_static(
//    "create_device",
//    [](const std::string& hostname, const std::string& interface)
//      -> std::shared_ptr<::gloo::transport::Device> {
//      if (!hostname.empty()) {
//        return ::c10d::ProcessGroupGloo::createDeviceForHostname(hostname);
//      }
//      if (!interface.empty()) {
//        return ::c10d::ProcessGroupGloo::createDeviceForInterface(interface);
//      }
//      throw std::invalid_argument(
//        "Specify either `hostname` or `interface` argument.");
//    },
//    py::arg("hostname") = "",
//    py::arg("interface") = "");

  processGroupOCCL
    .def(py::init<
//      const std::shared_ptr<::c10d::Store>&,
      int,
      int//,
      /*std::chrono::milliseconds*/>())
    .def(
      py::init([](const std::shared_ptr<::c10d::Store>& store,
                  int rank,
                  int size,
                  std::chrono::milliseconds timeout) {
        return std::make_shared<::c10d::ProcessGroupCCL>(rank, size);
      }),
      py::arg("store"),
      py::arg("rank"),
      py::arg("size"),
      py::arg("timeout") = std::chrono::milliseconds(10 * 1000));

}