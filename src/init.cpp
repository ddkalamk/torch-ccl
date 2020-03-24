//
// Created by johnlu on 2019/12/10.
//


#include <torch/extension.h>
#include <pybind11/chrono.h>
#include "ProcessGroupCCL.hpp"

PYBIND11_MODULE(liboccl, m) {
  m.def("occl_group", &c10d::ProcessGroupCCL::createProcessGroupCCL, "Create One CCL Backend");
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


}