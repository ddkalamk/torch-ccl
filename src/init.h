#ifndef TORCH_CCL_INIT_H
#define TORCH_CCL_INIT_H

#include <pybind11/pybind11.h>

#define TORCH_CCL_CPP_API __attribute__ ((visibility ("default")))

void torch_ccl_python_init(pybind11::module &m);

#endif //TORCH_CCL_INIT_H
