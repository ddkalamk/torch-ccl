//
// Created by johnlu on 2020/9/27.
//

#include "utils.h"

namespace torch_ccl {

// Op mapping
using c10d::ReduceOp;
std::map<c10d::ReduceOp, ccl::reduction> cclOps =
  {
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
  };

// Get the deviceList String from the list of devices
std::string get_key_from_devs(const std::vector<at::Device>& devices) {
  std::string deviceList;
  for (auto& device : devices) {
    if (deviceList.empty()) {
      deviceList = std::to_string(device.index());
    } else {
      deviceList += "," + std::to_string(device.index());
    }
  }
  return deviceList;
}

// Get the list of devices from list of tensors
std::vector<at::Device> get_device_list(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.device());
  }
  return res;
}

void prologue_wrap(const void* in_buf,
                    size_t in_count,
                    ccl::datatype in_dtype,
                    void** out_buf,
                    size_t* out_count,
                    ccl::datatype* out_dtype,
                    const ccl::fn_context* context) {
  if (out_buf)
    *out_buf = (void*)in_buf;
  if (out_count)
    *out_count = in_count;
  if (out_dtype)
    *out_dtype = in_dtype;
}


void epilogue_wrap(const void* in_buf,
                    size_t in_count,
                    ccl::datatype in_dtype,
                    void* out_buf,
                    size_t* out_count,
                    ccl::datatype* out_dtype,
                    const ccl::fn_context* context) {
  if (out_count)
    *out_count = in_count;
}

}