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
  std::string deviceList = DeviceTypeName(devices[0].type(), /* lower case */ true) + ":";
  for (auto& device : devices) {
    deviceList.append(std::to_string(device.index()) + ",");
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

std::vector<at::Device> get_device_list(const std::vector<std::vector<at::Tensor> >& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor[0].device());
  }
  return res;
}

}
