//
// Created by johnlu on 2020/9/26.
//

#include "ccl_comm_collector.h"

namespace torch_ccl {

template <>
std::shared_ptr<CCLCommsCollector<ccl::communicator>::CommsType>
CCLCommsCollector<ccl::communicator>::get_ccl_comms(const std::string& devices_key) {
  if (comms_map.find(devices_key) != comms_map.end()) {
    // Reuse the cached communicator if there is one.
    return comms_map[devices_key];
  }
  else {
    return nullptr;
  }
}

template <>
void CCLCommsCollector<ccl::communicator>::set_ccl_comms(const std::string& devices_key,
                                                         std::shared_ptr<CCLCommsCollector<ccl::communicator>::CommsType>& gpu_comms) {
  comms_map.emplace(devices_key, gpu_comms);
}

template <>
CCLCommsCollector<ccl::communicator>::CommsType&
CCLCommsCollector<ccl::communicator>::get_ccl_comms(const std::string& devices_key, const std::vector<at::Device>& devices) {
  // Sanity check
  if (devices_key.empty()) {
    throw std::runtime_error(
      "Not able to create/get the CCL Communicator since "
      "the devices are empty ");
  }

  TORCH_CHECK(devices.size() == 1, "CPU device size must be 1");

  std::shared_ptr<CCLCommsCollector<ccl::communicator>::CommsType> cpu_comms_ptr = get_ccl_comms(devices_key);
  if (cpu_comms_ptr) {
    // Reuse the cached communicator if there is one.
    return *cpu_comms_ptr.get();
  }
  ccl::vector_class<ccl::communicator> cpu_comms;
  cpu_comms.emplace_back(ccl::environment::instance().create_communicator(size_, rank_, kvs_));
  std::vector<ccl::stream> cpu_streams = {ccl::default_stream};
  cpu_comms_ptr = std::make_shared<CCLCommsCollector<ccl::communicator>::CommsType>(cpu_comms, cpu_streams);
  set_ccl_comms(devices_key, cpu_comms_ptr);

  return *cpu_comms_ptr.get();
}
}

