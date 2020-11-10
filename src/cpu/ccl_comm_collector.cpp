//
// Created by johnlu on 2020/9/26.
//
#include "ccl_cpu.h"
#include "ccl_comm_collector.h"

namespace torch_ccl {
template <>
void Comms<class CPU>::sync_streams(std::vector<at::Device> devices) {}

//std::unordered_map<std::string, std::shared_ptr<CCLCommsCollector<class CPU>::CommsType>> 
//CCLCommsCollector<class CPU>::get_comms_map(){
//  return comms_map;
//}
/*template <>
std::vector<CCLCommsCollector<class CPU>::CommsType>
CCLCommsCollector<class CPU>::get_all_ccl_comms() {
  std::vector<CCLCommsCollector<class CPU>::CommsType> comms;
  for(auto iter = comms_map.begin(); iter != comms_map.end(); iter++){
     comms.push_back(*(iter->second).get());
  }
  return comms;
}*/

template <>
CCLCommsCollector<class CPU>::CommsType&
CCLCommsCollector<class CPU>::get_ccl_comms(const std::string& devices_key, const std::vector<at::Device>& devices) {
  // Sanity check
  if (devices_key.empty()) {
    throw std::runtime_error(
      "Not able to create/get the CCL Communicator since "
      "the devices are empty ");
  }

  TORCH_CHECK(devices.size() == 1, "CPU device size must be 1");

  std::shared_ptr<CommsType> cpu_comms_ptr = get_ccl_comms_(devices_key);
  if (cpu_comms_ptr) {
    // Reuse the cached communicator if there is one.
    return *cpu_comms_ptr.get();
  }
  ccl::vector_class<ccl::communicator> cpu_comms;
  cpu_comms.emplace_back(ccl::create_communicator(size_, rank_, kvs_));
  cpu_comms_ptr = std::make_shared<CommsType>(cpu_comms);
  set_ccl_comms_(devices_key, cpu_comms_ptr);

  return *cpu_comms_ptr.get();
}
}

