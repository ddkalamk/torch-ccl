//
// Created by johnlu on 2020/9/26.
//

#include "ccl_comm_collector.h"
#include <core/DPCPPUtils.h>
#include <core/Context.h>

namespace torch_ccl {

template <typename CommType>
void set_ccl_comm(const std::string& devices_key, std::shared_ptr<CommType>& comms);

template <>
std::shared_ptr<CPUComms> CCLCommsCollector::get_ccl_comms<CPUComms>(const std::string& devices_key) {
  if (cpu_comms_map.find(devices_key) != cpu_comms_map.end()) {
    // Reuse the cached communicator if there is one.
    return cpu_comms_map[devices_key];
  }
  else {
    return nullptr;
  }
}

template <>
std::shared_ptr<GPUComms> CCLCommsCollector::get_ccl_comms<GPUComms>(const std::string& devices_key) {
  if (gpu_comms_map.find(devices_key) != gpu_comms_map.end()) {
    // Reuse the cached communicator if there is one.
    return gpu_comms_map[devices_key];
  }
  else {
    return nullptr;
  }
}

template <>
void CCLCommsCollector::set_ccl_comms<CPUComms>(const std::string& devices_key, std::shared_ptr<CPUComms>& gpu_comms) {
  cpu_comms_map.emplace(devices_key, gpu_comms);
}

template <>
void CCLCommsCollector::set_ccl_comms<GPUComms>(const std::string& devices_key, std::shared_ptr<GPUComms>& cpu_comms) {
  gpu_comms_map.emplace(devices_key, cpu_comms);
}

template <>
CPUComms& CCLCommsCollector::get_ccl_comms<CPUComms>(const std::string& devices_key, const std::vector<at::Device>& devices) {
  // Sanity check
  if (devices_key.empty()) {
    throw std::runtime_error(
      "Not able to create/get the CCL Communicator since "
      "the devices are empty ");
  }

  assert(devices.size() == 1 && "device size must be 1");

  std::shared_ptr<CPUComms> cpu_comms_ptr = get_ccl_comms<CPUComms>(devices_key);
  if (cpu_comms_ptr) {
    // Reuse the cached communicator if there is one.
    return *cpu_comms_ptr.get();
  }
  ccl::vector_class<ccl::communicator> cpu_comms;
  cpu_comms.emplace_back(ccl::environment::instance().create_communicator(size_, rank_, kvs_));
  std::vector<ccl::stream> cpu_streams = {ccl::default_stream};
  cpu_comms_ptr = std::make_shared<CPUComms>(cpu_comms, cpu_streams);
  set_ccl_comms<CPUComms>(devices_key, cpu_comms_ptr);

  return *cpu_comms_ptr.get();
}

template <>
GPUComms& CCLCommsCollector::get_ccl_comms<GPUComms>(const std::string& devices_key, const std::vector<at::Device>& devices) {
  // Sanity check
  if (devices_key.empty()) {
    throw std::runtime_error(
      "Not able to create/get the CCL Communicator since "
      "the devices are empty ");
  }
//
//  for (auto& device : devices) {
//    used_gpu_device_idxs.insert(device.index());
//  }
  std::shared_ptr<GPUComms> gpu_comms_ptr = get_ccl_comms<GPUComms>(devices_key);
  if (gpu_comms_ptr) {
    // Reuse the cached communicator if there is one.
    return *gpu_comms_ptr.get();
  }

  //Create the gpu communicators
  ccl::vector_class<ccl::pair_class<ccl::rank_t, cl::sycl::device>> devs_rank;

  for (size_t i = 0; i < devices.size(); ++i) {
    // GPU world size and GPU rank
    int rank = rank_ * devices.size() + i;
    auto sycl_dev = at::dpcpp::dpcppGetRawDevice(devices[i].index());
    devs_rank.emplace_back( rank, sycl_dev );
  }

  auto ctx = at::dpcpp::getGlobalContext();
  auto communcators = ccl::environment::instance().create_device_communicators(
    size_,
    devs_rank,
    ctx,
    kvs_);

  // Create the gpu streams
  auto &comm = *communcators.begin();
  std::vector<ccl::stream> ccl_streams;
  ccl_streams.reserve(devices.size());

  for(size_t i = 0; i < devices.size(); ++i)
  {
    auto q = at::dpcpp::getCurrentDPCPPStream(devices[i].index()).dpcpp_queue();
    ccl_streams.push_back(ccl::environment::instance().create_stream(q));
  }

  gpu_comms_ptr = std::make_shared<GPUComms>(communcators, ccl_streams);

  // Move the CCL resource to cache
  set_ccl_comms<GPUComms>(devices_key, gpu_comms_ptr);

  return *gpu_comms_ptr.get();
}

}

