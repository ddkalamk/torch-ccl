//
// Created by johnlu on 2020/9/26.
//

#include <ccl_comm_collector.h>
#include <core/DPCPPUtils.h>
#include <core/Context.h>

namespace torch_ccl {
template <>
Comms&
CCLCommsCollector<class DPCPP>::get_ccl_comms(const std::string& devices_key, const std::vector<at::Device>& devices) {
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
  std::shared_ptr<Comms> gpu_comms_ptr = get_ccl_comms_(devices_key);
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
  auto communcators = ccl::create_communicators(
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
    ccl_streams.push_back(ccl::create_stream(q));
  }

  gpu_comms_ptr = std::make_shared<Comms>(communcators, ccl_streams);

  // Move the CCL resource to cache
  set_ccl_comms_(devices_key, gpu_comms_ptr);

  return *gpu_comms_ptr.get();
}
}

