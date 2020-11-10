//
// Created by johnlu on 2020/9/26.
//

#ifndef TORCH_CCL_CCL_COMM_COLLECTOR_H
#define TORCH_CCL_CCL_COMM_COLLECTOR_H

#include <c10/core/Device.h>
#include <oneapi/ccl.hpp>
#include <unordered_map>

namespace torch_ccl {

template <typename DevType>
class Comms {
public:
  explicit Comms(ccl::vector_class<ccl::communicator> &comms) :
    comms(std::move(comms)), streams{} {}

  explicit Comms(ccl::vector_class<ccl::communicator> &comms, std::vector<ccl::stream> &streams) :
    comms(std::move(comms)), streams(std::move(streams)) {}

  ~Comms() noexcept(false) {}

  Comms() = delete;

  // Must not be copyable
  Comms(const Comms &) = delete;

  Comms &operator=(const Comms &) = delete;

  // Move constructable
  Comms(Comms &&other) : comms(std::move(other.comms)), streams(other.streams) {}

  // Move assignable
  Comms &operator=(Comms &&other) {
    std::swap(comms, other.comms);
    std::swap(streams, other.streams);
    return *this;
  }

  void sync_streams(std::vector<at::Device>);

public:
  // The Communicators used by CCL
  ccl::vector_class<ccl::communicator> comms;
  // The steams used by CCL
  ccl::vector_class<ccl::stream> streams;
};

template <typename DevType>
class CCLCommsCollector {
public:
  using CommsType = Comms<DevType>;
  CCLCommsCollector(int rank = -1, int size = -1, ccl::shared_ptr_class<ccl::kvs> kvs = nullptr) :
    rank_(rank), size_(size), kvs_(kvs), comms_map{} {}

  ~CCLCommsCollector() noexcept {}

  // Must not be copyable
  CCLCommsCollector(const CCLCommsCollector &) = delete;

  CCLCommsCollector &operator=(const CCLCommsCollector &) = delete;

  // Move constructable
  CCLCommsCollector(CCLCommsCollector &&other) = delete;

  // Move assignable
  CCLCommsCollector &operator=(CCLCommsCollector &&other) = delete;

  CommsType& get_ccl_comms(const std::string& devices_key, const std::vector<at::Device>& devices);
  
  std::unordered_map<std::string, std::shared_ptr<CommsType>> get_comms_map(){
     return comms_map;
  }

private:
  std::shared_ptr<CommsType> get_ccl_comms_(const std::string& devices_key) {
    if (comms_map.find(devices_key) != comms_map.end()) {
      // Reuse the cached communicator if there is one.
      return comms_map[devices_key];
    }
    else {
      return nullptr;
    }
  }

  void set_ccl_comms_(const std::string& devices_key, std::shared_ptr<CommsType>& comms) {
    comms_map.emplace(devices_key, comms);
  }

  int rank_;
  int size_;
  // The kvs is unique ID among the processes.
  ccl::shared_ptr_class<ccl::kvs> kvs_;

  // The CCL communicator that the process group has cached.
  // The key is a list of GPU devices that an operation is operating on
  // The GPU devices are stored in a device sequence and the cache CCL
  // communicator is associated with this GPU device sequence
  //
  // e.g. If the process group op only uses device 0, then the value of
  // the used device string stored (value of the hashmap) would be "0".
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
  //      then the value of the used device string (key) stored would be
  //      "0,1,2,3,4,5,6,7"
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
  //      then the value of the used device string stored would be
  //      "0,4,5,6,7,1,2,3"
  //
  //      Note that the order of the device for the tensor list matters.
  std::unordered_map<std::string, std::shared_ptr<CommsType>> comms_map;
};

}
#endif //TORCH_CCL_CCL_COMM_COLLECTOR_H
