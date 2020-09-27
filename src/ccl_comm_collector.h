//
// Created by johnlu on 2020/9/26.
//

#ifndef TORCH_CCL_CCL_COMM_COLLECTOR_H
#define TORCH_CCL_CCL_COMM_COLLECTOR_H

#include <c10/core/Device.h>
#include <oneapi/ccl.hpp>

namespace torch_ccl {

template <typename CommType>
class Comms {
public:
  explicit Comms(ccl::vector_class<CommType> &comms, std::vector<ccl::stream> &streams) :
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

public:
  // The Communicators used by CCL
  ccl::vector_class<CommType> comms;
  // The steams used by CCL
  std::vector<ccl::stream> streams;
};

using GPUComms = Comms<ccl::device_communicator>;

using CPUComms = Comms<ccl::communicator>;

class CCLCommsCollector {
public:
  CCLCommsCollector(int rank, int size, ccl::shared_ptr_class<ccl::kvs> kvs) :
    rank_(rank), size_(size), kvs_(kvs) {}

  ~CCLCommsCollector() noexcept {}

  // Must not be copyable
  CCLCommsCollector(const CCLCommsCollector &) = delete;

  CCLCommsCollector &operator=(const CCLCommsCollector &) = delete;

  // Move constructable
  CCLCommsCollector(CCLCommsCollector &&other) = delete;

  // Move assignable
  CCLCommsCollector &operator=(CCLCommsCollector &&other) = delete;

  template <typename CommType>
  std::shared_ptr<CommType> get_ccl_comms(const std::string& devices_key);

  template <typename CommType>
  void set_ccl_comms(const std::string& devices_key, std::shared_ptr<CommType>& comms);

  template<class comm_type>
  comm_type& get_ccl_comms(const std::string& devices_key, const std::vector<at::Device>& devices);

private:
  int rank_;
  int size_;
  // The kvs is unique ID among the processes.
  ccl::shared_ptr_class<ccl::kvs> kvs_;

  // host communicator
  std::unordered_map<std::string, std::shared_ptr<CPUComms>> cpu_comms_map;

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
  std::unordered_map<std::string, std::shared_ptr<GPUComms>> gpu_comms_map;

};

}
#endif //TORCH_CCL_CCL_COMM_COLLECTOR_H
