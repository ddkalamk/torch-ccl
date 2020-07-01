#pragma once


#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

#include <ccl.hpp>

namespace c10d {

// WorkCCL is the state associated with a CCL operarion.
//
// ProcessGroupCCL implements CCL bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group.
//
// All collective functions provided by this class is scheduled for asynchronous execution by CCL.
//
// Also note that ProcessGroupCCL only supports a single Tensor operation. In
// other words, the size of the input Tensor vector should always be 1.
//
// RAII CCL communicators collector
class CCLCommsCollector {
public:
  explicit CCLCommsCollector(ccl::communicator_t gcomm, std::vector<ccl::communicator_t> comms) :
    group_gomm(std::move(gcomm)),
    gpu_comms(std::move(comms))
  {}

  ~CCLCommsCollector() noexcept(false) {
  }

  CCLCommsCollector() = delete;

  // Must not be copyable
  CCLCommsCollector(const CCLCommsCollector&) = delete;
  CCLCommsCollector& operator=(const CCLCommsCollector&) = delete;

  // Move constructable
  CCLCommsCollector(CCLCommsCollector&& other) {
    std::swap(group_gomm, other.group_gomm);
    std::swap(gpu_comms, other.gpu_comms);
  }
  // Move assignable
  CCLCommsCollector& operator=(CCLCommsCollector&& other) {
    std::swap(group_gomm, other.group_gomm);
    std::swap(gpu_comms, other.gpu_comms);
    return *this;
  }

public:
  ccl::communicator_t group_gomm;
  std::vector<ccl::communicator_t> gpu_comms;
};

class ProcessGroupCCL : public ProcessGroup, public std::enable_shared_from_this<ProcessGroupCCL> {
public:
  class WorkCCL final : public ProcessGroup::Work {
  public:
    WorkCCL(std::vector<at::Device> devices)
    {
      requests_.resize(devices.size());
    }

    WorkCCL(ccl::communicator::coll_request_t& req)
    {
      requests_.push_back(std::move(req));
    }

    WorkCCL() {}

    virtual ~WorkCCL();

    bool isCompleted() override;

    bool isSuccess() const override;

    bool wait() override;

  protected:
    std::vector<ccl::communicator::coll_request_t> requests_;
    mutable std::mutex mutex_;

    friend class ProcessGroupCCL;
  };

  explicit ProcessGroupCCL(int rank = -1, int size = -1);
  virtual ~ProcessGroupCCL();

  std::shared_ptr<ProcessGroup::Work> broadcast(
          std::vector<at::Tensor>& data,
          const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce(
          std::vector<at::Tensor>& tensors,
          const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce(
          std::vector<at::Tensor>& tensors,
          const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather(
          std::vector<std::vector<at::Tensor>>& outputTensors,
          std::vector<at::Tensor>& inputTensors,
          const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> barrier(
          const BarrierOptions& opts = BarrierOptions()) override;

  // Unsupported Ops
  std::shared_ptr<ProcessGroup::Work> gather(
          std::vector<std::vector<at::Tensor>>& outputTensors,
          std::vector<at::Tensor>& inputTensors,
          const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> scatter(
          std::vector<at::Tensor>& outputTensors,
          std::vector<std::vector<at::Tensor>>& inputTensors,
          const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce_scatter(
          std::vector<at::Tensor>& outputTensors,
          std::vector<std::vector<at::Tensor>>& inputTensors,
          const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> send(
          std::vector<at::Tensor>& tensors,
          int dstRank,
          int tag) override;

  std::shared_ptr<ProcessGroup::Work> recv(
          std::vector<at::Tensor>& tensors,
          int srcRank,
          int tag) override;

  std::shared_ptr<ProcessGroup::Work> recvAnysource(
          std::vector<at::Tensor>& tensor,
          int tag) override;

  std::shared_ptr<ProcessGroup::Work> allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts = AllreduceCoalescedOptions()) override;

  // Creating a new ProcessGroupCCL, will initiialize CCL if not initialized
  static std::shared_ptr<ProcessGroup> createProcessGroupCCL(const std::shared_ptr<Store>& store,
                                                             int rank,
                                                             int size,
                                                             const std::chrono::milliseconds& op_time_out =
                                                             std::chrono::milliseconds(OP_TIMEOUT_MILLIS));

  static const int64_t OP_TIMEOUT_MILLIS;
private:

  // Helper that either looks up the cached CCL communicators or creates
  // a new set of CCL communicators as a cache entry
  std::vector<ccl::communicator_t>& getCCLComm(
          const std::string& devicesKey,
          const std::vector<at::Device>& devices);

  // Helper that encapsulates work shared across all collective communication
  // primitives.  The callbacks have the following signatures:
  //
  //    ncclResult_t fn(at::Tensor& input, at::Tensor& output,
  //                    ncclComm_t, at::cuda::CUDAStream&);
  //    void {pre,post}(std::vector<at::cuda::CUDAStream&>);
  template <typename Fn>
  std::shared_ptr<ProcessGroup::Work> collective(
          std::vector<at::Tensor>& input,
          std::vector<at::Tensor>& output,
          Fn fn);
  template <typename Fn, typename PreProcess, typename PostProcess>
  std::shared_ptr<ProcessGroup::Work> collective(
          std::vector<at::Tensor>& input,
          std::vector<at::Tensor>& output,
          Fn fn,
          PreProcess pre,
          PostProcess post);

  template <typename Fn, typename PreProcess, typename PostProcess>
  std::shared_ptr<ProcessGroup::Work> collective_gpu(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post);

  template <typename Fn, typename PreProcess, typename PostProcess>
  std::shared_ptr<ProcessGroup::Work> collective_cpu(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post);

protected:

  // Global states
  static void cclInit();
  static void cclExit();
  static std::once_flag init_flag;
  static std::mutex pg_global_mutex;
  static ccl::coll_attr attr;
  ccl::communicator_t global_comm;
  std::vector<ccl::stream_t> cpu_streams;

  // The steams used by CCL kernels
  std::unordered_map<std::string, std::vector<ccl::stream_t >> gpu_streams;

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
  using ccl_comm_t = std::shared_ptr<class CCLCommsCollector>;
  std::unordered_map<std::string, ccl_comm_t> gpu_comms;

  // GPU device indexes used for all collectives in this group
  std::set<int> used_gpu_device_idxs;
};

} // namespace c10d
