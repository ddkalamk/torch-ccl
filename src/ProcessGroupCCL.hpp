/*
 * Copyright (c) 2020, Intel Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Intel Corporation nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once


#include <exception>
#include <memory>
#include <mutex>
#include <vector>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>
#include <oneapi/ccl.hpp>

#define CCL_CHECK(cmd)                                               \
  do {                                                               \
    try {                                                            \
        cmd;                                                         \
    }                                                                \
    catch (ccl::ccl_error& e) {                                      \
        throw std::runtime_error("CCL error in: " +                  \
            std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
            ", with error message: " + e.what());                    \
    }                                                                \
    catch (...) {                                                    \
        throw std::runtime_error("unknown error in: " +              \
            std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
    }                                                                \
  } while (0)

namespace c10d {

// WorkCCL is the state associated with a CCL operarion.
//
// ProcessGroupCCL implements CCL bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group.
//
// All collective functions provided by this class are scheduled
// for asynchronous execution by CCL.

class CCLCommsCollector {
public:
  explicit CCLCommsCollector(ccl::vector_class<ccl::device_communicator>& comms, std::vector<ccl::stream>& gpu_streams) :
    gpu_comms(std::move(comms)), gpu_streams(std::move(gpu_streams))
  {}

  ~CCLCommsCollector() noexcept(false) {
  }

  CCLCommsCollector() = delete;

  // Must not be copyable
  CCLCommsCollector(const CCLCommsCollector&) = delete;
  CCLCommsCollector& operator=(const CCLCommsCollector&) = delete;

  // Move constructable
  CCLCommsCollector(CCLCommsCollector&& other) : gpu_comms(std::move(other.gpu_comms)), gpu_streams(other.gpu_streams){}
  // Move assignable
  CCLCommsCollector& operator=(CCLCommsCollector&& other) {
    std::swap(gpu_comms, other.gpu_comms);
    std::swap(gpu_streams, other.gpu_streams);
    return *this;
  }

public:
  ccl::vector_class<ccl::device_communicator> gpu_comms;
  // The steams used by CCL kernels
  std::vector<ccl::stream> gpu_streams;
};

class ProcessGroupCCL : public ProcessGroup
{

public:

  class AsyncWorkCCL : public ProcessGroup::Work {
  public:
    AsyncWorkCCL(const std::vector<at::Tensor>& tensors) : inputs(tensors) {};
    AsyncWorkCCL(const std::vector<at::Tensor>& inputs,
                 const std::vector<at::Tensor>& outputs) : inputs(inputs), outputs(outputs) {};

    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait() override;
    void abort() override;

    virtual void run() = 0;

    ~AsyncWorkCCL();

    std::vector<at::Tensor>& getOutputTensors()
    {
      return outputs;
    }

    std::vector<at::Tensor>& getInputTensors()
    {
      return inputs;
    }

    std::vector<at::Tensor> result() const override
    {
      TORCH_CHECK(outputs.size() == 1, "unexpected result size");
      return outputs;
    }

  public:
    ccl::communicator::coll_request_t req;
    /*
        keep copy of tensors to incrememt tensor reference counters
        while CCL operation is in progress
    */
    std::vector<at::Tensor> inputs;
    std::vector<at::Tensor> outputs;

    std::string debugName;

    friend class ProcessGroupCCL;
  };

  explicit ProcessGroupCCL(const std::shared_ptr<Store>& store, int rank, int size, const std::chrono::milliseconds& op_time_out);
  virtual ~ProcessGroupCCL();

  std::shared_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

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

  std::shared_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  std::shared_ptr<ProcessGroup::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

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

  std::shared_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  // create a new ProcessGroupCCL and initialize CCL if not initialized
  static std::shared_ptr<ProcessGroup> createProcessGroupCCL(
      const std::shared_ptr<Store>& store,
      int rank = -1,
      int size = -1,
      const std::chrono::milliseconds& op_time_out =
      std::chrono::milliseconds(OP_TIMEOUT_MILLIS));
  static const int64_t OP_TIMEOUT_MILLIS;
 public:

  static void cclInitOnce();
  static void cclFini();

  // Store that is used to exchange ccl kvs
  std::shared_ptr<Store> store_;
  std::chrono::milliseconds op_timeout_millis;

  // The kvs is unique ID among the processes.
  ccl::shared_ptr_class<ccl::kvs> kvs;

  // ID of this process group
  std::string processGroupID_;

  // Group Prefix and ID of this process group
  std::string groupPgID_;

  // Maintain all the communicators in process group.
  ccl::communicator comm;

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

  // processGroupID tracking
  static std::mutex pgTrackingLock_;

  static std::unordered_map<std::string, ssize_t> pgUniqueNCCLIDCnt_;

  static std::unordered_map<std::string, ssize_t> processGroupCounterMap_;
};


} // namespace c10d
