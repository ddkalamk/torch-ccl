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
#include "ccl_comm_collector.h"

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
class ProcessGroupCCL : public ProcessGroup
{

public:

  class AsyncWorkCCL : public ProcessGroup::Work {
  public:
    AsyncWorkCCL(const std::vector<at::Tensor>& inputs,
                 const std::vector<at::Tensor>& outputs) : inputs(inputs), outputs(outputs) {};

    virtual void run() = 0;

    std::vector<at::Tensor>& getOutputTensors()
    {
      return outputs;
    }

    std::vector<at::Tensor>& getInputTensors()
    {
      return inputs;
    }

//    std::vector<at::Tensor> result() const override
//    {
//      TORCH_CHECK(outputs.size() == 1, "unexpected result size");
//      return outputs;
//    }

  public:
    /*
        keep copy of tensors to increment tensor reference counters
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

  // ID of this process group
  std::string processGroupID_;

  // Group Prefix and ID of this process group
  std::string groupPgID_;

  ccl::shared_ptr_class<ccl::kvs> kvs;

  // processGroupID tracking
  static std::mutex pgTrackingLock_;

  static std::unordered_map<std::string, ssize_t> pgUniqueNCCLIDCnt_;

  static std::unordered_map<std::string, ssize_t> processGroupCounterMap_;
};


} // namespace c10d
