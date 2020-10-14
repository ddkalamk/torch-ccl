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

#include "ProcessGroupCCL.hpp"
#include "dispatch_stub.h"

#include <map>
#include <ATen/record_function.h>

namespace c10d
{
using torch_ccl::DispatchStub;

namespace {

static std::once_flag cclInitOnceFlag;
static std::mutex globalMutex;

void checkRank(int rank, int size)
{
  TORCH_CHECK((rank >= 0) && (rank < size), "unexpected rank");
}

} // namespace

const int64_t ProcessGroupCCL::OP_TIMEOUT_MILLIS = 10 * 1000;

void ProcessGroupCCL::cclFini()
{
  std::cout << "cclFini " << std::endl;
  DispatchStub::reset_all();
}

void ProcessGroupCCL::cclInitOnce()
{
    std::call_once(cclInitOnceFlag, []() {

      /* create CCL environment at once */
      auto &env = ccl::details::environment::instance();
      (void)env;

      if (std::atexit(ProcessGroupCCL::cclFini))
      {
          throw std::runtime_error("failed to register the CCL exit handler");
      }
  });
}

std::shared_ptr<ProcessGroup> ProcessGroupCCL::createProcessGroupCCL(
    const std::shared_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& op_time_out)
{
    cclInitOnce();

    printf("torch ccl create process group rank %d, size %d\n", rank, size);
    return std::make_shared<ProcessGroupCCL>(store, rank, size, op_time_out);
}

ProcessGroupCCL::ProcessGroupCCL(const std::shared_ptr<Store>& store, int rank, int size, const std::chrono::milliseconds& op_time_out)
    : ProcessGroup(rank, size), store_(store), op_timeout_millis(op_time_out),
      kvs([=](){
        ccl::shared_ptr_class<ccl::kvs> kvs;

        std::string storeKey = "ccl_kvs";

        // Rank 0 writes to the store as bcast
        if (rank == 0) {
          kvs = ccl::details::environment::instance().create_main_kvs();
          ccl::kvs::address_type main_addr = kvs->get_address();
          auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
          printf("torch ccl rank %d, send kvs addr\n", rank);
          store_->set(storeKey, ccl_kvs_addr);
        }
        else {
          printf("torch ccl rank %d, receiving kvs addr\n", rank);
          auto ccl_kvs_addr = store_->get(storeKey);
          printf("torch ccl rank %d, got kvs addr\n", rank);
          if (ccl_kvs_addr.size() != ccl::kvs::address_max_size) {
            throw std::runtime_error(
              "Unexpected ccl kvs addr from the store\n");
          }
          ccl::kvs::address_type main_addr;
          std::copy_n(std::make_move_iterator(ccl_kvs_addr.begin()),
                      ccl::kvs::address_max_size,
                      main_addr.begin());
          kvs = ccl::details::environment::instance().create_kvs(main_addr);
        }
        return kvs;
      }())
{
}

ProcessGroupCCL::~ProcessGroupCCL()
{
  std::cout << "Destroy the outstanding work here " << std::endl;
  std::cout << "Destroy the related comm here " << std::endl;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts)
{
  checkRank(opts.rootRank, getSize());
  auto work = DispatchStub::broadcast(tensors, opts, *this);

  // sync run
  work->run();
  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce(
  std::vector<at::Tensor>& tensors,
  const AllreduceOptions& opts)
{
  auto work = DispatchStub::allreduce(tensors, opts, *this);

  // sync run
  work->run();
  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support allreduce_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts)
{
  checkRank(opts.rootRank, getSize());

  auto work = DispatchStub::reduce(tensors, opts, *this);

  // sync run
  work->run();
  return work;
}


std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts)
{
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support allgather_base");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support allgather_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts)
{
#if 0
    RECORD_FUNCTION("pg::gather", std::vector<c10::IValue>({inputTensors[0]}));

    checkSingleTensor(inputTensors);

    if (rank_ != opts.rootRank)
    {
        TORCH_CHECK(outputTensors.size() == 0,
            "gather: number of output tensors should be 0 "
            "for non-root");
    }
    else
    {
        TORCH_CHECK(outputTensors.size() == 1,
            "gather: multi-GPU collective is not supported");

        TORCH_CHECK(static_cast<size_t>(size_) == outputTensors[0].size(),
            "gather: number of output tensors should equal "
            "to the world size");

        checkSameType(inputTensors[0], outputTensors[0]);
    }

    std::vector<size_t> sendCounts(size_, 0);
    std::vector<size_t> recvCounts(size_, 0);
    sendCounts[opts.rootRank] = inputTensors[0].numel();

    at::Tensor flatOutput;
    int64_t flatRecvCount = 0;
    bool isOutputFlat = false;

    if (rank_ == opts.rootRank)
    {
        isOutputFlat =
            computeLengthsAndCheckAndGetFlat(outputTensors[0],
                                             recvCounts, flatOutput, flatRecvCount);
        TORCH_CHECK(sendCounts[rank_] == recvCounts[rank_],
            "gather: send and recv count doesn't match");
    }
    else
    {
        flatOutput = at::empty({0}, inputTensors[0].options());
    }


    std::shared_ptr<ccl::request> req;

    {
        std::unique_lock<std::mutex> globalLock(globalMutex);
        CCL_CHECK(req = comm.alltoallv(inputTensors[0].data_ptr(),
                                        sendCounts,
                                        flatOutput.data_ptr(),
                                        recvCounts,
                                        cclDatatypes.at(flatOutput.scalar_type())));
    }

    std::vector<at::Tensor> gatherTensors;

    if (rank_ == opts.rootRank)
    {
        if (!isOutputFlat)
        {
            req->wait();

            auto flatOutputSplits =
                flatOutput.split_with_sizes(c10::IntArrayRef((int64_t*)recvCounts.data(),
                                            recvCounts.size()), 0);

            for (int i = 0; i < size_; i++)
            {
                outputTensors[0][i].view({-1}).copy_(flatOutputSplits[i]);
            }
        }
        else
        {
            gatherTensors.emplace_back(flatOutput);
            gatherTensors.emplace_back(inputTensors[0]);
        }
    }
    else
    {
        gatherTensors.emplace_back(inputTensors[0]);
    }

    std::string debugName = std::string("gather::sz:") + std::to_string(inputTensors[0].numel());

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(gatherTensors), std::move(debugName));
#endif
  TORCH_CHECK(false, "ProcessGroupCCL does not support allgather_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts)
{
#if 0
    RECORD_FUNCTION("pg::scatter", std::vector<c10::IValue>({outputTensors}));

    checkSingleTensor(outputTensors);

    if (rank_ != opts.rootRank)
    {
        TORCH_CHECK(inputTensors.size() == 0,
            "scatter: number of input tensors should be 0 "
            "for non-root");
    }
    else
    {
        TORCH_CHECK(inputTensors.size() == 1,
            "scatter: multi-GPU collective is not supported");

        TORCH_CHECK(static_cast<size_t>(size_) == inputTensors[0].size(),
            "scatter: number of input tensors should equal "
            "to the world size");

        checkSameType(outputTensors[0], inputTensors[0]);
    }

    std::vector<size_t> sendCounts(size_, 0);
    std::vector<size_t> recvCounts(size_, 0);
    recvCounts[opts.rootRank] = outputTensors[0].numel();

    at::Tensor flatInput;
    int64_t flatSendCount = 0;

    if (rank_ == opts.rootRank)
    {
        bool isInputFlat =
            computeLengthsAndCheckAndGetFlat(inputTensors[0],
                                             sendCounts, flatInput, flatSendCount);

        if (!isInputFlat)
        {
            auto flatInputSplits =
                flatInput.split_with_sizes(c10::IntArrayRef((int64_t*)sendCounts.data(),
                                           sendCounts.size()), 0);

            for (int i = 0; i < size_; i++)
            {
                flatInputSplits[i].copy_(inputTensors[0][i].view({-1}));
            }
        }
        TORCH_CHECK(recvCounts[rank_] == sendCounts[rank_],
            "scatter: send and recv count doesn't match");
    }
    else
    {
        flatInput = at::empty({0}, outputTensors[0].options());
    }

    std::shared_ptr<ccl::request> req;

    {
        std::unique_lock<std::mutex> globalLock(globalMutex);
        CCL_CHECK(req = comm.alltoallv(flatInput.data_ptr(),
                                        sendCounts,
                                        outputTensors[0].data_ptr(),
                                        recvCounts,
                                        cclDatatypes.at(flatInput.scalar_type())));
    }

    std::vector<at::Tensor> scatterTensors;
    scatterTensors.emplace_back(outputTensors[0]);
    if (rank_ == opts.rootRank)
        scatterTensors.emplace_back(flatInput);

    std::string debugName = std::string("scatter::sz:") + std::to_string(outputTensors[0].numel());

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(scatterTensors), std::move(debugName));
#endif
  TORCH_CHECK(false, "ProcessGroupCCL does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts)
{
    auto work = DispatchStub::alltoall_base(outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts, *this);
    // sync run
    work->run();
    return work;
  
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts)
{
    auto work = DispatchStub::alltoall(outputTensors, inputTensors, opts, *this);
    // sync run
    work->run();
    return work;

}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support send");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support recv");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support recvAnysource");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::barrier(
    const BarrierOptions& opts)
{
#if 0
    RECORD_FUNCTION("pg::barrier", std::vector<c10::IValue>());

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(comm.barrier());

    return std::make_shared<ProcessGroupCCL::WorkCCL>();
#endif
  TORCH_CHECK(false, "ProcessGroupCCL does not support recvAnysource");
}

} // namespace c10d
