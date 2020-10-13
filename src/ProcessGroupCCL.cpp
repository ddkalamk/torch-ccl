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
#if 0
    RECORD_FUNCTION("pg::alltoall_base", std::vector<c10::IValue>({inputTensor, outputTensor}));

    checkSingleTensorHelper(inputTensor);
    checkSingleTensorHelper(outputTensor);

    std::shared_ptr<ccl::request> req;

    if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0)
    {
        // We can use alltoall
        TORCH_CHECK(outputTensor.numel() == inputTensor.numel() &&
                    outputTensor.scalar_type() == inputTensor.scalar_type(),
                    "alltoall_base: tensors are not equal in size or data type");

        TORCH_CHECK(outputTensor.size(0) % size_ == 0,
            "alltoall_base: tensor's dim 0 does not divide equally across group size");

        {
            std::unique_lock<std::mutex> globalLock(globalMutex);
            CCL_CHECK(req = comm.alltoall(inputTensor.data_ptr(),
                                           outputTensor.data_ptr(),
                                           (size_t)outputTensor.numel() / comm.size(),
                                           cclDatatypes.at(outputTensor.scalar_type())));
        }
    }
    else
    {
        // Need alltoallv
        checkSplitSizes(inputSplitSizes, inputTensor, size_);
        checkSplitSizes(outputSplitSizes, outputTensor, size_);

        std::vector<size_t> sendCounts(size_);
        std::vector<size_t> recvCounts(size_);

        // inLen or outLen can be 0 so we need explicit flag
        bool inputSplitsEqual = inputSplitSizes.size() == 0;
        bool outputSplitsEqual = outputSplitSizes.size() == 0;

        size_t inLen = inputTensor.numel();
        size_t outLen = outputTensor.numel();
        if (inLen) inLen /= (inputSplitsEqual ? size_ : inputTensor.size(0));
        if (outLen) outLen /= (outputSplitsEqual ? size_ : outputTensor.size(0));

        for (int i = 0; i < size_; i++)
        {
            sendCounts[i] = (inputSplitsEqual ? inLen : inputSplitSizes[i] * inLen);
            recvCounts[i] = (outputSplitsEqual ? outLen : outputSplitSizes[i] * outLen);
        }

        {
            std::unique_lock<std::mutex> globalLock(globalMutex);
            CCL_CHECK(req = comm.alltoallv(inputTensor.data_ptr(),
                                            sendCounts,
                                            outputTensor.data_ptr(),
                                            recvCounts,
                                            cclDatatypes.at(outputTensor.scalar_type())));
        }
    }

    auto a2aTensors = std::vector<at::Tensor> { inputTensor, outputTensor };
    std::string debugName = std::string("alltoall_base::sz:") +
        std::to_string((inputTensor.numel() + outputTensor.numel()) / (2 * size_));

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(a2aTensors), std::move(debugName));
#endif

  TORCH_CHECK(false, "ProcessGroupCCL does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts)
{
#if 0
    RECORD_FUNCTION("pg::alltoall", std::vector<c10::IValue>());

    TORCH_CHECK(inputTensors.size() == (size_t)size_,
        "alltoall: number of input tensors are not equal to group size");

    TORCH_CHECK(outputTensors.size() == (size_t)size_,
        "alltoall: number of output tensors are not equal to group size");

    checkSameType(outputTensors[0], inputTensors);
    checkSameType(inputTensors[0], outputTensors);

    std::vector<size_t> sendCounts(size_);
    std::vector<size_t> recvCounts(size_);

    at::Tensor flatInput;
    at::Tensor flatOutput;

    int64_t flatSendCount;
    int64_t flatRecvCount;

    bool isInputFlat =
        computeLengthsAndCheckAndGetFlat(inputTensors, sendCounts, flatInput, flatSendCount);

    bool isOutputFlat =
        computeLengthsAndCheckAndGetFlat(outputTensors, recvCounts, flatOutput, flatRecvCount);

    if (!isInputFlat)
    {
        auto flatInputSplits =
            flatInput.split_with_sizes(c10::IntArrayRef((int64_t*)sendCounts.data(),
                                       sendCounts.size()), 0);

        for (int i = 0; i < size_; i++)
        {
            flatInputSplits[i].copy_(inputTensors[i].view({-1}));
        }
    }

    std::shared_ptr<ccl::request> req;

    {
        std::unique_lock<std::mutex> globalLock(globalMutex);
        CCL_CHECK(req = comm.alltoallv(flatInput.data_ptr(),
                                        sendCounts,
                                        flatOutput.data_ptr(),
                                        recvCounts,
                                        cclDatatypes.at(flatOutput.scalar_type())));
    }

    std::vector<at::Tensor> a2aTensors;

    if (!isOutputFlat)
    {
        req->wait();

        auto flatOutputSplits =
            flatOutput.split_with_sizes(c10::IntArrayRef((int64_t*)recvCounts.data(),
                                        recvCounts.size()), 0);

        for (int i = 0; i < size_; i++)
        {
            outputTensors[i].view({-1}).copy_(flatOutputSplits[i]);
        }
    }
    else
    {
        a2aTensors.emplace_back(flatOutput);
        a2aTensors.emplace_back(flatInput);
    }

    std::string debugName = std::string("alltoall::sz:") +
        std::to_string((flatSendCount + flatRecvCount) / (2 * size_));

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(a2aTensors), std::move(debugName));
#endif
  TORCH_CHECK(false, "ProcessGroupCCL does not support send");
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
