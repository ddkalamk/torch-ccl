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
#include <sys/types.h>
#include <unistd.h>
#include <map>
#include <ATen/record_function.h>


namespace c10d
{

using torch_ccl::DispatchStub;

namespace {

static std::once_flag cclInitOnceFlag;

void checkRank(int rank, int size)
{
  TORCH_CHECK((rank >= 0) && (rank < size), "unexpected rank");
}

} // namespace

const int64_t ProcessGroupCCL::OP_TIMEOUT_MILLIS = 10 * 1000;

void ProcessGroupCCL::cclFini()
{
  DispatchStub::reset_all();
}

void ProcessGroupCCL::cclInitOnce()
{
  std::call_once(cclInitOnceFlag, []() {
    /* initial the at once */
    ccl::init();

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
  return std::make_shared<ProcessGroupCCL>(store, rank, size, op_time_out);
}

ProcessGroupCCL::ProcessGroupCCL(const std::shared_ptr<Store>& store, int rank, int size, const std::chrono::milliseconds& op_time_out)
    : ProcessGroup(rank, size), store_(store), op_timeout_millis(op_time_out),
      kvs([=](){
        ccl::shared_ptr_class<ccl::kvs> kvs;

        std::string storeKey = "ccl_kvs";

        // Rank 0 broadcast the bootstrap network information to other ranks
        if (rank == 0) {
          kvs = ccl::create_main_kvs();
          ccl::kvs::address_type main_addr = kvs->get_address();
          auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
          store_->set(storeKey, ccl_kvs_addr);
        }
        else {
          auto ccl_kvs_addr = store_->get(storeKey);
          if (ccl_kvs_addr.size() != ccl::kvs::address_max_size) {
            throw std::runtime_error(
              "Unexpected ccl kvs addr from the store\n");
          }
          ccl::kvs::address_type main_addr;
          std::copy_n(std::make_move_iterator(ccl_kvs_addr.begin()),
                      ccl::kvs::address_max_size,
                      main_addr.begin());
          kvs = ccl::create_kvs(main_addr);
        }
        return kvs;
      }())
{
}

ProcessGroupCCL::~ProcessGroupCCL()
{
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts)
{
  checkRank(opts.rootRank, getSize());
  auto work = DispatchStub::broadcast(tensors, opts, *this);

  work->run();
  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce(
  std::vector<at::Tensor>& tensors,
  const AllreduceOptions& opts)
{
  auto work = DispatchStub::allreduce(tensors, opts, *this);

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

  work->run();
  return work;
}


std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts)
{
  auto work = DispatchStub::allgather(outputTensors, inputTensors, opts, *this);

  
  work->run();
  return work;
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
  auto work = DispatchStub::gather(outputTensors, inputTensors, opts, *this);

  work->run();
  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts)
{
  auto work = DispatchStub::scatter(outputTensors, inputTensors, opts, *this);

  work->run();
  return work;
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

  work->run();
  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts)
{
  auto work = DispatchStub::alltoall(outputTensors, inputTensors, opts, *this);

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
 return DispatchStub::barrier(opts, *this);
}

} // namespace c10d
