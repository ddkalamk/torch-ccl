/*
 * Copyright (c) 2020-2021, Intel Corporation
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

#include <ccl_comm_collector.h>
#include <core/DPCPPUtils.h>
#include <core/Context.h>

namespace torch_ccl {

//template <>
//void Comms<class DPCPP>::sync_streams(std::vector<at::Device> devices) {
//  for (size_t i = 0; i < devices.size(); ++i) {
//    ccl::stream& ccl_stream = streams[i];
//    cl::sycl::queue& communication_queue = ccl_stream.get_native();
//    cl::sycl::queue& computation_queue = at::dpcpp::getCurrentDPCPPStream(devices[i].index()).dpcpp_queue();
//    if (communication_queue != computation_queue) {
//      cl::sycl::event barrier_event = computation_queue.submit_barrier();
//      // The CCL algorithm should wait the computation to be finished.
//      communication_queue.submit_barrier({barrier_event});
//    }
//  }
//}

}

