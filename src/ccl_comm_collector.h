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

#pragma once

#include <c10/core/Device.h>
#include <oneapi/ccl.hpp>
#include <unordered_map>
#include "ProcessGroupCCL.hpp"

namespace torch_ccl {

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
  Comms(Comms &&other) : comms(std::move(other.comms)), streams(std::move(other.streams)) {}

  // Move assignable
  Comms &operator=(Comms &&other) {
    std::swap(comms, other.comms);
    std::swap(streams, other.streams);
    return *this;
  }

public:
  // The Communicators used by CCL
  ccl::vector_class<ccl::communicator> comms;
  // The streams used by CCL
  ccl::vector_class<ccl::stream> streams;
};

struct CCLMember {

  CCLMember() : kvs(nullptr) {};

  ccl::shared_ptr_class<ccl::kvs> get_kvs(int rank, c10d::Store& store);

  // ccl kvs to identify the community.
  ccl::shared_ptr_class<ccl::kvs> kvs;

  // The CCL communicator that the process group has cached.
  // The key is a list of devices that an operation is operating on
  // The devices are stored in a device sequence and the cache CCL
  // communicator is associated with this device sequence
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
  std::unordered_map<std::string, std::shared_ptr<torch_ccl::Comms>> ccl_comms;

};

}
