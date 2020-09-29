//
// Created by johnlu on 2020/9/24.
//

#include "dispatch_stub.h"

namespace torch_ccl {

static DispatchStub default_stubs;
constexpr DispatchStub* default_stubs_addr = &default_stubs;

DispatchStub* DispatchStub::stubs_[to_int(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)] = {
  [0 ... (to_int(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES) - 1)] = default_stubs_addr
};

void DispatchStub::register_ccl_stub(c10::DeviceType dev_type, DispatchStub* stub) {
  std::cout <<"register backend: " << dev_type << std::endl;
  stubs_[to_int(dev_type)] = stub;
}

}