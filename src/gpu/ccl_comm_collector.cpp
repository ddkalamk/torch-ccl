//
// Created by john on 2020/12/28.
//

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

