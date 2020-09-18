#include "../ProcessGroupCCL.hpp"

#include <map>
#include <ccl.h>
#include <core/Stream.h>
#include <core/Memory.h>

#define CCL_CHECK_AND_THROW(result, diagnostic)   \
  do {                                            \
      if (result != ccl_status_success)           \
      {                                           \
          throw ccl::ccl_error(diagnostic);       \
      }                                           \
  } while (0);


#define CCL_CHECK(cmd)                                                   \
  do {                                                                   \
    try {                                                                \
        cmd;                                                             \
    }                                                                    \
    catch (ccl::ccl_error& e) {                                          \
        std::string err = "CCL error in: " + std::string(__FILE__) +     \
            ":" + std::to_string(__LINE__) +                             \
            ", with error message: " + e.what();                         \
        fprintf(stderr, "\n%s\n", err.c_str());                          \
        throw std::runtime_error(err);                                   \
    }                                                                    \
    catch (...) {                                                        \
        std::string err = "unknown error in: " + std::string(__FILE__) + \
            ":" + std::to_string(__LINE__);                              \
        fprintf(stderr, "\n%s\n", err.c_str());                          \
        throw std::runtime_error(err);                                   \
    }                                                                    \
  } while (0)


#define CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      /*AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, char, __VA_ARGS__)*/          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

namespace c10d {
namespace {
// Op mapping
std::map<ReduceOp, ccl::reduction> cclOp = {
        {ReduceOp::MIN, ccl::reduction::min},
        {ReduceOp::MAX, ccl::reduction::max},
        {ReduceOp::SUM, ccl::reduction::sum},
        {ReduceOp::PRODUCT, ccl::reduction::prod},
};

//template <typename buffer_data_type, int dims = 1>
//cl::sycl::buffer<buffer_data_type, dims> make_buffer(void* virtual_ptr) {
//  static_assert(dims == 1, "buffer dims is not 1");
//
//  //reinterpret the buffer to the required type.
//  auto buf = at::dpcpp::dpcppGetBufferMap().template get_buffer<uint8_t>(virtual_ptr);
//  auto range = cl::sycl::range<dims>(buf.get_size()/sizeof(buffer_data_type));
//  return buf.template reinterpret<buffer_data_type, dims>(range);
//}

// Check that all `tensors' have the same device and type and shape and
// are distributed across distinct GPUs if these are GPU tensors.
c10::DeviceType check_tensors_properties(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    throw std::runtime_error("Tensor list must be nonempty");
  }
  int device_count;
  at::dpcpp::dpcppGetDeviceCount(&device_count);
  if (tensors.size() > static_cast<size_t>(device_count)) {
    throw std::runtime_error(
      "Tensor list mustn't be larger than the number of available GPUs");
  }

  const auto& first = tensors.front();
  auto dev_type = first.device().type();

  // Set for ensuring that tensors are on separate devices.
  std::unordered_set<decltype(first.get_device())> usedDevices;
  usedDevices.reserve(tensors.size());

  for (const auto& t : tensors) {
    if (t.is_sparse()) {
      throw std::runtime_error("Tensors must be dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      throw std::runtime_error("Tensors must have identical type");
    }
    if (t.sizes() != first.sizes()) {
      throw std::runtime_error("Tensors must have identical size");
    }
    if (!t.is_contiguous()) {
      throw std::runtime_error("Tensors must be contiguous");
    }
    if (dev_type != t.device().type()) {
      throw std::runtime_error("Tensors must be on the same device type");
    }
    const auto inserted = usedDevices.insert(t.get_device()).second;
    if (!inserted) {
      throw std::runtime_error("Tensors must be on distinct devices");
    }
  }

  return dev_type;
}

// Get the list of devices from list of tensors
std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.device());
  }
  return res;
}

// Get the list of devices from list of tensors
std::vector<at::Device> getDeviceList(const std::vector<std::vector<at::Tensor>>& tensor_lists) {
  std::vector<at::Device> res;
  res.reserve(tensor_lists.size());
  for (auto& tensors : tensor_lists) {
    res.push_back(tensors[0].device());
  }
  return res;
}

// Get the device type from list of tensors
at::Device getDeviceType(const std::vector<at::Tensor>& tensors) {
  return tensors[0].device();
}

// Get the list of devices from list of tensors
at::Device getDeviceType(const std::vector<std::vector<at::Tensor>>& tensor_lists) {
  return tensor_lists[0][0].device();
}

// Get the deviceList String from the list of devices
std::string getKeyFromDevices(const std::vector<at::Device>& devices) {
  std::string deviceList;
  for (auto& device : devices) {
    if (deviceList.empty()) {
      deviceList = std::to_string(device.index());
    } else {
      deviceList += "," + std::to_string(device.index());
    }
  }
  return deviceList;
}

// [Sync Streams] Helper that lets the input cclStreams to wait for the current
// stream. CCL communications run on cclStreams, but input tensors are
// allocated on different streams (i.e., current streams). Communications on
// cclStreams cannot start before pending input tensor ops on current streams
// finish. Otherwise, ops on two streams might read/write same tensors
// concurrently.
//
#if 0
// The synchronization above alone is not enough. We also need to make sure
// input tensors are not freed before their usages on cclStreams finish. This
// can be achieved by calling c10::cuda::CUDACachingAllocator::recordStream,
// which remembers the usage stream (ncclStream), creates an event on the usage
// stream when GC attempts to free the input tensor, and delays GC until that
// event is done.
#endif
void syncStreams(
        const std::vector<at::Device>& devices,
        /*std::vector<at::cuda::CUDAEvent>& ncclEvents,*/
        std::vector<ccl::stream_t >& cclStreams) {
//  for (size_t i = 0; i < devices.size(); ++i) {
//    at::cuda::CUDAStream& ncclStream = ncclStreams[i];
//    at::cuda::CUDAEvent& ncclEvent = ncclEvents[i];
//    ncclEvent.record(at::cuda::getCurrentCUDAStream(devices[i].index()));
//    ncclEvent.block(ncclStream);
//  }
}


// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
  std::vector<std::vector<at::Tensor>>& tensor_lists,
  std::vector<at::Tensor>& other,
  size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    throw std::runtime_error(
      "Tensor list operands to scatter/gather must have the same length");
  }
  const auto num_devices = tensor_lists.size();

  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (auto i = size_t{}; i < num_devices; ++i) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      throw std::runtime_error(
        "Tensor list input to scatter/gather must match number of collective"
        " participants");
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      throw std::runtime_error(
        "Corresponding input/output tensors to scatter/gather must all reside"
        " on the same device");
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        throw std::runtime_error(
          "All tensor operands to scatter/gather must have the same size");
      }
    }
    // Flatten the tensors (from all ranks) into a single big tensor.
    flattened[i] = newLikeFlat(tensor_lists, i);
  }
  return flattened;
}

} // namespace

const int64_t ProcessGroupCCL::OP_TIMEOUT_MILLIS = 10 * 1000;

std::vector<ccl::communicator_t>& ProcessGroupCCL::getCCLComm(
  const std::string& devicesKey,
  const std::vector<at::Device>& devices) {
  // Sanity check
  if (devicesKey.empty()) {
    throw std::runtime_error(
      "Not able to create/get the CCL Communicator since "
      "the devices are empty ");
  }

  assert(devices.size() == 1 && "device size must be 1");

  for (auto& device : devices) {
    used_gpu_device_idxs.insert(device.index());
  }

  if (gpu_comms.find(devicesKey) != gpu_comms.end()) {
    // Reuse the cached communicator if there is one.
    return gpu_comms[devicesKey]->gpu_comms;
  }

  auto g_comm = ccl::environment::instance().create_communicator();
  std::vector<ccl::communicator_t> gcomms;
  gcomms.push_back(std::move(g_comm));
  ccl::communicator_t null_comm;

  std::vector<ccl::stream_t > ccl_streams;
  ccl_streams.reserve(devices.size());

  for(size_t i = 0; i < devices.size(); ++i)
  {
    /* create SYCL stream */
    auto q = at::dpcpp::getCurrentDPCPPStream(devices[i].index()).dpcpp_queue();
    ccl_streams.push_back(ccl::environment::instance().create_stream(q));
  }

  auto comms_collector = std::make_shared<class CCLCommsCollector>(std::move(null_comm), std::move(gcomms));

  // Move the CCL resource to cache
  gpu_comms.emplace(devicesKey, std::move(comms_collector));
  gpu_streams.emplace(devicesKey, std::move(ccl_streams));
#if 0
  // CCL communicator not cached, create a new gpu group
  ccl::comm_group_t group_gomm = ccl::environment::instance().create_comm_group(devices.size(),
                                                                               devices.size() * size_,
                                                                               global_comm);
  // create device communicator attributes
  ccl::shared_comm_device_attr_t my_device_comm_attr = ccl::environment::instance().create_device_comm_attr(ccl::comm_attr{});
  // set preferred device topology (OPTIONAL)
//  my_device_comm_attr->set_value<ccl_device_preferred_topology>(ccl::device_topology_type::allied_process_group_ring);
//  std::cout << "Create device communicators, expected count: " << devices.size()
//            << ", preferred topology: " << my_device_comm_attr->get_value<ccl_device_preferred_topology>() << std::endl;
  // Create communicators (auto rank balancing, based on ids): container based API
  std::vector<ccl::device_communicator_t> gpu_comms = group_gomm->create_communicators(devices,
                                                                              my_device_comm_attr);

  std::vector<ccl::stream_t > streamVal;
  streamVal.reserve(devices.size());

  std::stringstream ss;

  for(size_t i = 0; i < devices.size(); ++i)
  {
    /* create SYCL stream */
    auto q = c10::sycl::getCurrentSYCLStream(devices[i].index()).sycl_queue();
    streamVal.push_back(ccl::environment::instance().create_stream(ccl::stream_type::device, &q));
  }
  std::cout << ss.str() <<  std::endl;

  ccl_comm_t cclComm = std::make_shared<class CCLCommsCollector>(std::move(group_gomm), std::move(gpu_comms));

  // Move the CCL resource to cache
  gpu_comms.emplace(devicesKey, std::move(cclComm));
  gpu_streams.emplace(devicesKey, std::move(streamVal));
#endif
  return gpu_comms[devicesKey]->gpu_comms;
}

template <typename fn, typename pre_process, typename post_process, typename input_t, typename output_t>
std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::collective_cpu(
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun,
  pre_process pre,
  post_process post) {
  if (inputs.size() > 1 || outputs.size() > 1) {
    throw std::runtime_error(
      "CCL process group does not support multi CPU tensor collectives");
  }
  pre(cpu_streams);

  auto req = fun(inputs[0], outputs[0], global_comm, cpu_streams[0]);

  post(cpu_streams);

  return std::make_shared<ProcessGroupCCL::WorkCCL>(req);
}

template <typename fn, typename pre_process, typename post_process, typename input_t, typename output_t>
std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::collective_gpu(
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun,
  pre_process pre,
  post_process post) {
  const auto devices = getDeviceList(inputs);
  const auto key = getKeyFromDevices(devices);
  auto& comms_collector = getCCLComm(key, devices);

  // First let CCL streams wait for computing kernel on the input tensors's finished.
  syncStreams(devices,gpu_streams[key]);

  // Work itself will create the CUDA events on all GPUs of tensors
  auto work = std::make_shared<ProcessGroupCCL::WorkCCL>(devices);

  pre(gpu_streams[key]);

  for (size_t i = 0; i < inputs.size(); ++i) {
    ccl::stream_t& cclStream = gpu_streams[key][i];
    auto req = fun(inputs[i], outputs[i], comms_collector[i], cclStream);
    work->requests_[i] = std::move(req);
  }

  post(gpu_streams[key]);

  return work;

}

template <typename fn, typename pre_process, typename post_process, typename input_t, typename output_t>
std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::collective(
        std::vector<input_t>& inputs,
        std::vector<output_t>& outputs,
        fn fun,
        pre_process pre,
        post_process post) {
  auto dev_type = getDeviceType(inputs).type();
  if (dev_type == c10::DeviceType::DPCPP) {
    return collective_gpu(inputs, outputs, fun, pre, post);
  }
  else if (dev_type == c10::DeviceType::CPU) {
    return collective_cpu(inputs, outputs, fun, pre, post);
  }
  else {
    throw std::runtime_error("OCCL doesn't support input device ");
  }
}


template <typename fn, typename input_t, typename output_t>
std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::collective(
        std::vector<input_t>& inputs,
        std::vector<output_t>& outputs,
        fn fun) {
  return collective(
          inputs,
          outputs,
          fun,
          [](std::vector<ccl::stream_t>&) {},
          [](std::vector<ccl::stream_t>&) {});
}

ProcessGroupCCL::WorkCCL::~WorkCCL() {
  std::unique_lock<std::mutex> lock(mutex_);
  for(auto& request_ : requests_) {
    if (request_.get()) {
      request_.reset();
    }
  }
}

bool ProcessGroupCCL::WorkCCL::isCompleted() {
  std::unique_lock<std::mutex> lock(mutex_);
  for(auto& request : requests_) {
    bool flag;
    CCL_CHECK( flag = request->test());
    if (!flag) {
      return false;
    }
  }

  return true;
}

bool ProcessGroupCCL::WorkCCL::isSuccess() const {
  std::unique_lock<std::mutex> lock(mutex_);
//  for(auto& request : requests_) {
//  }
  return true;
}

bool ProcessGroupCCL::WorkCCL::wait() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    for(auto& request : requests_) {
      CCL_CHECK(request->wait());
    }
  }
  auto success = isSuccess();
  if (!success) { throw std::runtime_error("request is not success"); }
  return success;
}

std::mutex ProcessGroupCCL::pg_global_mutex;
std::once_flag ProcessGroupCCL::init_flag;
ccl::coll_attr ProcessGroupCCL::attr;

void ProcessGroupCCL::cclExit() {
  std::unique_lock<std::mutex> globalLock(pg_global_mutex);
  // ccl clean up
}

void ProcessGroupCCL::cclInit() {
  // Initialize CCL environment
  std::call_once(init_flag, []() {
//  Maybe we can put the MPI environ here
//  setenv("CCL_PM_TYPE", "resizable", 1);
//  setenv("CCL_KVS_IP_EXCHANGE", "env", 1);
//  setenv("CCL_KVS_IP_PORT", "4789", 1);
//  setenv("CCL_VECTOR_ALLGATHERV", "1", 1);

//    attr.to_cache = 0;
//    ccl_status_t status = ccl_init();
//    CCL_CHECK_AND_THROW(status, "failed to initialize ccl");

    // register exit handler
    if (std::atexit(ProcessGroupCCL::cclExit)) {
      throw std::runtime_error("Fail to register the CCL exit handler");
    }
  });
}

std::shared_ptr<ProcessGroup> ProcessGroupCCL::createProcessGroupCCL(const std::shared_ptr<Store>& store,
                                                                     int rank,
                                                                     int size,
                                                                     const std::chrono::milliseconds& op_time_out) {
  cclInit();
  return std::make_shared<ProcessGroupCCL>(rank, size);
}

ProcessGroupCCL::ProcessGroupCCL(int rank, int size)
        : ProcessGroup(rank, size),
          global_comm(ccl::environment::instance().create_communicator()) {
  cpu_streams.emplace_back(ccl::stream_t());
  std::cout<< "create occl pg rank " << global_comm->rank() << " size " << global_comm->size() << std::endl;
}

ProcessGroupCCL::~ProcessGroupCCL() {}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::broadcast(
        std::vector<at::Tensor>& tensors,
        const BroadcastOptions& opts) {
  auto dev_type = check_tensors_properties(tensors);

  return collective(tensors, tensors,
                    [&](at::Tensor input,
                        at::Tensor output,
                        ccl::communicator_t & comm,
                        ccl::stream_t& stream) {
                          auto count = input.numel();
                          auto coll_attr = attr;
#ifdef USE_CACHE
                          coll_attr.match_id = tensorName.c_str();
#endif
                          const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
                          ccl::communicator::coll_request_t ret_req;

                          CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "broadcast", [&] {
                            if (dev_type == c10::DeviceType::DPCPP) {
                              auto input_buf = at::dpcpp::make_buffer<scalar_t>(input.data_ptr());
                              std::cout<< "broadcast pg rank " << global_comm->rank() << " size " << global_comm->size() << " count " << count << " buffer count " << input_buf.get_count() << std::endl;
                              CCL_CHECK(ret_req = comm->bcast(
                                input_buf,
                                (size_t)count,
                                root,
                                &coll_attr,
                                stream));
                            }
                            else {
                              CCL_CHECK(ret_req = comm->bcast(
                                static_cast<scalar_t*>(input.data_ptr()),
                                (size_t)count,
                                root,
                                &coll_attr));
                            }
                          });
                          return ret_req;
                        });
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce(
        std::vector<at::Tensor>& tensors,
        const AllreduceOptions& opts) {
  auto dev_type = check_tensors_properties(tensors);

  return collective(tensors, tensors,
                    [&](at::Tensor& input,
                        at::Tensor& output,
                        ccl::communicator_t & comm,
                        ccl::stream_t& stream) {
                          auto count = input.numel();
                          auto coll_attr = attr;
#ifdef USE_CACHE
                          coll_attr.match_id = tensorName.c_str();
#endif
                          auto reduce_op = cclOp.at(opts.reduceOp);
                          ccl::communicator::coll_request_t ret_req;

                          CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "allreduce", [&] {
                            if (dev_type == c10::DeviceType::DPCPP) {
                              auto input_buf = at::dpcpp::make_buffer<scalar_t>(input.data_ptr());
                              auto output_buf = at::dpcpp::make_buffer<scalar_t>(output.data_ptr());

                              CCL_CHECK(ret_req = comm->allreduce(
                                input_buf,
                                output_buf,
                                (size_t)count,
                                reduce_op,
                                &coll_attr,
                                stream));
                            }
                            else {
                              CCL_CHECK(ret_req = comm->allreduce(
                                static_cast<scalar_t*>(input.data_ptr()),
                                static_cast<scalar_t*>(output.data_ptr()),
                                (size_t)count,
                                reduce_op,
                                &coll_attr));
                            }
                          });
                          return ret_req;
                        });
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce(
        std::vector<at::Tensor>& tensors,
        const ReduceOptions& opts) {
  auto dev_type = check_tensors_properties(tensors);

  return collective(tensors, tensors,
                    [&](at::Tensor& input,
                        at::Tensor& output,
                        ccl::communicator_t & comm,
                        ccl::stream_t& stream) {
                      auto count = input.numel();
                      auto coll_attr = attr;
#ifdef USE_CACHE
                      coll_attr.match_id = tensorName.c_str();
#endif
                      const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
                      auto reduce_op = cclOp.at(opts.reduceOp);
                      ccl::communicator::coll_request_t ret_req;

                      CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "reduce", [&] {
                        if (dev_type == c10::DeviceType::DPCPP) {
                          auto input_buf = at::dpcpp::make_buffer<scalar_t>(input.data_ptr());
                          auto output_buf = at::dpcpp::make_buffer<scalar_t>(output.data_ptr());

                          CCL_CHECK(ret_req = comm->reduce(
                            input_buf,
                            output_buf,
                            (size_t)count,
                            reduce_op,
                            root,
                            &coll_attr,
                            stream));
                        }
                        else {
                          CCL_CHECK(ret_req = comm->reduce(
                            static_cast<scalar_t*>(input.data_ptr()),
                            static_cast<scalar_t*>(output.data_ptr()),
                            (size_t)count,
                            reduce_op,
                            root,
                            &coll_attr));
                        }
                      });
                      return ret_req;
                    });
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather(
        std::vector<std::vector<at::Tensor>>& output_tensors,
        std::vector<at::Tensor>& input_tensors,
        const AllgatherOptions& opts) {
  auto dev_type = check_tensors_properties(input_tensors);

  if (dev_type == c10::DeviceType::DPCPP) {
    // flatten the output tensor for sycl_buffer. TODO: use the USM version.
    auto output_flattened =
      flatten_for_scatter_gather(output_tensors, input_tensors, size_);
    check_tensors_properties(output_flattened);

    return collective(input_tensors, output_flattened,
                      [&](at::Tensor& input,
                          at::Tensor& output,
                          ccl::communicator_t & comm,
                          ccl::stream_t& stream) {
                            auto send_count = input.numel();
                            std::vector<size_t> recv_count(this->size_, send_count);
                            auto coll_attr = attr;
#ifdef USE_CACHE
                            coll_attr.match_id = tensorName.c_str();
#endif
                            ccl::communicator::coll_request_t ret_req;

                            CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "allgather", [&] {

                              std::cout<< "allgather pg rank " << global_comm->rank() << " size " << global_comm->size() << std::endl;
                              auto send_buf = at::dpcpp::make_buffer<scalar_t>(input.data_ptr());
                              auto recv_buf = at::dpcpp::make_buffer<scalar_t>(output.data_ptr());

                              CCL_CHECK(ret_req = comm->allgatherv(
                                send_buf,
                                (size_t)send_count,
                                recv_buf,
                                (size_t*)recv_count.data(),
                                &coll_attr,
                                stream));
                            });

                            return ret_req;
                          },
                      [&](std::vector<ccl::stream_t>& ccl_stream) {},
                      [&](std::vector<ccl::stream_t>& ccl_stream) {
                        // Copy the flattened output tensors to the outputs.
                        for (size_t i = 0; i < output_tensors.size(); ++i) {
//                          at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
                          for (size_t j = 0; j < output_tensors[0].size(); ++j) {
                            // See [Sync Streams].
//                            c10::cuda::CUDACachingAllocator::recordStream(
//                              outputTensors[i][j].storage().data(), ncclStreams[i]);

                            output_tensors[i][j].copy_(output_flattened[i][j], true);
                          }
                        }
                      });
  }
  else {
    return collective(input_tensors, output_tensors,
                      [&](at::Tensor& input,
                          std::vector<at::Tensor>& outputs,
                          ccl::communicator_t & comm,
                          ccl::stream_t& stream) {
                        auto send_count = input.numel();
                        std::vector<size_t> recv_count(this->size_, send_count);
                        auto coll_attr = attr;
#ifdef USE_CACHE
                        coll_attr.match_id = tensorName.c_str();
#endif
                        ccl::communicator::coll_request_t ret_req;

                        CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "allgather", [&] {
                          std::vector<scalar_t *> recv_bufs;
                          std::transform(outputs.begin(), outputs.end(),
                                         std::back_inserter(recv_bufs),
                                         [](const at::Tensor &t) { return t.data_ptr<scalar_t>(); });
                          coll_attr.vector_buf = 1;
                          CCL_CHECK(ret_req = comm->allgatherv(
                            static_cast<scalar_t *>(input.data_ptr()),
                            (size_t) send_count,
                            (scalar_t * )(recv_bufs.data()),// This cast is safe. The vector buffer is used.
                            (size_t *) recv_count.data(),
                            &coll_attr));
                        });

                        return ret_req;
                      });
  }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::barrier(
        const BarrierOptions& opts) {
  // call the barrier in with global communicator.
  CCL_CHECK(global_comm->barrier());

  return std::make_shared<ProcessGroupCCL::WorkCCL>();
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::gather(
        std::vector<std::vector<at::Tensor>>& output_tensors_list,
        std::vector<at::Tensor>& input_tensors,
        const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    throw std::invalid_argument("ProcessGroupCCL::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);

  if (getRank() == opts.rootRank) {
    if (output_tensors_list.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (output_tensors_list[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << output_tensors_list[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = input_tensors[0].options();
    const auto& sizes = input_tensors[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, output_tensors_list[0], options, sizes);
  } else {
    if (output_tensors_list.size() != 0) {
      invalidArgument("requires empty output on non-root");
    }
  }

  auto dev_type = check_tensors_properties(input_tensors);

  if (dev_type == c10::DeviceType::DPCPP) {
  }
  else {
    if (rank_ == opts.rootRank)
    {
      return collective(input_tensors, output_tensors_list,
                        [&](at::Tensor& input,
                           std::vector<at::Tensor>& output,
                           ccl::communicator_t & comm,
                           ccl::stream_t& stream) {
                         std::vector<size_t> send_count(size_, 0);
                         std::vector<size_t> recv_count(size_, input.numel());
                         auto coll_attr = attr;
                         send_count[opts.rootRank] = input.numel();
                         ccl::communicator::coll_request_t ret_req;

                         CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "gather", [&] {
                           std::vector<scalar_t *> recv_bufs;
                           std::transform(output.begin(), output.end(),
                                          std::back_inserter(recv_bufs),
                                          [](const at::Tensor &t) { return t.data_ptr<scalar_t>(); });
                           std::cout<< "gather root rank " << global_comm->rank() << " size " << global_comm->size() <<
                            " recv buffer " << recv_bufs << " recv count " << recv_count << std::endl;
                           CCL_CHECK(ret_req = comm->alltoallv(
                             input.data_ptr<scalar_t>(),
                             (size_t *) send_count.data(),
                             (scalar_t *)(recv_bufs.data()),// This cast is safe. The vector buffer is used.
                             (size_t *) recv_count.data(),
                             &coll_attr));

                           //have to add wait here.
                           ret_req->wait();
                         });
                         return ret_req;
                       });
    }
    else {
      std::vector<at::Tensor> dummy_output{at::empty({0}, input_tensors[0].options())};
      return collective(input_tensors, dummy_output,
                        [&](at::Tensor& input,
                            at::Tensor& output,
                            ccl::communicator_t & comm,
                            ccl::stream_t& stream) {
                          std::vector<size_t> send_count(size_, 0);
                          std::vector<size_t> recv_count(size_, 0);
                          auto coll_attr = attr;
                          send_count[opts.rootRank] = input.numel();
                          ccl::communicator::coll_request_t ret_req;

                          CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "gather", [&] {
                            std::cout<< "gather slave rank " << global_comm->rank() << " size " << global_comm->size() << std::endl;
                            CCL_CHECK(ret_req = comm->alltoallv(
                              input.data_ptr<scalar_t>(),
                              (size_t *) send_count.data(),
                              output.data_ptr<scalar_t>(),
                              (size_t *) recv_count.data(),
                              &coll_attr));
                          });
                          ret_req->wait();
                          return ret_req;
                        });
    }

  }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::scatter(
        std::vector<at::Tensor>& output_tensors,
        std::vector<std::vector<at::Tensor>>& input_tensors_list,
        const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    throw std::invalid_argument("ProcessGroupCCL::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);

  if (getRank() == opts.rootRank) {
    if (input_tensors_list.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors";
      invalidArgument(ss.str());
    } else if (input_tensors_list[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect input list size " << input_tensors_list[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }
    const auto& options = output_tensors[0].options();
    const auto& sizes = output_tensors[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, input_tensors_list[0], options, sizes);
  } else {
    if (input_tensors_list.size() != 0) {
      invalidArgument("requires empty input on non-root");
    }
  }

  auto dev_type = check_tensors_properties(output_tensors);

  if (dev_type == c10::DeviceType::DPCPP) {
  }
  else {
    if (rank_ == opts.rootRank)
    {
      return collective(input_tensors_list, output_tensors,
                        [&](std::vector<at::Tensor>& input,
                            at::Tensor& output,
                            ccl::communicator_t & comm,
                            ccl::stream_t& stream) {
                          std::vector<size_t> send_count(size_, 0);
                          std::vector<size_t> recv_count(size_, 0);
                          auto coll_attr = attr;
                          recv_count[opts.rootRank] = output.numel();
                          ccl::communicator::coll_request_t ret_req;

                          CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(output.scalar_type(), "scatter", [&] {
                            std::vector<scalar_t *> send_bufs;
                            std::transform(input.begin(), input.end(),
                                           std::back_inserter(send_bufs),
                                           [](const at::Tensor &t) { return t.data_ptr<scalar_t>(); });
                            std::cout<< "scatter root rank " << global_comm->rank() << " size " << global_comm->size() <<
                                     " recv buffer " << send_bufs << " recv count " << recv_count << std::endl;
                            CCL_CHECK(ret_req = comm->alltoallv(
                              (scalar_t *)(send_bufs.data()),// This cast is safe. The vector buffer is used.
                              (size_t *) send_count.data(),
                              output.data_ptr<scalar_t>(),
                              (size_t *) recv_count.data(),
                              &coll_attr));

                            //have to add wait here.
                            ret_req->wait();
                          });
                          return ret_req;
                        });
    }
    else {
      std::vector<at::Tensor> dummy_input{at::empty({0}, output_tensors[0].options())};
      return collective(dummy_input, output_tensors,
                        [&](at::Tensor& input,
                            at::Tensor& output,
                            ccl::communicator_t & comm,
                            ccl::stream_t& stream) {
                          std::vector<size_t> send_count(size_, 0);
                          std::vector<size_t> recv_count(size_, 0);
                          auto coll_attr = attr;
                          recv_count[opts.rootRank] = output.numel();
                          ccl::communicator::coll_request_t ret_req;

                          CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(input.scalar_type(), "scatter", [&] {
                            std::cout<< "scatter slave rank " << global_comm->rank() << " size " << global_comm->size() << std::endl;
                            CCL_CHECK(ret_req = comm->alltoallv(
                              input.data_ptr<scalar_t>(),
                              (size_t *) send_count.data(),
                              output.data_ptr<scalar_t>(),
                              (size_t *) recv_count.data(),
                              &coll_attr));
                          });
                          ret_req->wait();
                          return ret_req;
                        });
    }
  }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce_scatter(
        std::vector<at::Tensor>& /* unused */,
        std::vector<std::vector<at::Tensor>>& /* unused */,
        const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::send(
        std::vector<at::Tensor>& /* unused */,
        int dst_rank,
        int tag) {
  throw std::runtime_error("ProcessGroupCCL does not support send");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::recv(
        std::vector<at::Tensor>& /* unused */,
        int /* unused */,
        int /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support recv");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::recvAnysource(
        std::vector<at::Tensor>& /* unused */,
        int /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support recvAnysource");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather_base(
  at::Tensor& /* unused */,
  at::Tensor& /* unused */,
  const AllgatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support allgather_base");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce_coalesced(
  std::vector<at::Tensor>& /* unused */,
  const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support allreduce_coalesced");
}

} // namespace c10d
