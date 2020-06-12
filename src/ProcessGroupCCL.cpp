#include "ProcessGroupCCL.hpp"

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

namespace c10d {
namespace {
// Op mapping
std::map<ReduceOp, ccl::reduction> cclOp = {
        {ReduceOp::MIN, ccl::reduction::min},
        {ReduceOp::MAX, ccl::reduction::max},
        {ReduceOp::SUM, ccl::reduction::sum},
        {ReduceOp::PRODUCT, ccl::reduction::prod},
};

// Type mapping
std::map<at::ScalarType, ccl::datatype> cclDatatype = {
        {at::kByte, ccl::datatype::dt_char},
        {at::kChar, ccl::datatype::dt_char},
        {at::kDouble, ccl::datatype::dt_double},
        {at::kFloat, ccl::datatype::dt_float},
        {at::kInt, ccl::datatype::dt_int},
        {at::kLong, ccl::datatype::dt_int64}
};


template <typename buffer_data_type, int dims = 1>
cl::sycl::buffer<buffer_data_type, dims> make_buffer(void* virtual_ptr) {
  static_assert(dims == 1, "buffer dims is not 1");

  //reinterpret the buffer to the required type.
  auto buf = at::dpcpp::dpcppGetBufferMap().template get_buffer<uint8_t>(virtual_ptr);
  auto range = cl::sycl::range<dims>(buf.get_size()/sizeof(buffer_data_type));
  return buf.template reinterpret<buffer_data_type, dims>(range);
}

// Checking the input tensor's validity
void checkSingleTensorHelper(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    throw std::runtime_error("input tensor has to be dense");
  }
  if (tensor.is_cuda()) {
    throw std::runtime_error(
            "CUDA tensor detected and CCL doesn't support CUDA buffers");
  }
  if (tensor.numel() < 0) {
    throw std::runtime_error("input tensor numel should be non-negative");
  }
}

void checkRank(int rank, int size) {
  if (rank < 0 || rank >= size) {
    throw std::runtime_error("unexpected rank");
  }
}

void checkSingleTensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error(
            "CCL process group does not support multi-GPU collectives");
  }
  checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(
        const at::Tensor& tensor,
        const std::vector<at::Tensor>& tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if ((tensors[i].numel() != tensor.numel()) ||
        (tensors[i].scalar_type() != tensor.scalar_type())) {
      throw std::runtime_error("Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensors[i]);
  }
}

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

template <typename Fn, typename PreProcess, typename PostProcess>
std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::collective(
        std::vector<at::Tensor>& inputs,
        std::vector<at::Tensor>& outputs,
        Fn fn,
        PreProcess pre,
        PostProcess post) {
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
    auto req = fn(inputs[i], outputs[i], comms_collector[i], cclStream);
    work->requests_[i] = std::move(req);
  }

  post(gpu_streams[key]);

  return work;
}


template <typename Fn>
std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::collective(
        std::vector<at::Tensor>& inputs,
        std::vector<at::Tensor>& outputs,
        Fn fn) {
  return collective(
          inputs,
          outputs,
          fn,
          [](std::vector<ccl::stream_t>&) {},
          [](std::vector<ccl::stream_t>&) {});
}

ProcessGroupCCL::WorkCCL::~WorkCCL() {
  for(auto& request_ : requests_) {
    if (request_.get()) {
      std::cerr
              << "Attempted destruction of WorkCCL before work has completed, "
              << "terminating the program." << std::endl;
      std::terminate();
    }
  }
}

bool ProcessGroupCCL::WorkCCL::isCompleted() {
  for(auto& request_ : requests_) {
    if (!request_.get()) {
      return true;
    }
  }

  bool flag = false;

  std::unique_lock<std::mutex> globalLock(pg_global_mutex);

  for(auto& request_ : requests_) {
    bool _flag;
    CCL_CHECK( _flag = request_->test());

    if (_flag) {
      request_.reset();
      flag = _flag;
    }
  }

  return flag;
}

bool ProcessGroupCCL::WorkCCL::isSuccess() const {
  for(auto& request_ : requests_) {
    if (request_.get()) {
      throw std::runtime_error(
              "Invalid call to WorkCCL::isSuccess before work has completed");
    }
  }
  return true;
}

bool ProcessGroupCCL::WorkCCL::wait() {
  std::cout<< "work wait  " << requests_.size() << std::endl;
  for(auto& request_ : requests_) {
    std::cout<< "johnlu " << request_.get() << std::endl;
    if (!request_.get()) {
      return false;
    }
  }

  std::cout<< "johnlu22 " << requests_.size()  << std::endl;
  std::unique_lock<std::mutex> lock(mutex_);
  for(auto& request_ : requests_) {
    CCL_CHECK(request_->wait());
    request_.reset();
  }

  return true;
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
  std::cout<< "create occl pg rank " << global_comm->rank() << " size " << global_comm->size() << std::endl;
}

ProcessGroupCCL::~ProcessGroupCCL() {}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::broadcast(
        std::vector<at::Tensor>& tensors,
        const BroadcastOptions& opts) {
  auto dev_type = check_tensors_properties(tensors);

  const auto devices = getDeviceList(tensors);
  const auto key = getKeyFromDevices(devices);
  std::cout << "johnlu device name:" << key << std::endl;

  auto coll_attr = attr;
  const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
  if (dev_type == c10::DeviceType::DPCPP) {
    return collective(tensors, tensors,
                      [&](at::Tensor input,
                          at::Tensor output,
                          ccl::communicator_t & gpu_comm,
                          ccl::stream_t& stream) {
                        auto count = input.numel();
                        ccl::communicator::coll_request_t ret_req;

                        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "allreduce_sycl", [&] {
                          auto input_buf = make_buffer<scalar_t>(input.data_ptr());
                          ret_req = gpu_comm->bcast(
                            input_buf,
                            (size_t)count,
                            root,
                            &coll_attr,
                            stream);
                        });

                        std::cout<< "rank " << this->rank_ << " ret_req " << ret_req.get() << std::endl;
                        return ret_req;
                      });
  }
  else if (dev_type == c10::DeviceType::CPU) {
    checkSingleTensor(tensors);

#ifdef USE_CACHE
    coll_attr.match_id = tensorName.c_str();
#endif
    ccl::communicator::coll_request_t ret_req;
    std::shared_ptr<ccl::request> req;

    AT_DISPATCH_FLOATING_TYPES(tensors[0].scalar_type(), "allreduce", [&] {
      CCL_CHECK(ret_req = global_comm->bcast(static_cast<scalar_t*>(tensors[0].data_ptr()),
                                                 (size_t)tensors[0].numel(),
                                                 root,
                                                 &coll_attr));
    });

    return std::make_shared<ProcessGroupCCL::WorkCCL>(ret_req);
  }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce(
        std::vector<at::Tensor>& tensors,
        const AllreduceOptions& opts) {
  auto dev_type = check_tensors_properties(tensors);

  const auto devices = getDeviceList(tensors);
  const auto key = getKeyFromDevices(devices);
  std::cout << "johnlu device name:" << key << std::endl;

  auto coll_attr = attr;
  if (dev_type == c10::DeviceType::DPCPP) {
    return collective(tensors, tensors,
                      [&](at::Tensor input,
                          at::Tensor output,
                          ccl::communicator_t & gpu_comm,
                          ccl::stream_t& stream) {
                        auto count = input.numel();
                        auto reduce_op = cclOp.at(opts.reduceOp);
                        ccl::communicator::coll_request_t ret_req;

                        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "allreduce_sycl", [&] {
                          auto input_buf = make_buffer<scalar_t>(input.data_ptr());
                          auto output_buf = make_buffer<scalar_t>(output.data_ptr());

                          ret_req = gpu_comm->allreduce(
                            input_buf,
                            output_buf,
                            (size_t)count,
                            reduce_op,
                            &coll_attr,
                            stream);
                        });

                        std::cout<< "rank " << this->rank_ << " ret_req " << ret_req.get() << std::endl;
                        return ret_req;
                      });
  }
  else if (dev_type == c10::DeviceType::CPU) {
    checkSingleTensor(tensors);

#ifdef USE_CACHE
    coll_attr.match_id = tensorName.c_str();
#endif

    ccl::communicator::coll_request_t ret_req;
    std::shared_ptr<ccl::request> req;
    auto reduce_op = cclOp.at(opts.reduceOp);

    AT_DISPATCH_FLOATING_TYPES(tensors[0].scalar_type(), "allreduce", [&] {
      CCL_CHECK(ret_req = global_comm->allreduce(static_cast<scalar_t*>(tensors[0].data_ptr()),
        static_cast<scalar_t*>(tensors[0].data_ptr()),
        (size_t)tensors[0].numel(),
        reduce_op,
        &coll_attr));
    });

    return std::make_shared<ProcessGroupCCL::WorkCCL>(ret_req);
  }
}

//std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce_coalesced(
//        std::vector<at::Tensor>& tensors,
//        const AllreduceCoalescedOptions& opts) {
//  throw std::runtime_error(
//          "allreduce_coalesced is currently not supported with CCL");
//}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce(
        std::vector<at::Tensor>& tensors,
        const ReduceOptions& opts) {
  throw std::runtime_error("ProcessGroupCCL does not support reduce");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllgatherOptions& opts) {
  throw std::runtime_error("ProcessGroupCCL does not support allgather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::barrier(
        const BarrierOptions& opts) {

#ifdef CCL_IN
  std::vector<at::Device> devices;
  if (used_gpu_device_idxs.empty()) {
    // This means there is not yet a NCCL collective being called
    // Here we have to use the best guesses and will use a single GPU to call
    // allreduce to achieve barrier.
    // In case the multiple processes fall into the same node, we use rank to
    // ensure that each process is on a different GPU
//    auto numGPUs = at::cuda::getNumGPUs();
//    int16_t deviceIdx = static_cast<int16_t>(rank_ % numGPUs);
//    devices.push_back(at::Device(at::DeviceType::CUDA, deviceIdx));
  } else {
    for (auto usedDeviceIdx : used_gpu_device_idxs) {
      devices.push_back(at::Device(at::DeviceType::SYCL, usedDeviceIdx));
    }
  }

  std::unique_lock<std::mutex> globalLock(pg_global_mutex);
  CCL_CHECK(global_comm->barrier());
  //TODO: the barrier must be async returned.
  return std::make_shared<ProcessGroupCCL::WorkCCL>(devices);
#else
  throw std::runtime_error("ProcessGroupCCL does not support barrier");
#endif
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::gather(
        std::vector<std::vector<at::Tensor>>& /* unused */,
        std::vector<at::Tensor>& /* unused */,
        const GatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support gather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::scatter(
        std::vector<at::Tensor>& /* unused */,
        std::vector<std::vector<at::Tensor>>& /* unused */,
        const ScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce_scatter(
        std::vector<at::Tensor>& /* unused */,
        std::vector<std::vector<at::Tensor>>& /* unused */,
        const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::send(
        std::vector<at::Tensor>& /* unused */,
        int /* unused */,
        int /* unused */) {
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
  at::Tensor& outputBuffer,
  at::Tensor& inputBuffer,
  const AllgatherOptions& opts) {
  throw std::runtime_error("ProcessGroupCCL does not support allgather_base");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce_coalesced(
  std::vector<at::Tensor>& tensors,
  const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error("ProcessGroupCCL does not support allreduce_coalesced");
}

} // namespace c10d
