//
// Created by johnlu on 2020/9/23.
//
#include <ATen/record_function.h>
#include <ProcessGroupCCL.hpp>

namespace c10d
{

namespace {

enum class
SparseResultMode : std::uint8_t
{
  DIRECT,
  OOP,
  COPY
};

static ccl::sparse_coalesce_mode sparseCoalesceMode;
static SparseResultMode sparseResultMode;

// Type mapping
#if CCL_MAJOR_VERSION == 0 && CCL_MINOR_VERSION < 6
std::map<at::ScalarType, ccl::data_type> cclDatatypes =
{
    {at::kByte, ccl::data_type::dt_char},
    {at::kChar, ccl::data_type::dt_char},
    {at::kDouble, ccl::data_type::dt_double},
    {at::kBFloat16, ccl::data_type::dt_bfp16},
    {at::kFloat, ccl::data_type::dt_float},
    {at::kInt, ccl::data_type::dt_int},
    {at::kLong, ccl::data_type::dt_int64}
};
#else
std::map<at::ScalarType, ccl::datatype> cclDatatypes =
  {
    {at::kByte, ccl::datatype::uint8},
    {at::kChar, ccl::datatype::uint8},
    {at::kDouble, ccl::datatype::float64},
    {at::kBFloat16, ccl::datatype::bfloat16},
    {at::kFloat, ccl::datatype::float32},
//    {at::kInt, ccl::datatype::int64},
    {at::kLong, ccl::datatype::int64}
  };

std::map<ccl_datatype_t, at::ScalarType> ptDatatypes =
  {
    {ccl_dtype_char, at::kByte},
    {ccl_dtype_int, at::kInt},
    {ccl_dtype_bfp16, at::kBFloat16},
    {ccl_dtype_float, at::kFloat},
    {ccl_dtype_double, at::kDouble},
    {ccl_dtype_int64, at::kLong}
  };
#endif

// Op mapping
std::map<ReduceOp, ccl::reduction> cclOps =
  {
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
  };

std::ostream& operator << (std::ostream& os, const SparseResultMode& mode)
{
  os << static_cast<std::underlying_type<SparseResultMode>::type>(mode);
  return os;
}

// Checking the input tensor's validity
void checkSingleTensorHelper(const at::Tensor& tensor)
{
  TORCH_CHECK(tensor.is_sparse() || tensor.is_contiguous(), "input dense tensor has to be contiguous");
  TORCH_CHECK(!tensor.is_cuda(), "CUDA tensor detected and CCL doesn't support CUDA buffers");
  TORCH_CHECK(tensor.numel() >= 0, "input tensor numel should be non-negative");
}

void checkSingleTensor(const std::vector<at::Tensor>& tensors)
{
  TORCH_CHECK(tensors.size() == 1,
              "CCL process group does not support tensors count " + std::to_string(tensors.size()));

  checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(const at::Tensor& tensor,
                          const std::vector<at::Tensor>& tensors) __attribute__((unused));

void checkSameSizeAndType(const at::Tensor& tensor,
                          const std::vector<at::Tensor>& tensors)
{
  for (size_t i = 0; i < tensors.size(); ++i)
  {
    TORCH_CHECK((tensors[i].numel() == tensor.numel()) &&
                (tensors[i].scalar_type() == tensor.scalar_type()),
                "tensors are not equal in size or data type");

    checkSingleTensorHelper(tensors[i]);
  }
}

void checkSameType(const at::Tensor& tensor,
                   const std::vector<at::Tensor>& tensors)
{
  for (size_t i = 0; i < tensors.size(); ++i)
  {
    TORCH_CHECK(tensors[i].scalar_type() == tensor.scalar_type(),
                "tensors are not equal in data type");

    checkSingleTensorHelper(tensors[i]);
  }
}

void checkSplitSizes(
  const std::vector<int64_t>& split_sizes,
  const at::Tensor& tensor,
  int groupSize)
{
  if (split_sizes.size() == 0)
  {
    TORCH_CHECK(tensor.size(0) % groupSize == 0,
                "tensor's dim 0 does not divide equally across group size");
  }
  else
  {
    TORCH_CHECK(split_sizes.size() == (size_t)groupSize,
                "number of tensor splits not equal to group size");

    int sum = std::accumulate(split_sizes.begin(), split_sizes.end(), 0);

    TORCH_CHECK(sum == tensor.size(0),
                "split sizes doesn't match total dim 0 size");
  }
}

typedef struct
{
  bool isFlat;
  int64_t size;
  at::Tensor firstTensor;
} FlatCheckResult;

FlatCheckResult computeLengthsAndCheckFlat(
  const std::vector<at::Tensor>& tensors,
  std::vector<size_t>& lengths)
{
  int64_t groupSize = lengths.size();
  auto firstTensor = tensors[0];
  int64_t offset = 0;
  auto firstLength = firstTensor.numel();
  auto storage = firstTensor.storage();
  auto firstStorageOffset = firstTensor.storage_offset();
  bool isFlat = true;

  for (int i = 0; i < groupSize; i++)
  {
    auto& curTensor = tensors[i];
    int64_t length = curTensor.numel();

    if (firstLength == 0 && length != 0)
    {
      firstLength = length;
      firstTensor = curTensor;
      storage = curTensor.storage();
      firstStorageOffset = curTensor.storage_offset();
    }

    lengths[i] = length;

    if (isFlat && length != 0 &&
        (!storage.is_alias_of(curTensor.storage()) ||
         curTensor.storage_offset() != firstStorageOffset + offset))
      isFlat = false;

    offset += length;
  }

  return FlatCheckResult{isFlat, offset, firstTensor};
}

bool computeLengthsAndCheckAndGetFlat(
  const std::vector<at::Tensor>& tensors,
  std::vector<size_t>& lengths,
  at::Tensor& flatTensor,
  int64_t& flatLength)
{
  auto flatRes = computeLengthsAndCheckFlat(tensors, lengths);

  flatLength = flatRes.size;

  if (flatRes.isFlat)
  {
    flatTensor = flatRes.firstTensor;
  }
  else
  {
    flatTensor = at::empty({flatRes.size}, flatRes.firstTensor.options());
  }

  return flatRes.isFlat;
}

} //namespace anonymous

class VanillaCPU final: public CCLStubs {

public:
  VanillaCPU() {}

  bool enabled() override {
    return true;
  }

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce(std::vector<at::Tensor>& tensors,
                                                           const AllreduceOptions& opts,
                                                           ccl::communicator& comm) override;


  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce(std::vector<at::Tensor>& tensors,
                                                        const ReduceOptions& opts,
                                                        ccl::communicator& comm) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast(std::vector<at::Tensor>& tensors,
                                                           const BroadcastOptions& opts,
                                                           ccl::communicator& comm) override;

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                           std::vector<at::Tensor>& inputTensors,
                                                           const AllgatherOptions& opts,
                                                           ccl::communicator& comm) override;

  ~VanillaCPU() {}

private:

};

struct RegisterCPUPMethods {
  RegisterCPUPMethods() {
    static VanillaCPU methods;

    sparseCoalesceMode = ccl::sparse_coalesce_mode::regular;
    const char* sparseCoalesceModeEnv = getenv("CCL_SPARSE_COALESCE_MODE");
    if (sparseCoalesceModeEnv)
    {
      sparseCoalesceMode = ccl::sparse_coalesce_mode(atoi(sparseCoalesceModeEnv));
    }

    sparseResultMode = SparseResultMode::DIRECT;
    const char* sparseResultModeEnv = getenv("CCL_SPARSE_RESULT_MODE");
    if (sparseResultModeEnv)
    {
      sparseResultMode = (SparseResultMode)atoi(sparseResultModeEnv);
    }

    printf("sparse options: coalesce mode %d, result mode %d\n",
           sparseCoalesceMode, (int)sparseResultMode);

    registerCPUMethods(&methods);
  }
};

void sparseAllreduceCompletionFn(
  const void* indBuf, size_t indCount, ccl::datatype indDatatype,
  const void* valBuf, size_t valCount, ccl::datatype valDatatype,
  const void* fnCtx)
{
  TORCH_CHECK(fnCtx, "null fn ctx");

  /*printf("sparseAllreduceCompletionFn: "
         "indices buf %p, count %zu, dt %d, "
         "values buf %p, count %zu, dt %d, "
         "fn_ctx %p\n",
         indBuf, indCount, indDatatype,
         valBuf, valCount, valDatatype,
         fnCtx); fflush(stdout);*/

  ProcessGroupCCL::AsyncWorkCCL* work_cts = (ProcessGroupCCL::AsyncWorkCCL*)fnCtx;

  std::vector<at::Tensor>& inputTensors = work_cts->getInputTensors();
  std::vector<at::Tensor>& outputTensors = work_cts->getOutputTensors();

  TORCH_CHECK(inputTensors.size() == 1, "unexpected inputTensors size");
  TORCH_CHECK(outputTensors.size() == 0, "unexpected outputTensors size");

  outputTensors.reserve(inputTensors.size());

  at::Tensor& inputTensor = inputTensors[0];

  TORCH_CHECK(inputTensor.sparse_dim() == 1, "unexpected sparse_dim");

  const auto valueShape = inputTensor.sizes().slice(inputTensor.sparse_dim());
  auto resultValueShape = std::vector<int64_t>({(int64_t)indCount});
  std::copy(valueShape.begin(), valueShape.end(), std::back_inserter(resultValueShape));

  auto rawIndices = at::from_blob((void*)indBuf,
                                  {1, (long int)indCount},
                                  ptDatatypes.at(static_cast<int>(indDatatype)));

  auto rawValues = at::from_blob((void*)valBuf,
                                 resultValueShape,
                                 ptDatatypes.at(static_cast<int>(valDatatype)));

  auto indices = at::empty({1, (long int)indCount}, inputTensor._indices().options());

  auto values = at::empty(resultValueShape,
                          inputTensor._values().options());

  indices.copy_(rawIndices);
  values.copy_(rawValues);

  /*int64_t* indPtr = indices.data_ptr<int64_t>();
  for (size_t idx = 0; idx < indCount; idx++)
  {
      printf("indices[%zu] = %ld\n", idx, indPtr[idx]);
  }

  float* valPtr = values.data_ptr<float>();
  for (size_t idx = 0; idx < valCount; idx++)
  {
      printf("values[%zu] = %f\n", idx, valPtr[idx]);
  }*/

  auto resultTensor =
    at::_sparse_coo_tensor_unsafe(indices,
                                  values,
                                  inputTensor.sizes(),
                                  inputTensor.options());

  if (sparseCoalesceMode != ccl::sparse_coalesce_mode::disable)
    resultTensor._coalesced_(true);

  if (sparseResultMode == SparseResultMode::COPY)
  {
    /* propagate result using 2 ways - inputTensors and outputTensors */
    for (size_t i = 0; i < inputTensors.size(); i++)
    {
      inputTensors[i].copy_(resultTensor);
      if (resultTensor.is_sparse())
      {
        outputTensors.push_back(resultTensor.clone());
      }
      else
      {
        outputTensors.push_back(resultTensor.clone(at::MemoryFormat::Contiguous));
      }
    }
  }
  else if (sparseResultMode == SparseResultMode::OOP)
  {
    /* propagate result using 1 way - outputTensors */
    TORCH_CHECK(resultTensor.layout() == c10::kSparse, "unexpected tensor layout");
    outputTensors.push_back(resultTensor);
  }
  else
  {
    TORCH_CHECK(0, "unexpected sparseResultMode ", sparseResultMode);
  }

  return ;
}

void sparseAllreduceAllocFn(
  size_t indCount, ccl::datatype indDatatype,
  size_t valCount, ccl::datatype valDatatype,
  const void* fnCtx, void** outIndBuf, void** outValBuf)
{
  TORCH_CHECK(fnCtx, "fnCtx");
  TORCH_CHECK(outIndBuf, "outIndBuf");
  TORCH_CHECK(outValBuf, "outValBuf");

  TORCH_CHECK(sparseResultMode == SparseResultMode::DIRECT,
              "unexpected sparseResultMode ", sparseResultMode);

  ProcessGroupCCL::AsyncWorkCCL* work_cts = (ProcessGroupCCL::AsyncWorkCCL*)fnCtx;

  std::vector<at::Tensor>& inputTensors = work_cts->getInputTensors();
  std::vector<at::Tensor>& outputTensors = work_cts->getOutputTensors();

  TORCH_CHECK(inputTensors.size() == 1, "unexpected inputTensors size");
  TORCH_CHECK(outputTensors.size() == 0, "unexpected outputTensors size");

  outputTensors.reserve(inputTensors.size());

  at::Tensor& inputTensor = inputTensors[0];

  TORCH_CHECK(inputTensor.sparse_dim() == 1, "unexpected sparse_dim");

  const auto valueShape = inputTensor.sizes().slice(inputTensor.sparse_dim());
  auto resultValueShape = std::vector<int64_t>({(int64_t)indCount});
  std::copy(valueShape.begin(), valueShape.end(), std::back_inserter(resultValueShape));

  auto indices = at::empty({1, (long int)indCount}, inputTensor._indices().options());
  auto values = at::empty(resultValueShape,
                          inputTensor._values().options());

  auto resultTensor =
    at::_sparse_coo_tensor_unsafe(indices,
                                  values,
                                  inputTensor.sizes(),
                                  inputTensor.options());

  if (sparseCoalesceMode != ccl::sparse_coalesce_mode::disable)
    resultTensor._coalesced_(true);

  /* propagate result using 1 way - outputTensors */
  TORCH_CHECK(resultTensor.layout() == c10::kSparse, "unexpected tensor layout");
  outputTensors.push_back(resultTensor);

  *outIndBuf = resultTensor._indices().data_ptr();
  *outValBuf = resultTensor._values().data_ptr();

  TORCH_CHECK(*outIndBuf, "result outIndBuf");
  TORCH_CHECK(*outValBuf, "result outValBuf");

  return ;
}

template <class RunF>
class CPUWorkCCL : public ProcessGroupCCL::AsyncWorkCCL {
public:
  CPUWorkCCL(const std::vector<at::Tensor>& inputs, const RunF f) : AsyncWorkCCL(inputs), f(f) {}

  void run() override {
    f(this);
  };

private:
  RunF f;
};

template <class RunF>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> make_work_ccl(const std::vector<at::Tensor>& inputs, RunF f) {
  return std::make_shared<CPUWorkCCL<RunF>>(inputs, f);
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::allreduce(std::vector<at::Tensor>& tensors,
                                                                     const AllreduceOptions& opts,
                                                                     ccl::communicator& comm) {
  const auto& layout = tensors[0].layout();
  std::string funcName = (layout == c10::kStrided) ? "pg::allreduce" : "pg::sparse_allreduce";
  RECORD_FUNCTION(funcName, std::vector<c10::IValue>({tensors[0]}));

  checkSingleTensor(tensors);

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  if (layout == c10::kStrided) {
    work = make_work_ccl(tensors,
      [&](ProcessGroupCCL::AsyncWorkCCL* work){
        const auto& layout = tensors[0].layout();
        CCL_CHECK(work->req = comm.allreduce(tensors[0].data_ptr(),
                                   tensors[0].data_ptr(),
                                   (size_t)tensors[0].numel(),
                                   cclDatatypes.at(tensors[0].scalar_type()),
                                   cclOps.at(opts.reduceOp)));
      });

  } else if (layout == c10::kSparse) {
    work = make_work_ccl(tensors,
      [&](ProcessGroupCCL::AsyncWorkCCL* work){
        at::Tensor input = tensors[0];
        TORCH_CHECK(input.sparse_dim() == 1, "allreduce: only single sparse_dim is supported");

        auto indices = input._indices();
        auto values = input._values();

        auto &env = ccl::environment::instance();
        auto attr = env.create_operation_attr<ccl::sparse_allreduce_attr>();

        if (sparseResultMode == SparseResultMode::DIRECT)
          attr.set<ccl::sparse_allreduce_attr_id::alloc_fn>(static_cast<ccl::sparse_allreduce_alloc_fn>(sparseAllreduceAllocFn));
        else
          attr.set<ccl::sparse_allreduce_attr_id::completion_fn>(static_cast<ccl::sparse_allreduce_completion_fn>(sparseAllreduceCompletionFn));
        attr.set<ccl::sparse_allreduce_attr_id::fn_ctx>(static_cast<const void*>(work));
        attr.set<ccl::sparse_allreduce_attr_id::coalesce_mode>(sparseCoalesceMode);

        CCL_CHECK(work->req = comm.sparse_allreduce(indices.data_ptr(),
                                              (size_t)indices.numel(),
                                              values.data_ptr(),
                                              (size_t)values.numel(),
                                              nullptr, 0, nullptr, 0,
                                              cclDatatypes.at(indices.scalar_type()),
                                              cclDatatypes.at(values.scalar_type()),
                                              cclOps.at(opts.reduceOp),
                                              attr));
      });
  }

  return work;
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::reduce(std::vector<at::Tensor>& tensors,
                                                                  const ReduceOptions& opts,
                                                                  ccl::communicator& comm) {
  RECORD_FUNCTION("pg::reduce", std::vector<c10::IValue>({tensors[0]}));

  checkSingleTensor(tensors);

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = make_work_ccl(tensors,
                       [&](ProcessGroupCCL::AsyncWorkCCL* work) {
                          CCL_CHECK(work->req = comm.reduce(tensors[0].data_ptr(),
                                                      tensors[0].data_ptr(),
                                                      (size_t)tensors[0].numel(),
                                                      cclDatatypes.at(tensors[0].scalar_type()),
                                                      cclOps.at(opts.reduceOp),
                                                      (size_t)opts.rootRank));
                       });

  return work;
}

std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::broadcast(std::vector<at::Tensor>& tensors,
                                                                     const BroadcastOptions &opts,
                                                                     ccl::communicator& comm) {
  RECORD_FUNCTION("pg::bcast", std::vector<c10::IValue>({tensors[0]}));

  checkSingleTensor(tensors);

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = make_work_ccl(tensors,
                       [&](ProcessGroupCCL::AsyncWorkCCL* work) {
                          CCL_CHECK(work->req = comm.broadcast(tensors[0].data_ptr(),
                                                         (size_t)tensors[0].numel(),
                                                         cclDatatypes.at(tensors[0].scalar_type()),
                                                         (size_t)opts.rootRank));
                       });

  return work;
}


std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                     std::vector<at::Tensor>& inputTensors,
                                                                     const AllgatherOptions& opts,
                                                                     ccl::communicator& comm) {
  RECORD_FUNCTION("pg::allgather", std::vector<c10::IValue>({inputTensors[0]}));

  checkSingleTensor(inputTensors);

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = make_work_ccl(inputTensors,
                       [&](ProcessGroupCCL::AsyncWorkCCL* work) {
                          work->debugName = std::string("allgather::sz:") + std::to_string(inputTensors[0].numel());
                          RECORD_FUNCTION("pg::allgather", std::vector<c10::IValue>({inputTensors[0]}));

                          checkSingleTensor(inputTensors);

                          TORCH_CHECK(outputTensors.size() == 1,
                                      "allgather: multi-GPU collective is not supported");

                          TORCH_CHECK(comm.size() == outputTensors[0].size(),
                                      "allgather: number of output tensors should equal to the world size");

                          checkSameType(inputTensors[0], outputTensors[0]);

                          void* recvBuf = nullptr;
                          std::vector<void*> recvBufs;
                          std::vector<size_t> recvCounts(comm.size(), 0);

                          auto flatRes = computeLengthsAndCheckFlat(outputTensors[0], recvCounts);

                          TORCH_CHECK((size_t)inputTensors[0].numel() == recvCounts[comm.rank()],
                                      "allgather: send and recv count doesn't match");

                          if (flatRes.isFlat)
                          {
                            recvBuf = flatRes.firstTensor.data_ptr();
                          }
                          else
                          {
                            std::transform(outputTensors[0].begin(), outputTensors[0].end(),
                                           std::back_inserter(recvBufs),
                                           [](const at::Tensor& t) { return t.data_ptr(); } );
                            recvBuf = recvBufs.data();
                          }

                          {
                            auto &env = ccl::environment::instance();
                            auto attr = env.create_operation_attr<ccl::allgatherv_attr>();
                            attr.set<ccl::allgatherv_attr_id::vector_buf>(flatRes.isFlat ? 0 : 1);
                            CCL_CHECK(work->req = comm.allgatherv(inputTensors[0].data_ptr(),
                                                            (size_t)inputTensors[0].numel(),
                                                            recvBuf,
                                                            recvCounts,
                                                            cclDatatypes.at(inputTensors[0].scalar_type()),
                                                            attr));
                          }

                          std::vector<at::Tensor> agTensors;

                          if (flatRes.isFlat)
                            agTensors.emplace_back(flatRes.firstTensor);
                          else
                            agTensors.assign(outputTensors[0].begin(), outputTensors[0].end());
                          agTensors.emplace_back(inputTensors[0]);
                       });

  return work;
}



static RegisterCPUPMethods cpu_register;

} // namespace c10d