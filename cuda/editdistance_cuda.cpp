#include <torch/extension.h> 

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor
editdistance_cuda_kernel(const torch::Tensor& src, 
			 const torch::Tensor& trg, 
			 torch::Tensor& result, 
			 int64_t padToken);

torch::Tensor editdistance_cuda(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    int64_t padToken) {

    CHECK_INPUT(src);
    CHECK_INPUT(trg);

    auto numBatch = src.size(0);
    at::TensorOptions options(src.device());
    options = options.dtype(at::ScalarType::Int);
    auto result = at::empty({numBatch, 1}, options);

    return editdistance_cuda_kernel(src, trg, result, padToken);
}

TORCH_LIBRARY_IMPL(editdistance, CUDA, m) {
  m.impl("editdistance", editdistance_cuda);
}
