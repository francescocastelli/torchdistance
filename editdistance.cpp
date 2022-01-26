#include "editdistance.h"

torch::Tensor editdistance_cpu(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    torch::Tensor& result);

torch::Tensor editdistance_cuda(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    torch::Tensor& result);

torch::Tensor editdistance(
    const torch::Tensor& src, 
    const torch::Tensor& trg)
{
    int64_t srcDims = src.ndimension();
    int64_t trgDims = trg.ndimension();

    TORCH_CHECK(srcDims == 2 || srcDims == 1, 
                "editdistance: Expect 1D or 2D Tensor, got: ",
                src.sizes());

    TORCH_CHECK(trgDims == 2 || trgDims == 1, 
                "editdistance: Expect 1D or 2D Tensor, got: ",
                trg.sizes());

    auto src_ = src;
    auto trg_ = trg; 
    if (srcDims == 1)
    {
        src_ = src_.reshape({1, src_.size(0)});
    }
    if (trgDims == 1)
    {
        trg_ = trg_.reshape({1, trg_.size(0)});
    }

    auto numBatch = src_.size(0);
    at::TensorOptions options(src_.device());
    options = options.dtype(at::ScalarType::Int);
    auto result = at::empty({numBatch, 1}, options);

    // dispatch
    if (src.device() == torch::kCUDA)
    {
	return editdistance_cuda(src_, trg_, result);
    }

    // by default dispatch on cpu
    return editdistance_cpu(src_, trg_, result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("editdistance", &editdistance, "editdistance forward");
}
