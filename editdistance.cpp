#include "editdistance.h"

torch::Tensor editdistance_cpu(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    torch::Tensor& result, 
    int64_t padToken);

torch::Tensor editdistance_cuda(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    torch::Tensor& result, 
    int64_t padToken);

torch::Tensor editdistance(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    int64_t padToken)
{
    int64_t srcDims = src.ndimension();
    int64_t trgDims = trg.ndimension();

    TORCH_CHECK(srcDims == 2 || srcDims == 1, 
                "editdistance: Expect 1D or 2D Tensor, got: ",
                src.sizes());

    TORCH_CHECK(trgDims == 2 || trgDims == 1, 
                "editdistance: Expect 1D or 2D Tensor, got: ",
                trg.sizes());

    TORCH_CHECK(srcDims == trgDims, 
                "editdistance: Expect src and trg to have the same number of dimensions");

    auto src_ = src;
    auto trg_ = trg; 
    if (srcDims == 1)
    {
        src_ = src_.reshape({1, src_.size(0)});
        trg_ = trg_.reshape({1, trg_.size(0)});
    }

    auto numBatch = src_.size(0);
    TORCH_CHECK(src_.size(0) == trg_.size(0), 
	        "editdistance: expected src and trg to have same batch size");

    at::TensorOptions options(src_.device());
    options = options.dtype(at::ScalarType::Int);
    auto result = at::empty({numBatch, 1}, options);


    // dispatch
    if (src_.device().is_cuda())
    {
	return editdistance_cuda(src_, trg_, result, padToken);
    }

    // by default dispatch on cpu
    return editdistance_cpu(src_, trg_, result, padToken);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("editdistance", &editdistance, "editdistance forward");
}
