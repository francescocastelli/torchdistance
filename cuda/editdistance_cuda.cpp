#include <torch/extension.h> 

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor
editdistance_cuda_kernel(const torch::Tensor& src, const torch::Tensor& trg);

torch::Tensor editdistance_cuda(
    const torch::Tensor& src, 
    const torch::Tensor& trg) {

    CHECK_INPUT(src);
    CHECK_INPUT(trg);
    return editdistance_cuda_kernel(src, trg);
}