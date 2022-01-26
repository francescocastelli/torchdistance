#include <torch/extension.h> 
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__global__ void distance_cuda_kernel(
    scalar_t* const __restrict__ src, 
    scalar_t* const __restrict__ trg, 
    int* __restrict__ result,
    int* dMatrix,
    int64_t srcLen,
    int64_t trgLen) {
    
    const int batch = blockIdx.x;
    auto src_ = src + batch * srcLen;
    auto trg_ = trg + batch * trgLen;
    auto result_ = result + batch;
    auto d = dMatrix + batch;
    
    int cols = (trgLen+1);
    //std::vector<std::vector<int32_t>> d(2, std::vector<int32_t>(trgLen+1));

    d[0] = 0;
    d[cols] = 1;
    for (int i = 0; i < trgLen + 1; i++) d[i*cols] = i;
    for (int i = 1; i < srcLen + 1; i++) {
        for (int j = 1; j < trgLen + 1; j++) {
            d[(i&1)*cols + j] = std::min(std::min(d[((i-1)&1)*cols + j], d[(i&1)*cols + (j-1)]) + 1, d[((i-1)&1)*cols + (j-1)] + (src_[i-1] == trg_[j-1] ? 0 : 1));
        }
    }

    *result_ = d[(srcLen&1)*cols + trgLen];
}
}

torch::Tensor editdistance_cuda_kernel(
    const torch::Tensor& src, 
    const torch::Tensor& trg) {

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
    auto srcLen = src_.size(1);
    auto trgLen = trg_.size(1);
    
    at::TensorOptions options(src_.device());
    options = options.dtype(at::ScalarType::Int);

    auto result = at::empty({numBatch, 1}, options);

    const int threads = 1;
    const dim3 blocks(numBatch);

    int* d;
    cudaMalloc(&d, numBatch * 2 * (trgLen+1) * sizeof(int));

    AT_DISPATCH_ALL_TYPES(
        src_.scalar_type(),
        "editdistance_cuda",
        ([&] {
         distance_cuda_kernel<scalar_t><<<blocks, threads>>>(
            src_.data<scalar_t>(),
            trg_.data<scalar_t>(),
            result.data<int>(),
	    d,
            srcLen, 
            trgLen
          );
        }));

    cudaFree(&d);
    return result;
}