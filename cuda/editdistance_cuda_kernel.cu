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
    int64_t trgLen, 
    int64_t padToken) {
    
    const int batch = blockIdx.x;
    int cols = trgLen+1;

    auto src_ = src + batch * srcLen;
    auto trg_ = trg + batch * trgLen;
    auto result_ = result + batch;
    auto d = dMatrix + (batch * (trgLen+1) * 2);

    // handle padding
    for (int i=0; i < srcLen; i++)
    {
	    if (src_[i] == padToken)
	    {
		    srcLen = i;
		    break;
	    }
    }

    for (int i=0; i < trgLen; i++)
    {
	    if (trg_[i] == padToken)
	    {
		    trgLen = i;
		    break;
	    }
    }

    d[0] = 0;
    d[cols] = 1;
    for (int i = 0; i < trgLen + 1; i++) d[i] = i;
    for (int i = 1; i < srcLen + 1; i++) {
        for (int j = 1; j < trgLen + 1; j++) {
            d[(i&1)*cols + j] = std::min(std::min(d[((i-1)&1)*cols + j], d[(i&1)*cols + (j-1)]) + 1, 
			    		 d[((i-1)&1)*cols + (j-1)] + (src_[i-1] == trg_[j-1] ? 0 : 1));
        }
    }

    *result_ = d[(srcLen&1)*cols + trgLen];
}
}

torch::Tensor editdistance_cuda_kernel(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    torch::Tensor& result, 
    int64_t padToken) {

    const auto numBatch = src.size(0);
    const auto srcLen = src.size(1);
    const auto trgLen = trg.size(1);

    const int threads = 1;
    const dim3 blocks(numBatch);

    int* d;
    cudaMalloc(&d, numBatch * 2 * (trgLen+1) * sizeof(int));

    AT_DISPATCH_ALL_TYPES(
        src.scalar_type(),
        "editdistance_cuda",
        ([&] {
         distance_cuda_kernel<scalar_t><<<numBatch, threads>>>(
            src.data<scalar_t>(),
            trg.data<scalar_t>(),
            result.data<int>(),
	        d,
            srcLen, 
            trgLen, 
	        padToken);
        }));

    cudaFree(&d);
    return result;
}
