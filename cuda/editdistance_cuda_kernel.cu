#include <torch/extension.h> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAFunctions.h>

#ifdef DEBUG
#include "utils.cuh"
#include <iostream>
#endif

namespace {

// TODO: this can probabliy be parallelized using a gpu
template <typename scalar_t>
__device__ int64_t handlePadLen(scalar_t* str, int64_t strLen, int64_t padToken) {
    for (int i=0; i < strLen; i++)
	    if (str[i] == padToken) return i;

    return strLen;
}

template <typename scalar_t>
__global__ void distance_cuda_kernel(
    scalar_t* const __restrict__ src, 
    scalar_t* const __restrict__ trg, 
    int* __restrict__ result,
    int64_t srcLen,
    int64_t trgLen, 
    int64_t padToken) {
    
    const int batch = blockIdx.x;

    auto srcBatch = src + batch * srcLen;
    auto trgBatch = trg + batch * trgLen;
    auto result_ = result + batch;

    // handle padding
    srcLen = handlePadLen(srcBatch, srcLen, padToken);
    trgLen = handlePadLen(trgBatch, trgLen, padToken);

    // base case
    if (srcLen == 0) { *result = trgLen; return; }
    if (trgLen == 0) { *result = srcLen; return; }

    auto src_ = srcBatch, trg_ = trgBatch;
    auto srcLen_ = srcLen, trgLen_ = trgLen;
    if (trgLen < srcLen) src_ = trgBatch, trg_ = srcBatch, srcLen_ = trgLen, trgLen_ = srcLen;

    int cols = trgLen_+1;
    // TODO: cudaMalloc is probably better, but first we need to fix the lengths
    auto d = new int[2*cols];

    d[0] = 0;
    d[cols] = 1;
    for (int i = 0; i < trgLen_ + 1; i++) d[i] = i;
    for (int i = 1; i < srcLen_ + 1; i++) {
        for (int j = 1; j < trgLen_ + 1; j++) {
            d[(i&1)*cols + j] = std::min(std::min(d[((i-1)&1)*cols + j], d[(i&1)*cols + (j-1)]) + 1, 
			    		 d[((i-1)&1)*cols + (j-1)] + (src_[i-1] == trg_[j-1] ? 0 : 1));
        }
    }

    *result_ = d[(srcLen_&1)*cols + trgLen_];
    delete(d);
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

    // see https://github.com/pytorch/pytorch/issues/21819
    // to avoid random errors when executing on cuda:1 we need to set the device manually
    c10::cuda::set_device(static_cast<c10::DeviceIndex>(src.device().index()));

#ifdef DEBUG 
    TimingGPU timerGPU;
    timerGPU.StartCounter(); 
#endif

    AT_DISPATCH_ALL_TYPES(
        src.scalar_type(),
        "editdistance_cuda",
        ([&] {
         distance_cuda_kernel<scalar_t><<<numBatch, threads>>>(
            src.data<scalar_t>(),
            trg.data<scalar_t>(),
            result.data<int>(),
            srcLen, 
            trgLen, 
	    padToken);
        }));

#ifdef DEBUG 
    std::cout << "GPU Timing = " << timerGPU.GetCounter() << " ms  " << std::flush;
#endif

    return result;
}
