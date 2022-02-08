#include <torch/extension.h> 

namespace {

template <typename scalar_t>
int64_t handlePadLen(scalar_t* str, int64_t strLen, int64_t padToken) {
    for (int i=0; i < strLen; i++)
	    if (str[i] == padToken) return i;

    return strLen;
}

// https://github.com/roy-ht/editdistance
template <typename scalar_t>
static void distance_single_batch_frame(
    scalar_t* const src, 
    scalar_t* const trg, 
    int32_t* result,
    int64_t srcLen,
    int64_t trgLen, 
    int64_t padToken) {

    // handle padding
    srcLen = handlePadLen(src, srcLen, padToken);
    trgLen = handlePadLen(trg, trgLen, padToken);

    // base case
    if (srcLen == 0) { *result = trgLen; return; }
    if (trgLen == 0) { *result = srcLen; return; }

    auto src_ = src, trg_ = trg;
    auto srcLen_ = srcLen, trgLen_ = trgLen;
    if (trgLen < srcLen) src_ = trg, trg_ = src, srcLen_ = trgLen, trgLen_ = srcLen;

    std::vector<std::vector<int32_t>> d(2, std::vector<int32_t>(trgLen_+1));

    d[0][0] = 0;
    d[1][0] = 1;
    for (int i = 1; i < trgLen_ + 1; i++) d[0][i] = i;
    for (int i=1; i < srcLen_ + 1; i++) {
        for (int j=1; j < trgLen_ + 1; j++) {
            d[i&1][j] = std::min(std::min(d[(i-1)&1][j], d[i&1][j-1]) + 1, d[(i-1)&1][j-1] + (src_[i-1] == trg_[j-1] ? 0 : 1));
        }
    }

    *result = d[srcLen_&1][trgLen_];
}

template <typename scalar_t>
static void distance_frame(
    scalar_t* const src, 
    scalar_t* const trg, 
    int32_t* result,
    int64_t srcLen,
    int64_t trgLen, 
    int64_t numBatch, 
    int64_t padToken) {

    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
        for (const auto batch : c10::irange(start, end)) {
            distance_single_batch_frame<scalar_t>(
                src + batch * srcLen, 
                trg + batch * trgLen,
                result + batch,
                srcLen, 
                trgLen,
		padToken
            );
        }
    });
}
}

torch::Tensor editdistance_cpu(
    const torch::Tensor& src, 
    const torch::Tensor& trg, 
    int64_t padToken){

    auto numBatch = src.size(0);
    auto srcLen = src.size(1);
    auto trgLen = trg.size(1);

    at::TensorOptions options(src.device());
    options = options.dtype(at::ScalarType::Int);
    auto result = at::empty({numBatch, 1}, options);

    AT_DISPATCH_ALL_TYPES(
        src.scalar_type(),
        "editdistance_cpu",
        [&] {
            distance_frame<scalar_t>(
            src.data_ptr<scalar_t>(),
            trg.data_ptr<scalar_t>(),
            result.data_ptr<int32_t>(),
            srcLen, 
            trgLen,
            numBatch,
	    padToken
          );
        }
    );

    return result;
}

TORCH_LIBRARY_IMPL(editdistance, CPU, m) {
  m.impl("editdistance", editdistance_cpu);
}
