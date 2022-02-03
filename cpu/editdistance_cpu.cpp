#include <torch/extension.h> 

namespace {

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
    for (int i=0; i < srcLen; i++)
    {
	    if (src[i] == padToken)
	    {
		    srcLen = i;
		    break;
	    }
    }

    for (int i=0; i < trgLen; i++)
    {
	    if (trg[i] == padToken)
	    {
		    trgLen = i;
		    break;
	    }
    }

    std::vector<std::vector<int32_t>> d(2, std::vector<int32_t>(trgLen+1));

    d[0][0] = 0;
    d[1][0] = 1;
    for (int i = 1; i < trgLen + 1; i++) d[0][i] = i;
    for (int i=1; i < srcLen + 1; i++) {
        for (int j=1; j < trgLen + 1; j++) {
            d[i&1][j] = std::min(std::min(d[(i-1)&1][j], d[i&1][j-1]) + 1, d[(i-1)&1][j-1] + (src[i-1] == trg[j-1] ? 0 : 1));
        }
    }

    *result = d[srcLen&1][trgLen];
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
    torch::Tensor& result, 
    int64_t padToken){

    auto numBatch = src.size(0);
    auto srcLen = src.size(1);
    auto trgLen = trg.size(1);

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
