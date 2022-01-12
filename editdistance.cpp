#include <torch/extension.h>


// https://github.com/roy-ht/editdistance
template <typename scalar_t>
static void wer_single_batch_frame(
    scalar_t* const src, 
    scalar_t* const trg, 
    int64_t* result,
    int64_t srcLen,
    int64_t trgLen) {
    
    std::vector<std::vector<int64_t>> d(2, std::vector<int64_t>(trgLen+1));

    d[0][0] = 0;
    d[1][0] = 1;
    for (int i = 0; i < trgLen + 1; i++) d[0][i] = i;
    for (int i = 1; i < srcLen + 1; i++) {
        for (int j = 1; j < trgLen + 1; j++) {
            d[i&1][j] = min(min(d[(i-1)&1][j], d[i&1][j-1]) + 1, d[(i-1)&1][j-1] + (src[i-1] == trg[j-1] ? 0 : 1));
        }
    }

    result = d[size1&1][size2];
}

template <typename scalar_t>
static void wer_frame(
    scalar_t* const src, 
    scalar_t* const trg, 
    int64_t* result,
    int64_t srcLen,
    int64_t trgLen, 
    int64_t numBatch) {

    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
          std::vector<int64_t> range(end);
          std::iota(range.begin(), range.end(), start);
          for (const auto batch : range) {
            wer_single_batch_frame<scalar_t>(
                src + batch * srcLen, 
                trg + batch * trgLen,
                result,
                srcLen, 
                trgLen
            );
          }
    });
}

Tensor wer(
    at::Tensor src, 
    at::Tensor trg){

    // all the checks are done here
    auto numBatch = src.size(0);
    auto srcLen = src.size(1);
    auto trgLen = trg.size(1);

    Tensor result = at::empty({1});

    AT_DISPATCH_FLOATING_TYPES(
        src.scalar_type(),
        "wer",
        [&] {
          wer_frame<scalar_t>(
            src.data_ptr<scalar_t>(),
            trg.data_ptr<scalar_t>(),
            result.data_prt<int64_t>(),
            srcLen, 
            trgLen,
            numBatch
          );
        }
    );
}
