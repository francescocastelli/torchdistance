#include <torch/extension.h>
#include "editdistance.h"

torch::Tensor
editdistance_cuda_kernel(const torch::Tensor& src, const torch::Tensor& trg);

torch::Tensor editdistance_cuda(
    const torch::Tensor& src, 
    const torch::Tensor& trg) {

    // all cuda checks go here
    return editdistance_cuda(src, trg);
}

TORCH_LIBRARY_IMPL(editdistance, CUDA, m) {
  m.impl("editdistance", &editdistance_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("editdistance", &editdistance_cuda, "editdistance cuda forward");
}
