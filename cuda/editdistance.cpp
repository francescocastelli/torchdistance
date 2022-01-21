#include <torch/extension.h>
#include "editdistance.h"

torch::Tensor
editdistance_cuda(const torch::Tensor& src, const torch::Tensor& trg);

torch::Tensor editdistance(
    const torch::Tensor& src, 
    const torch::Tensor& trg) {

    // all cuda checks go here
    return editdistance_cuda(src, trg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("editdistance", &editdistance, "editdistance forward cuda");
}
