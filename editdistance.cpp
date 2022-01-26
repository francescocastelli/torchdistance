#include "editdistance.h"

torch::Tensor editdistance_cpu(
    const torch::Tensor& src, 
    const torch::Tensor& trg);

torch::Tensor editdistance_cuda(
    const torch::Tensor& src, 
    const torch::Tensor& trg);

torch::Tensor editdistance(
    const torch::Tensor& src, 
    const torch::Tensor& trg)
{
    if (src.device() == torch::kCPU)
    {
	return editdistance_cpu(src, trg);
    }

    if (src.device() == torch::kCUDA)
    {
	return editdistance_cuda(src, trg);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("editdistance", &editdistance, "editdistance forward");
}
