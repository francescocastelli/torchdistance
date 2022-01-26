#pragma once
#include <torch/extension.h>

torch::Tensor editdistance(
    const torch::Tensor& src, 
    const torch::Tensor& trg);

