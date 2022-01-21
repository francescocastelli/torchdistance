#pragma once
#include <torch/extension.h>
#include <tuple>

torch::Tensor
editdistance(const torch::Tensor& src, const torch::Tensor& trg);
