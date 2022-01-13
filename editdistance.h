#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> editdistance(torch::Tensor src, torch::Tensor trg);

