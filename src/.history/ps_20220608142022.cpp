#include <torch/extension.h>

#include <iostream>
/**
 * @brief Code follows the pattern defined here
 * https://github.com/pytorch/extension-cpp
 * 
 * @param np
 * @param alpha
 * @param beta
 * @return torch::Tensor 
 */
torch::Tensor zwgj(int np, torch::Scalar alpha, torch::Scalar beta) {
  
}
torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("zwgj", &zwgj, "zwgj");
}