#include <torch/extension.h>

#include <iostream>
/**
 * @brief Code follows the pattern defined here
 * https://github.com/pytorch/extension-cpp
 * 
 * @param z 
 * @param w 
 * @param np
 * @param alpha
 * @param beta
 * @return torch::Tensor 
 */
torch::Tensor zwgj (double *z, double *w, int np, double alpha, double beta) {

}
torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}