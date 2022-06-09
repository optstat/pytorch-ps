#include <torch/extension.h>

#include <iostream>
/**
 * @brief Code follows the pattern defined here
 * https://github.com/pytorch/extension-cpp
 * 
 * @param z 
 * @return torch::Tensor 
 */

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}