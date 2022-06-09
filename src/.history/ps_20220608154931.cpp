#include <torch/extension.h>
#include <iostream>
#include "polylib.hpp"

/**
 * @brief Code follows the pattern defined here
 * https://github.com/pytorch/extension-cpp
 * 
 * @param np
 * @param alpha
 * @param beta
 * @return torch::Tensor 
 */
torch::Tensor zwgj(int np, double alpha, double beta) {
    std::vector<double> vzs{np}, vws{np};
    polylib::zwgj(&vzs[0], &vws[0], np , alpha , beta);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor zs = torch::from_blob(&vzs[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor ws = torch::from_blob(&vws[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor zpw = torch::stack({zs, ws});
    zpw = zpw.contiguous();
    return torch::clone(zpw);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("zwgj", &zwgj, "zwgj");
}