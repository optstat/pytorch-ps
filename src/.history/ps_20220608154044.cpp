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
torch::Tensor zwgj(int np, torch::Scalar alpha, torch::Scalar beta) {
    std::vector<double> vzs(np), vws(np);
    zwgj    (&vzs[0], &vws[0], np , alpha , beta);
    for ( int i=0; i < np; i++) {
        zs[i]=vzs[i];
        ws[i]=vws[i];
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor zs = torch::from_blob(&vzs[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor ws = torch::from_blob(&vws[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor zpw = torch::stack({zs, ws});
    zpw = zpw.contiguous();
    return torch::clone(t);
}

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
    std::vector<double> vzs(np), vws(np);
    zwgj    (&vzs[0], &vws[0], np , alpha , beta);
    for ( int i=0; i < np; i++) {
        zs[i]=vzs[i];
        ws[i]=vws[i];
    }
    torch::Tensor zs = torch::from_blob(&zs[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor ws = torch::from_blob(&ws[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor zpw = torch::stack({zs, ws});
    zpw = zpw.contiguous();
    return torch::clone(t);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("zwgj", &zwgj, "zwgj");
}