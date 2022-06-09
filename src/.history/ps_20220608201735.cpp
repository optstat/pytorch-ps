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
torch::Tensor pyzwgj(int np, double alpha, double beta) {
    std::vector<double> vzs(np), vws(np);
    zwgj(&vzs[0], &vws[0], np , alpha , beta);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor zs = torch::from_blob(&vzs[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor ws = torch::from_blob(&vws[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor zpw = torch::stack({zs, ws});
    zpw = zpw.contiguous();
    return torch::clone(zpw);
}

torch::Tensor pyzwgrjm(int np, double alpha, double beta){
    std::vector<double> vzs(np), vws(np);
    zwgrjm(&vzs[0], &vws[0], np , alpha , beta);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor zs = torch::from_blob(&vzs[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor ws = torch::from_blob(&vws[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor zpw = torch::stack({zs, ws});
    zpw = zpw.contiguous();
    return torch::clone(zpw);
}

torch::Tensor pyzwgrjp(double *z, double *w, int np, double alpha, double beta){
    std::vector<double> vzs(np), vws(np);
    zwgrjp(&vzs[0], &vws[0], np , alpha , beta);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor zs = torch::from_blob(&vzs[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor ws = torch::from_blob(&vws[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor zpw = torch::stack({zs, ws});
    zpw = zpw.contiguous();
    return torch::clone(zpw);
}


torch::Tensor pyzwglj(double *z, double *w, int np, double alpha, double beta){
    std::vector<double> vzs(np), vws(np);
    zwglj(&vzs[0], &vws[0], np , alpha , beta);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor zs = torch::from_blob(&vzs[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor ws = torch::from_blob(&vws[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor zpw = torch::stack({zs, ws});
    zpw = zpw.contiguous();
    return torch::clone(zpw);
}

torch::Tensor pyDgj(int np,double alpha, double beta){
  std::vector<double> D(np*np), Dt(np*np), z(np);
  Dgj(D.data(), Dt.data(), z.data(), np,alpha, beta);
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor Dt = torch::from_blob(&D[0], {np,np}, opts).to(torch::kFloat64); //This assumes row ordering
  Dt = Dt.contiguous();
  return torch::clone(Dt);
}

torch::Tensor pyDgrjm(int np, double alpha, double beta){
  std::vector<double> D(np*np), Dt(np*np), z(np);
  Dgrjm(D.data(), Dt.data(), z.data(), np,alpha, beta);
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor Dt = torch::from_blob(&D[0], {np,np}, opts).to(torch::kFloat64); //This assumes row ordering
  Dt = Dt.contiguous();
  return torch::clone(Dt);
}

torch::Tensor pyDgrjp(int np, double alpha, double beta){
  std::vector<double> D(np*np), Dt(np*np), z(np);
  Dgrjp(D.data(), Dt.data(), z.data(), np,alpha, beta);
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor Dt = torch::from_blob(&D[0], {np,np}, opts).to(torch::kFloat64); //This assumes row ordering
  Dt = Dt.contiguous();
  return torch::clone(Dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pyzwgj", &pyzwgj, "pyzwgj");
  m.def("pyzwgrjm", &pyzwgrjm, "pyzwgrjm");
  m.def("pyzwgrjp", &pyzwgrjp, "pyzwgrjp");
  m.def("pyzwglj", &pyzwglj, "pyzwglj");
  m.def("pyDgj", &pyDgj, "pyDgj");
  m.def("pyDgrjm", &pyDgrjm, "pyDgrjm");
  m.def("pyDgrjp", &pyDgrjp, "pyDgrjp");
}