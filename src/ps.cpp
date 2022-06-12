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
    Polylib::zwgj(&vzs[0], &vws[0], np , alpha , beta);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor zs = torch::from_blob(&vzs[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor ws = torch::from_blob(&vws[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor zpw = torch::stack({zs, ws});
    zpw = zpw.contiguous();
    return torch::clone(zpw);
}

torch::Tensor pyzwgrjm(int np, double alpha, double beta){
    std::vector<double> vzs(np), vws(np);
    Polylib::zwgrjm(&vzs[0], &vws[0], np , alpha , beta);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor zs = torch::from_blob(&vzs[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor ws = torch::from_blob(&vws[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor zpw = torch::stack({zs, ws});
    zpw = zpw.contiguous();
    return torch::clone(zpw);
}

torch::Tensor pyzwgrjp(int np, double alpha, double beta){
    std::vector<double> vzs(np), vws(np);
    Polylib::zwgrjp(&vzs[0], &vws[0], np , alpha , beta);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor zs = torch::from_blob(&vzs[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor ws = torch::from_blob(&vws[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor zpw = torch::stack({zs, ws});
    zpw = zpw.contiguous();
    return torch::clone(zpw);
}


torch::Tensor pyzwglj(int np, double alpha, double beta){
    std::vector<double> vzs(np), vws(np);
    Polylib::zwglj(&vzs[0], &vws[0], np , alpha , beta);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor zs = torch::from_blob(&vzs[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor ws = torch::from_blob(&vws[0], {np}, opts).to(torch::kFloat64); //This assumes row ordering
    torch::Tensor zpw = torch::stack({zs, ws});
    zpw = zpw.contiguous();
    return torch::clone(zpw);
}

torch::Tensor pyDgj(int np,double alpha, double beta){
  std::vector<double> vzs(np), vws(np);
  Polylib::zwgj(&vzs[0], &vws[0], np , alpha , beta);  
  std::vector<double> D(np*np, 0.0);
  Polylib::Dgj(&D[0], &vzs[0], np, alpha, beta);
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor DM = torch::from_blob(&D[0], {np,np}, opts).to(torch::kFloat64); //The matrix is column ordered so we have to take the transpse
  torch::Tensor DMt = torch::transpose(DM,0,1);
  DMt = DMt.contiguous();
  return torch::clone(DMt);
}

torch::Tensor pyDgrjm(int np, double alpha, double beta){
  std::vector<double> vzs(np), vws(np);
  Polylib::zwgrjm(&vzs[0], &vws[0], np , alpha , beta);  
  std::vector<double> D(np*np, 0.0);
  Polylib::Dgrjm(&D[0], &vzs[0], np,alpha, beta);
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor DM = torch::from_blob(&D[0], {np,np}, opts).to(torch::kFloat64); //This assumes row ordering
  torch::Tensor DMt = torch::transpose(DM,0,1);
  DMt = DMt.contiguous();
  return torch::clone(DMt);

}

torch::Tensor pyDgrjp(int np, double alpha, double beta){
  std::vector<double> vzs(np), vws(np);
  Polylib::zwgrjp(&vzs[0], &vws[0], np , alpha , beta);  
  
  std::vector<double> D(np*np, 0.0);
  Polylib::Dgrjp(&D[0], &vzs[0], np,alpha, beta);
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor DM = torch::from_blob(&D[0], {np,np}, opts).to(torch::kFloat64); //This assumes row ordering
  torch::Tensor DMt = torch::transpose(DM,0,1);
  DMt = DMt.contiguous();
  return torch::clone(DMt);

}

torch::Tensor pyDglj(int np, double alpha, double beta){
  std::vector<double> vzs(np), vws(np);
  Polylib::zwglj(&vzs[0], &vws[0], np , alpha , beta);  
  std::vector<double> D(np*np, 0.0);
  Polylib::Dglj(&D[0], &vzs[0], np,alpha, beta);
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor DM = torch::from_blob(&D[0], {np,np}, opts).to(torch::kFloat64); //This assumes row ordering
  torch::Tensor DMt = torch::transpose(DM,0,1);
  DMt = DMt.contiguous();
  return torch::clone(DMt);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pyzwgj", &pyzwgj, "pyzwgj");
  m.def("pyzwgrjm", &pyzwgrjm, "pyzwgrjm");
  m.def("pyzwgrjp", &pyzwgrjp, "pyzwgrjp");
  m.def("pyzwglj", &pyzwglj, "pyzwglj");
  m.def("pyDgj", &pyDgj, "pyDgj");
  m.def("pyDgrjm", &pyDgrjm, "pyDgrjm");
  m.def("pyDgrjp", &pyDgrjp, "pyDgrjp");
  m.def("pyDglj", &pyDglj, "pyDglj");
}