# pytorch-ps
Pseudospectral utility methods adapted for use in the pytorch framework.
To build change to the src directory and simply invoke 

**python3 setup.py build**

To build and install for the current user invoke '

**python3 setup.py install --user**



The following functions can be utilized after importing from module ps_cpp

**pyzwgj(np, alpha, beta)** -Find the Gauss Jacobi Zero roots and Weights at np intervals given Jacobi polynomial parameters alpha and beta.  This will return a single tensor with the first dimension being the Zeroes and the second dimension containing the Weights.

**pyzwgrjm(np, alpha, beta)** -Find the Gauss-Radau-Jacobi Zero roots and Weights at np intervals given Jacobi polynomial parameters alpha and beta with end point at z=-1.  This will return a single tensor with the first dimension being the Zeroes and the second dimension containing the Weights.

**pyzwgrjm(np, alpha, beta)** -Find the Gauss-Radau-Jacobi Zero roots and Weights at np intervals given Jacobi polynomial parameters alpha and beta with end point at z=1.  This will return a single tensor with the first dimension being the Zeroes and the second dimension containing the Weights.

**pyzwgrjm(np, alpha, beta)** -Find the Gauss-Lobatto-Jacobi Zero roots and Weights at np intervals given Jacobi polynomial parameters alpha and beta with end points at z=-1 and z=1.  This will return a single tensor with the first dimension being the Zeroes and the second dimension containing the Weights.

**pyDgj(np, alpha, beta)** Compute the Derivative Matrix associated with the Gauss-Jacobi zeros at np intervals given Jacobi polynomial parameters alpha and beta.

**pyDgrjm(np, alpha, beta)** Compute the Derivative Matrix associated with the Gauss-Radau-Jacobi zeros with a zero at z=-1 at np intervals given Jacobi polynomial parameters alpha and beta.

**pyDgrjp(np, alpha, beta)** Compute the Derivative Matrix associated with the Gauss-Radau-Jacobi zeros with a zero at z=1 at np intervals given Jacobi polynomial parameters alpha and beta.

**pyDgrjp(np, alpha, beta)** Compute the Derivative Matrix with the Gauss-Lobatto-Jacobi zeros at np intervals given Jacobi polynomial parameters alpha and beta.
