#include <iostream>
#include <cuda.h>
#include "cuda_utils.h"
#ifdef USE_MPI
#include "mpi_utils.h"
#endif
#include "XYZQ.h"
#include "Force.h"
#include "NeighborList.h"
#include "DirectForce.h"

void test();

int numnode = 1;
int mynode = 0;

int main(int argc, char *argv[]) {

#ifdef USE_MPI
  start_mpi(argc, argv, numnode, mynode);
#endif

  start_gpu(numnode, mynode);
  
  test();

#ifdef USE_MPI
  stop_mpi();
#endif

  return 0;
}

//
// Test the code using data in test_data/ -directory
//
void test() {

  // Settings for the data:
  const double boxx = 62.23;
  const double boxy = 62.23;
  const double boxz = 62.23;
  const double kappa = 0.320;
  const int ncoord = 23558;
  const int nfftx = 64;
  const int nffty = 64;
  const int nfftz = 64;
  const int order = 4;

  // Setup reciprocal vectors
  double recip[9];
  for (int i=0;i < 9;i++) recip[i] = 0;
  recip[0] = 1.0/boxx;
  recip[4] = 1.0/boxy;
  recip[8] = 1.0/boxz;

  Force<float> force_comp("test_data/force_direct.txt");
  Force<float> force(ncoord);

  // Load coordinates
  XYZQ xyzq("test_data/xyzq.txt");

  double max_diff;
  double tol = 3.2e-4;
  if (!force_comp.compare(&force, tol, max_diff)) {
    std::cout<<"force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"force comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

}
