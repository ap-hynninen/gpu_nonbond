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
  const double roff = 9.0;
  const double ron = 7.5;
  const int ncoord = 23558;

  Force<float> force_comp("test_data/force_direct.txt");
  Force<long long int> force_fp(ncoord);
  Force<float> force(ncoord);

  force_fp.clear();

  // Load coordinates
  XYZQ xyzq("test_data/xyzq.txt", 32);

  NeighborList<32> nlist;
  nlist.load("test_data/nlist.txt");
  //nlist.remove_empty_tiles();
  //nlist.split_dense_sparse(512);
  nlist.analyze();

  DirectForce<long long int, float> dir;
  dir.setup(boxx, boxy, boxz, kappa, roff, ron, VDW_VSH, EWALD, true, true);
  dir.set_vdwparam("test_data/vdwparam.txt");
  dir.set_vdwtype("test_data/vdwtype.txt");
  dir.calc_force(ncoord, xyzq.xyzq, &nlist, false, false, force_fp.xyz.stride, force_fp.xyz.data);

  force_fp.convert(&force);

  double max_diff;
  double tol = 7.71e-4;
  if (!force_comp.compare(&force, tol, max_diff)) {
    std::cout<<"force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"force comparison OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

}
