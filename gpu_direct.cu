#include <iostream>
#include <fstream>
#include <cuda.h>
#include "cuda_utils.h"
#ifdef USE_MPI
#include "mpi_utils.h"
#endif
#include "XYZQ.h"
#include "Force.h"
#include "NeighborList.h"
#include "DirectForce.h"
#include "BondedForce.h"
#include "VirialPressure.h"

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
// Loads indices from file
//
template <typename T>
void load_ind(const int nind, const char *filename, const int n, T *ind) {
  std::ifstream file(filename);
  if (file.is_open()) {

    for (int i=0;i < n;i++) {
      for (int k=0;k < nind;k++) {
	if (!(file >> ind[i*nind+k])) {
	  std::cerr<<"Error reading file "<<filename<<std::endl;
	  exit(1);
	}
      }
    }

  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }

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
  const double ref_vpress[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  Force<float> force_nonbond("test_data/force_direct.txt");
  Force<float> force_bonded("test_data/force_bonded.txt");
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

  // ------------------- Non-bonded -----------------

  DirectForce<long long int, float> dir;
  dir.setup(boxx, boxy, boxz, kappa, roff, ron, VDW_VSH, EWALD, true, true);
  dir.set_vdwparam("test_data/vdwparam.txt");
  dir.set_vdwtype("test_data/vdwtype.txt");
  dir.calc_force(ncoord, xyzq.xyzq, &nlist, false, false, force_fp.xyz.stride, force_fp.xyz.data);

  force_fp.convert(&force);

  double max_diff;
  double tol = 7.71e-4;
  if (!force_nonbond.compare(&force, tol, max_diff)) {
    std::cout<<"Non-bonded force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"Non-bonded force comparison OK (tolerance " << tol << " max difference " 
	     << max_diff << ")" << std::endl;
  }

  //---------------------- Bonded -------------------
  const int nbondlist = 0;
  const int nanglelist = 0;
  bondlist_t *h_bondlist = new bondlist_t[nbondlist];
  float2 *h_bondcoef = new float2[nbondlist];
  load_ind<int>(4, "test_data/bondlist.txt", nbondlist, (int *)h_bondlist);
  load_ind<float>(2, "test_data/bondcoef.txt", nbondlist, (float *)h_bondcoef);

  anglelist_t *h_anglelist = new anglelist_t[nanglelist];
  float2 *h_anglecoef = new float2[nanglelist];
  load_ind<int>(6, "test_data/anglelist.txt", nanglelist, (int *)h_anglelist);
  load_ind<float>(2, "test_data/anglecoef.txt", nanglelist, (float *)h_anglecoef);

  force_fp.clear();
  BondedForce<long long int, float> bondedforce;
  bondedforce.setup(nbondlist, h_bondlist, h_bondcoef,
		    nanglelist, h_anglelist, h_anglecoef);
  bondedforce.calc_force(xyzq.xyzq, boxx, boxy, boxz, false, false,
			 force_fp.xyz.stride, force_fp.xyz.data);
  force_fp.convert(&force);

  tol = 1.0;
  if (!force_bonded.compare(&force, tol, max_diff)) {
    std::cout<<"Bonded force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"Bonded force comparison OK (tolerance " << tol << " max difference " 
	     << max_diff << ")" << std::endl;
  }

  delete [] h_bondlist;
  delete [] h_bondcoef;
  
  delete [] h_anglelist;
  delete [] h_anglecoef;
  
  return;

  //------------------ Virial pressure ---------------

  VirialPressure vir;
  double vpress[9];
  cudaXYZ<double> coord;
  cudaXYZ<double> force_double(ncoord, force_fp.xyz.stride, (double *)force_fp.xyz.data);
  float3 *xyz_shift = NULL;
  vir.calc_virial(&coord, &force_double, xyz_shift, boxx, boxy, boxz, vpress);
  force_double.data = NULL;
  
  tol = 1.0e-5;
  max_diff = 0.0;
  for (int i=0;i < 9;i++) {
    double diff = fabs(ref_vpress[i] - vpress[i]);
    max_diff = max(max_diff, diff);
  }
  if (max_diff > tol) {
    std::cout<<"vpress comparison FAILED"<<std::endl;
  } else {
    std::cout<<"vpress comparison OK (tolerance " << tol << " max difference "
	     << max_diff << ")" << std::endl;
  }

}
