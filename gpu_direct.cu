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
  const double e14fac = 1.5;
  const int ncoord = 23558;
  const double ref_vpress[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  Force<float> force_main("test_data/force_direct_main.txt");
  Force<float> force_total("test_data/force_direct.txt");
  Force<long long int> force_fp(ncoord);
  Force<float> force(ncoord);

  const int nin14list = 6556;
  const int nex14list = 28153;
  list14_t *in14list = new list14_t[nin14list];
  list14_t *ex14list = new list14_t[nex14list];
  load_ind<int>(3, "test_data/in14list.txt", nin14list, (int *)in14list);
  load_ind<int>(3, "test_data/ex14list.txt", nex14list, (int *)ex14list);

  force_fp.clear();

  // Load coordinates
  XYZQ xyzq("test_data/xyzq.txt", 32);
  XYZQ xyzq_unsorted("test_data/xyzq_unsorted.txt", 32);
  XYZQ xyzq_sorted_ref("test_data/xyzq_sorted.txt", 32);
  XYZQ xyzq_sorted(ncoord, 32);

  double max_diff;
  double tol;

  // ------------------- Neighborlist -----------------

  NeighborList<32> nlist_ref(1, 1, 1);
  nlist_ref.load("test_data/nlist.txt");
  //nlist.remove_empty_tiles();
  //nlist.split_dense_sparse(512);
  nlist_ref.analyze();

  int *loc2glo_ind = new int[ncoord];
  load_ind<int>(1, "test_data/loc2glo.txt", ncoord, loc2glo_ind);
  for (int i=0;i < ncoord;i++) loc2glo_ind[i]--;

  int zone_patom[8] = {23558, 23558, 23558, 23558, 23558, 23558, 23558, 23558};
  float3 min_xyz[8], max_xyz[8];
  min_xyz[0].x = -31.74800;
  min_xyz[0].y = -31.77600;
  min_xyz[0].z = -31.77900;
  max_xyz[0].x = 31.73900;
  max_xyz[0].y = 31.80500;
  max_xyz[0].z = 31.80300;

  NeighborList<32> nlist(1, 1, 1);
  nlist.sort(zone_patom, max_xyz, min_xyz, xyzq_unsorted.xyzq, xyzq_sorted.xyzq);
  nlist.build(boxx, boxy, boxz, roff, xyzq_sorted.xyzq);

  //tol = 7.71e-4;
  //if (!xyzq_sorted_ref.compare(xyzq_sorted, tol, max_diff)) {
  //}

  // ------------------- Non-bonded -----------------

  DirectForce<long long int, float> dir;
  dir.setup(boxx, boxy, boxz, kappa, roff, ron, e14fac, VDW_VSH, EWALD, true, true);
  dir.set_vdwparam("test_data/vdwparam.txt");
  dir.set_vdwtype("test_data/vdwtype.txt");
  dir.calc_force(xyzq.xyzq, &nlist_ref, false, false, force_fp.xyz.stride, force_fp.xyz.data);
  force_fp.convert(&force);
  tol = 7.71e-4;
  if (!force_main.compare(&force, tol, max_diff)) {
    std::cout<<"Non-bonded (main) force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"Non-bonded (main) force comparison OK"<<std::endl;
    std::cout<<"(tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

  dir.set_vdwparam14("test_data/vdwparam14.txt");
  dir.set_14_list(nin14list, nex14list, in14list, ex14list);
  dir.calc_14_force(xyzq.xyzq, false, false, force_fp.xyz.stride, force_fp.xyz.data);
  force_fp.convert(&force);
  tol = 7.71e-4;
  if (!force_total.compare(&force, tol, max_diff)) {
    std::cout<<"Non-bonded (total) force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"Non-bonded (total) force comparison OK"<<std::endl;
    std::cout<<"(tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

  delete [] in14list;
  delete [] ex14list;

  delete [] loc2glo_ind;

  return;

  //------------------ Virial pressure ---------------

  /*
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
  */

}
