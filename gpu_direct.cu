#include <iostream>
#include <fstream>
#include <cuda.h>
#include "cuda_utils.h"
#include "XYZQ.h"
#include "Force.h"
#include "NeighborList.h"
#include "CudaPMEDirectForce.h"
#include "CudaPMEDirectForceBlock.h"
#include "VirialPressure.h"

void test();

int numnode = 1;
int mynode = 0;

int main(int argc, char *argv[]) {

  start_gpu(numnode, mynode);
  
  test();

  stop_gpu();

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
  const double rcut = 11.0;
  const double roff = 9.0;
  const double ron = 7.5;
  const double e14fac = 1.5;
  const int ncoord = 23558;
  const double ref_virtensor[9] = {58748.2766568620, 159.656334638237, 483.080609561938,
				   159.656334638202, 57272.9410562695, 894.635309171291,
				   483.080609561938, 894.635309171288, 56639.3675265570};
  const double ref_vir = 57553.5284132295;
  const double ref_energy_vdw = 8198.14425;
  const double ref_energy_elec = -73396.45998;

  Force<float> force_main("test_data/force_direct_main.txt");
  Force<float> force_total("test_data/force_direct.txt");
  Force<long long int> force_fp(ncoord);
  Force<float> force(ncoord);

  const int nin14list = 6556;
  const int nex14list = 28153;
  xx14list_t *in14list = new xx14list_t[nin14list];
  xx14list_t *ex14list = new xx14list_t[nex14list];
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

  const int niblo14 = 23558;
  const int ninb14 = 34709;
  int *iblo14 = new int[niblo14];
  int *inb14 = new int[ninb14];
  load_ind<int>(1, "test_data/iblo14.txt", niblo14, iblo14);
  load_ind<int>(1, "test_data/inb14.txt", ninb14, inb14);

  NeighborList<32> nlist_ref(ncoord, "test_data/nlist.txt");
  //nlist.remove_empty_tiles();
  //nlist.split_dense_sparse(512);
  std::cout << "============== nlist_ref ==============" << std::endl;
  nlist_ref.analyze();
  std::cout << "=======================================" << std::endl;

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

  int *h_loc2glo = new int[ncoord];
  for (int i=0;i < ncoord;i++) h_loc2glo[i] = i;
  int *loc2glo = NULL;
  allocate<int>(&loc2glo, ncoord);
  copy_HtoD<int>(h_loc2glo, loc2glo, ncoord);
  delete [] h_loc2glo;

  NeighborList<32> nlist(ncoord, iblo14, inb14);
  //nlist.sort(zone_patom, max_xyz, min_xyz, xyzq_unsorted.xyzq, xyzq_sorted.xyzq);
  nlist.sort(zone_patom, xyzq_unsorted.xyzq, xyzq_sorted.xyzq, loc2glo);
  nlist.build(boxx, boxy, boxz, rcut, xyzq_sorted.xyzq, loc2glo);
  nlist.test_build(zone_patom, boxx, boxy, boxz, rcut, xyzq_sorted.xyzq, loc2glo);

  std::cout << "================ nlist ================" << std::endl;
  nlist.analyze();
  std::cout << "=======================================" << std::endl;

  deallocate<int>(&loc2glo);

  //tol = 7.71e-4;
  //if (!xyzq_sorted_ref.compare(xyzq_sorted, tol, max_diff)) {
  //}

  // ------------------- Non-bonded -----------------

  CudaPMEDirectForce<long long int, float> dir;
  dir.setup(boxx, boxy, boxz, kappa, roff, ron, e14fac, VDW_VSH, EWALD);
  dir.set_vdwparam(1260, "test_data/vdwparam.txt");
  dir.set_vdwtype(ncoord, "test_data/vdwtype.txt");
  dir.calc_force(xyzq.xyzq, &nlist_ref, false, false, force_fp.xyz.stride, force_fp.xyz.data);
  force_fp.convert(&force);
  tol = 7.71e-4;
  if (!force_main.compare(&force, tol, max_diff)) {
    std::cout<<"Non-bonded (main) force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"Non-bonded (main) force comparison OK"<<std::endl;
    std::cout<<"(tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

  dir.set_vdwparam14(1260, "test_data/vdwparam14.txt");
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

  // Check energy and virial
  force_fp.clear();
  dir.clear_energy_virial();
  dir.calc_force(xyzq.xyzq, &nlist_ref, true, true, force_fp.xyz.stride, force_fp.xyz.data);
  dir.calc_virial(ncoord, xyzq.xyzq, force_fp.xyz.stride, force_fp.xyz.data);

  double energy_vdw;
  double energy_elec;
  double energy_excl;
  double virtensor[9];
  dir.get_energy_virial(true, true, &energy_vdw, &energy_elec, &energy_excl, virtensor);
  double vir = (virtensor[0] + virtensor[4] + virtensor[8])/3.0;
  std::cout << "energy_vdw = " << energy_vdw << " energy_elec = " << energy_elec << std::endl;
  std::cout << "vir = " << vir << " virtensor=" << std::endl;
  max_diff = 0.0;
  for (int j=0;j < 3;j++) {
    for (int i=0;i < 3;i++) {
      double diff = fabs(virtensor[j*3+i] - ref_virtensor[j*3+i]);
      max_diff = max(max_diff, diff);
      std::cout << virtensor[j*3+i] << " (" << diff/ref_virtensor[j*3+i] << ") ";
    }
    std::cout << std::endl;
  }

  if (max_diff < 3.13) {
    std::cout << "Nonbonded virial comparison OK" << std::endl;
  } else {
    std::cout << "Nonbonded virial comparison FAILED" << std::endl;
  }
  std::cout << "max_diff(vir_tensor) = " << max_diff << std::endl;
  std::cout << "max_diff(vir) = " << fabs(vir - ref_vir) << std::endl;

  //--------------- Non-bonded using GPU build neighborlist -----------
  force_fp.clear();
  dir.clear_energy_virial();
  dir.calc_force(xyzq_sorted.xyzq, &nlist, true, true, force_fp.xyz.stride, force_fp.xyz.data);
  dir.calc_virial(ncoord, xyzq_sorted.xyzq, force_fp.xyz.stride, force_fp.xyz.data);

  dir.get_energy_virial(true, true, &energy_vdw, &energy_elec, &energy_excl, virtensor);
  vir = (virtensor[0] + virtensor[4] + virtensor[8])/3.0;
  std::cout << "energy_vdw = " << energy_vdw << " energy_elec = " << energy_elec << std::endl;

  // -------------------- END -----------------

  delete [] in14list;
  delete [] ex14list;

  delete [] loc2glo_ind;

  delete [] iblo14;
  delete [] inb14;

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
