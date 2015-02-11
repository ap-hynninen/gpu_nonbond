#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include "cuda_utils.h"
#include "XYZQ.h"
#include "Force.h"
#include "CudaNeighborList.h"
#include "CudaPMEDirectForce.h"
#include "CudaPMEDirectForceBlock.h"
#include "CudaEnergyVirial.h"

void test();

int numnode = 1;
int mynode = 0;

int main(int argc, char *argv[]) {

  std::vector<int> devices;
  start_gpu(numnode, mynode, devices);
  
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

  // Define force arrays
  Force<double> force_main("test_data/force_direct_main.txt");
  Force<double> force_total("test_data/force_direct.txt");
  Force<long long int> force_fp(ncoord);
  Force<double> force(ncoord);

  // Energy terms
  CudaEnergyVirial energyVirial;
  
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

  CudaTopExcl topExcl(ncoord, iblo14, inb14);

  // Create I vs. I interaction
  std::vector<int> numIntZone(8, 0);
  std::vector< std::vector<int> > intZones(8, std::vector<int>() );
  numIntZone.at(0) = 1;
  intZones.at(0).push_back(0);

  CudaNeighborList<32> nlist_ref(topExcl, 1, 1, 1);
  nlist_ref.registerList(numIntZone, intZones, "test_data/nlist.txt");
  //nlist.remove_empty_tiles();
  //nlist.split_dense_sparse(512);
  std::cout << "============== nlist_ref ==============" << std::endl;
  nlist_ref.analyze();
  std::cout << "=======================================" << std::endl;

  int zone_patom[9] = {0, 23558, 23558, 23558, 23558, 23558, 23558, 23558, 23558};

  int *h_loc2glo = new int[ncoord];
  for (int i=0;i < ncoord;i++) h_loc2glo[i] = i;
  int *loc2glo = NULL;
  allocate<int>(&loc2glo, ncoord);
  copy_HtoD_sync<int>(h_loc2glo, loc2glo, ncoord);
  delete [] h_loc2glo;

  CudaNeighborList<32> nlist(topExcl, 1, 1, 1);
  nlist.registerList(numIntZone, intZones);
  nlist.set_test(true);
  //nlist.sort(zone_patom, max_xyz, min_xyz, xyzq_unsorted.xyzq, xyzq_sorted.xyzq);
  nlist.sort(0, zone_patom, xyzq_unsorted.xyzq, xyzq_sorted.xyzq, loc2glo);
  nlist.build(0, zone_patom, boxx, boxy, boxz, rcut, xyzq_sorted.xyzq, loc2glo);
  cudaCheck(cudaDeviceSynchronize());

  std::cout << "================ nlist ================" << std::endl;
  nlist.analyze();
  std::cout << "=======================================" << std::endl;

  // ------------------- Non-bonded -----------------

  CudaPMEDirectForce<long long int, float> dir(energyVirial, "vdw", "elec", "excl");
  dir.setup(boxx, boxy, boxz, kappa, roff, ron, e14fac, VDW_VSH, EWALD);
  dir.set_vdwparam(1260, "test_data/vdwparam.txt");
  dir.set_vdwtype(ncoord, "test_data/vdwtype.txt");
  dir.calc_force(xyzq.xyzq, nlist_ref.getBuilder(0), false, false, force_fp.stride(), force_fp.xyz());
  force_fp.convert(force);
  cudaCheck(cudaDeviceSynchronize());
  tol = 7.72e-4;
  if (!force_main.compare(force, tol, max_diff)) {
    std::cout<<"Non-bonded (main) force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"Non-bonded (main) force comparison OK"<<std::endl;
    std::cout<<"(tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

  dir.set_vdwparam14(1260, "test_data/vdwparam14.txt");
  dir.set_14_list(nin14list, nex14list, in14list, ex14list);
  dir.calc_14_force(xyzq.xyzq, false, false, force_fp.stride(), force_fp.xyz());
  force_fp.convert(force);
  cudaCheck(cudaDeviceSynchronize());
  tol = 7.73e-4;
  if (!force_total.compare(force, tol, max_diff)) {
    std::cout<<"Non-bonded (total) force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"Non-bonded (total) force comparison OK"<<std::endl;
    std::cout<<"(tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

  // Check energy and virial
  force_fp.clear();
  energyVirial.clear();
  dir.calc_force(xyzq.xyzq, nlist_ref.getBuilder(0), true, true, force_fp.stride(), force_fp.xyz());
  //dir.calc_14_force(xyzq.xyzq, true, true, force_fp.stride(), force_fp.xyz());
  force_fp.convert(force);
  //dir.calc_virial(ncoord, xyzq.xyzq, force.stride(), force.xyz());
  energyVirial.calcVirial(ncoord, xyzq.xyzq, boxx, boxy, boxz, force.stride(), force.xyz());
  
  double energy_vdw;
  double energy_elec;
  //double energy_excl;
  double virtensor[9];
  //dir.get_energy_virial(true, true, &energy_vdw, &energy_elec, &energy_excl, virtensor);
  energyVirial.copyToHost();
  cudaCheck(cudaDeviceSynchronize());
  energy_vdw = energyVirial.getEnergy("vdw");
  energy_elec = energyVirial.getEnergy("elec");
  //energy_excl = energyVirial.getEnergy("excl");
  energyVirial.getVirial(virtensor);
  tol = 7.73e-4;
  if (!force_main.compare(force, tol, max_diff)) {
    std::cout<<"Non-bonded (main) force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"Non-bonded (main) force comparison OK"<<std::endl;
    std::cout<<"(tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

  std::cout << "energy_vdw = " << energy_vdw << " energy_elec = " << energy_elec << std::endl;
  max_diff = max(fabs(energy_vdw-ref_energy_vdw), fabs(energy_elec-ref_energy_elec));
  if (max_diff < 0.007) {
    std::cout << "Nonbonded energy comparison OK" << std::endl;
  } else {
    std::cout << "Nonbonded energy comparison FAILED max_diff=" << max_diff << std::endl;
  }

  double vir = (virtensor[0] + virtensor[4] + virtensor[8])/3.0;
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
    std::cout << "(1) Nonbonded virial comparison OK" << std::endl;
  } else {
    std::cout << "(1) Nonbonded virial comparison FAILED" << std::endl;
  }
  std::cout << "max_diff(vir_tensor) = " << max_diff << std::endl;
  std::cout << "max_diff(vir) = " << fabs(vir - ref_vir) << std::endl;

  //--------------- Non-bonded using GPU built neighborlist -----------

  // Update vdwtype to reflect new ordering of atoms
  int *h_glo_vdwtype = new int[ncoord];
  load_ind<int>(1, "test_data/glo_vdwtype.txt", ncoord, h_glo_vdwtype);
  int *glo_vdwtype;
  allocate<int>(&glo_vdwtype, ncoord);
  copy_HtoD_sync<int>(h_glo_vdwtype, glo_vdwtype, ncoord);
  dir.set_vdwtype(ncoord, glo_vdwtype, loc2glo);
  delete [] h_glo_vdwtype;
  
  force_fp.clear();
  //dir.clear_energy_virial();
  energyVirial.clear();
  dir.calc_force(xyzq_sorted.xyzq, nlist.getBuilder(0), true, true, force_fp.stride(), force_fp.xyz());
  force_fp.convert(force);
  //dir.calc_virial(ncoord, xyzq_sorted.xyzq, force.stride(), force.xyz());
  energyVirial.calcVirial(ncoord, xyzq_sorted.xyzq, boxx, boxy, boxz, force.stride(), force.xyz());

  energyVirial.copyToHost();
  cudaCheck(cudaDeviceSynchronize());
  energy_vdw = energyVirial.getEnergy("vdw");
  energy_elec = energyVirial.getEnergy("elec");
  //energy_excl = energyVirial.getEnergy("excl");
  energyVirial.getVirial(virtensor);
  //dir.get_energy_virial(true, true, &energy_vdw, &energy_elec, &energy_excl, virtensor);

  std::cout << "energy_vdw = " << energy_vdw << " energy_elec = " << energy_elec << std::endl;
  max_diff = max(fabs(energy_vdw-ref_energy_vdw), fabs(energy_elec-ref_energy_elec));
  if (max_diff < 0.005) {
    std::cout << "Nonbonded energy comparison OK" << std::endl;
  } else {
    std::cout << "Nonbonded energy comparison FAILED max_diff=" << max_diff << std::endl;
  }

  // NOTE: we're not doing the force comparison here because our reference forces are in
  //       different order
  /*
  tol = 7.71e-4;
  if (!force_main.compare(force, tol, max_diff)) {
    std::cout<<"Non-bonded (main) force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"Non-bonded (main) force comparison OK"<<std::endl;
    std::cout<<"(tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }
  */
  
  vir = (virtensor[0] + virtensor[4] + virtensor[8])/3.0;
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
    std::cout << "(2) Nonbonded virial comparison OK" << std::endl;
  } else {
    std::cout << "(2) Nonbonded virial comparison FAILED" << std::endl;
  }
  std::cout << "max_diff(vir_tensor) = " << max_diff << std::endl;
  std::cout << "max_diff(vir) = " << fabs(vir - ref_vir) << std::endl;

  // Before creating another CudaPMEDirectForce object, clear the textures
  dir.clearTextures();
  
  //--------------- Non-bonded with block  -----------
  CudaBlock cudaBlock(2);
  CudaPMEDirectForceBlock<long long int, float> dirblock(energyVirial, "vdw", "elec", "excl", cudaBlock);
  //CudaPMEDirectForce<long long int, float> dirblock;
  
  // Setup blockType
  int* h_blockType = new int[ncoord];
  for (int i=0;i < ncoord/2;i++) h_blockType[i] = 0;
  for (int i=ncoord/2;i < ncoord;i++) h_blockType[i] = 1;
  cudaBlock.setBlockType(ncoord, h_blockType);
  delete [] h_blockType;
  // Setup bixlam
  float h_bixlam[2] = {1.0f, 0.8f};
  cudaBlock.setBixlam(h_bixlam);
  float h_blockParam[4];
  for (int j=0;j < 2;j++) {
    for (int i=0;i < 2;i++) {
      h_blockParam[i+j*2] = h_bixlam[i]*h_bixlam[j];
    }
  }
  cudaBlock.setBlockParam(h_blockParam);
  dirblock.setup(boxx, boxy, boxz, kappa, roff, ron, e14fac, VDW_VSH, EWALD);
  dirblock.set_vdwparam(1260, "test_data/vdwparam.txt");
  dirblock.set_vdwtype(ncoord, glo_vdwtype, loc2glo);
    // Calculate forces, energies, and virial
  force_fp.clear();
  //dirblock.clear_energy_virial();
  energyVirial.clear();
  dirblock.calc_force(xyzq_sorted.xyzq, nlist.getBuilder(0), true, true, force_fp.stride(), force_fp.xyz());
  force_fp.convert(force);
  //dirblock.calc_virial(ncoord, xyzq_sorted.xyzq, force.stride(), force.xyz());
  energyVirial.calcVirial(ncoord, xyzq.xyzq, boxx, boxy, boxz, force.stride(), force.xyz());

  //dirblock.get_energy_virial(true, true, &energy_vdw, &energy_elec, &energy_excl, virtensor);
  //cudaCheck(cudaDeviceSynchronize());
  energyVirial.copyToHost();
  cudaCheck(cudaDeviceSynchronize());
  energy_vdw = energyVirial.getEnergy("vdw");
  energy_elec = energyVirial.getEnergy("elec");
  //energy_excl = energyVirial.getEnergy("excl");
  energyVirial.getVirial(virtensor);

  std::cout << "energy_vdw = " << energy_vdw << " energy_elec = " << energy_elec << std::endl;
  
  // -------------------- END -----------------

  delete [] in14list;
  delete [] ex14list;

  delete [] iblo14;
  delete [] inb14;

  deallocate<int>(&loc2glo);
  deallocate<int>(&glo_vdwtype);

  return;

}
