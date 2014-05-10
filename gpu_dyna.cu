#include <iostream>
#include <fstream>
#include <cuda.h>
#include "cuda_utils.h"
#include "CudaLeapfrogIntegrator.h"

void test();

int main(int argc, char *argv[]) {

  start_gpu(1, 0);
  
  test();

  stop_gpu();

  return 0;
}

//
// Loads vector from file
//
template <typename T>
void load_vec(const int nind, const char *filename, const int n, T *ind) {
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
  const int forder = 4;
  const double rcut = 11.0;
  const double roff = 9.0;
  const double ron = 7.5;
  const double e14fac = 1.0;
  const int ncoord = 23558;

  /*
  const int nin14list = 6556;
  const int nex14list = 28153;
  list14_t *in14list = new list14_t[nin14list];
  list14_t *ex14list = new list14_t[nex14list];
  load_vec<int>(3, "test_data/in14list.txt", nin14list, (int *)in14list);
  load_vec<int>(3, "test_data/ex14list.txt", nex14list, (int *)ex14list);
  */

  /*
  // Load coordinates
  XYZQ xyzq_unsorted("test_data/xyzq_unsorted.txt", 32);
  XYZQ xyzq_sorted(ncoord, 32);

  // ------------------- Neighborlist -----------------

  const int niblo14 = 23558;
  const int ninb14 = 34709;
  int *iblo14 = new int[niblo14];
  int *inb14 = new int[ninb14];
  load_vec<int>(1, "test_data/iblo14.txt", niblo14, iblo14);
  load_vec<int>(1, "test_data/inb14.txt", ninb14, inb14);

  int zone_patom[8] = {23558, 23558, 23558, 23558, 23558, 23558, 23558, 23558};
  float3 min_xyz[8], max_xyz[8];
  min_xyz[0].x = -31.74800;
  min_xyz[0].y = -31.77600;
  min_xyz[0].z = -31.77900;
  max_xyz[0].x = 31.73900;
  max_xyz[0].y = 31.80500;
  max_xyz[0].z = 31.80300;

  NeighborList<32> nlist(1, 1, 1);
  nlist.setup_top_excl(ncoord, iblo14, inb14);
  nlist.sort(zone_patom, max_xyz, min_xyz, xyzq_unsorted.xyzq, xyzq_sorted.xyzq);
  nlist.build(boxx, boxy, boxz, rcut, xyzq_sorted.xyzq);

  // ------------------- Non-bonded -----------------

  DirectForce<long long int, float> dir;
  dir.setup(boxx, boxy, boxz, kappa, roff, ron, e14fac, VDW_VSH, EWALD, true, true);
  dir.set_vdwparam("test_data/vdwparam.txt");
  dir.set_vdwtype("test_data/vdwtype.txt");
  dir.set_vdwparam14("test_data/vdwparam14.txt");
  dir.set_14_list(nin14list, nex14list, in14list, ex14list);

  dir.calc_force(xyzq_sorted.xyzq, &nlist_ref, false, false, force_fp.xyz.stride, force_fp.xyz.data);

  dir.calc_14_force(xyzq_sorted.xyzq, false, false, force_fp.xyz.stride, force_fp.xyz.data);

  delete [] in14list;
  delete [] ex14list;

  delete [] iblo14;
  delete [] inb14;

  // -------------------------------------------------
  */

  const int nbondlist = 23592;
  const int nbondcoef = 129;

  const int nureyblist = 11584;
  const int nureybcoef = 327;

  const int nanglelist = 11584;
  const int nanglecoef = 327;

  const int ndihelist = 6701;
  const int ndihecoef = 438;

  const int nimdihelist = 418;
  const int nimdihecoef = 40;

  const int ncmaplist = 0;
  const int ncmapcoef = 0;

  bondlist_t *bondlist = new bondlist_t[nbondlist];
  float2 *bondcoef = new float2[nbondcoef];
  load_vec<int>(4, "test_data/glo_bondlist.txt", nbondlist, (int *)bondlist);
  load_vec<float>(2, "test_data/bondcoef.txt", nbondcoef, (float *)bondcoef);

  bondlist_t *ureyblist = new bondlist_t[nureyblist];
  float2 *ureybcoef = new float2[nureybcoef];
  load_vec<int>(4, "test_data/glo_ureyblist.txt", nureyblist, (int *)ureyblist);
  load_vec<float>(2, "test_data/ureybcoef.txt", nureybcoef, (float *)ureybcoef);

  anglelist_t *anglelist = new anglelist_t[nanglelist];
  float2 *anglecoef = new float2[nanglecoef];
  load_vec<int>(6, "test_data/glo_anglelist.txt", nanglelist, (int *)anglelist);
  load_vec<float>(2, "test_data/anglecoef.txt", nanglecoef, (float *)anglecoef);

  dihelist_t *dihelist = new dihelist_t[ndihelist];
  float4 *dihecoef = new float4[ndihecoef];
  load_vec<int>(8, "test_data/glo_dihelist.txt", ndihelist, (int *)dihelist);
  load_vec<float>(4, "test_data/dihecoef.txt", ndihecoef, (float *)dihecoef);

  dihelist_t *imdihelist = new dihelist_t[nimdihelist];
  float4 *imdihecoef = new float4[nimdihecoef];
  load_vec<int>(8, "test_data/glo_imdihelist.txt", nimdihelist, (int *)imdihelist);
  load_vec<float>(4, "test_data/imdihecoef.txt", nimdihecoef, (float *)imdihecoef);

  cmaplist_t *cmaplist = NULL;
  float2 *cmapcoef = NULL;

  //-------------------------------------------------------------------------------------

  CudaLeapfrogIntegrator leapfrog;

  // Setup PME force field
  CudaPMEForcefield forcefield(// Bonded
			       nbondlist, bondlist, nureyblist, ureyblist, nanglelist, anglelist,
			       ndihelist, dihelist, nimdihelist, imdihelist, ncmaplist, cmaplist,
			       nbondcoef, bondcoef, nureybcoef, ureybcoef, nanglecoef, anglecoef,
			       ndihecoef, dihecoef, nimdihecoef, imdihecoef, ncmapcoef, cmapcoef,
			       // Direct non-bonded
			       rnl, roff, ron, kappa, e14fac, VDW_VSH, EWALD,
			       nvdwparam, vdwparam, vdwtype,
			       // Recip non-bonded
			       nfftx, nffty, nfftz, order);
  
  //forcefield.setup_direct_nonbonded();
  //forcefield.setup_recip_nonbonded();

  // Coordinates
  double *x = new double[ncoord];
  double *y = new double[ncoord];
  double *z = new double[ncoord];
  load_vec<double>(1, "test_data/x.txt", ncoord, x);
  load_vec<double>(1, "test_data/y.txt", ncoord, y);
  load_vec<double>(1, "test_data/z.txt", ncoord, z);

  // Step vector
  double *dx = new double[ncoord];
  double *dy = new double[ncoord];
  double *dz = new double[ncoord];
  load_vec<double>(1, "test_data/dx.txt", ncoord, dx);
  load_vec<double>(1, "test_data/dy.txt", ncoord, dy);
  load_vec<double>(1, "test_data/dz.txt", ncoord, dz);

  leapfrog.init(ncoord, x, y, z, dx, dy, dz);
  leapfrog.run(10);

  delete [] x;
  delete [] y;
  delete [] z;

  delete [] dx;
  delete [] dy;
  delete [] dz;

  //-------------------------------------------------------------------------------------

  delete [] bondlist;
  delete [] bondcoef;
  
  delete [] ureyblist;
  delete [] ureybcoef;
  
  delete [] anglelist;
  delete [] anglecoef;

  delete [] dihelist;
  delete [] dihecoef;
  
  delete [] imdihelist;
  delete [] imdihecoef;

  delete [] cmaplist;
  delete [] cmapcoef;

  return;
}
