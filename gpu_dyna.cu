#include <iostream>
#include <fstream>
#include <cuda.h>
#include "cuda_utils.h"
#include "CudaLeapfrogIntegrator.h"
#include "CudaDomdec.h"
#include "CudaDomdecBonded.h"
#include "CudaPMEForcefield.h"

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
// Loads constraints and masses from file
//
void load_constr_mass(const int nconstr, const int nmass, const char *filename, const int n,
		      double *constr, double *mass) {

  std::ifstream file(filename);
  if (file.is_open()) {

    for (int i=0;i < n;i++) {
      for (int k=0;k < nconstr;k++) {
	if (!(file >> constr[i*nconstr+k])) {
	  std::cerr<<"Error reading file "<<filename<<std::endl;
	  exit(1);
	}
      }
      for (int k=0;k < nmass;k++) {
	if (!(file >> mass[i*nmass+k])) {
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
// Writes (x, y, z) into a file
//
void write_xyz(const int n, const double *x, const double *y, const double *z, const char *filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    for (int i=0;i < n;i++) {
      file << x[i] << " " << y[i] << " " << z[i] << std::endl;
    }
  } else {
    std::cout << "write_xyz: Error opening file " << filename << std::endl;
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
  const int nfftx = 64;
  const int nffty = 64;
  const int nfftz = 64;
  const int forder = 4;
  const double rnl = 11.0;
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

  const int nbond = 23592;
  const int nbondcoef = 129;

  const int nureyb = 11584;
  const int nureybcoef = 327;

  const int nangle = 11584;
  const int nanglecoef = 327;

  const int ndihe = 6701;
  const int ndihecoef = 438;

  const int nimdihe = 418;
  const int nimdihecoef = 40;

  const int ncmap = 0;
  const int ncmapcoef = 0;

  bond_t *bond = new bond_t[nbond];
  load_vec<int>(3, "test_data/bond.txt", nbond, (int *)bond);
  float2 *bondcoef = new float2[nbondcoef];
  load_vec<float>(2, "test_data/bondcoef.txt", nbondcoef, (float *)bondcoef);

  bond_t *ureyb = new bond_t[nureyb];
  load_vec<int>(3, "test_data/ureyb.txt", nureyb, (int *)ureyb);
  float2 *ureybcoef = new float2[nureybcoef];
  load_vec<float>(2, "test_data/ureybcoef.txt", nureybcoef, (float *)ureybcoef);

  angle_t *angle = new angle_t[nangle];
  load_vec<int>(4, "test_data/angle.txt", nangle, (int *)angle);
  float2 *anglecoef = new float2[nanglecoef];
  load_vec<float>(2, "test_data/anglecoef.txt", nanglecoef, (float *)anglecoef);

  dihe_t *dihe = new dihe_t[ndihe];
  load_vec<int>(5, "test_data/dihe.txt", ndihe, (int *)dihe);
  float4 *dihecoef = new float4[ndihecoef];
  load_vec<float>(4, "test_data/dihecoef.txt", ndihecoef, (float *)dihecoef);

  dihe_t *imdihe = new dihe_t[nimdihe];
  load_vec<int>(5, "test_data/imdihe.txt", nimdihe, (int *)imdihe);
  float4 *imdihecoef = new float4[nimdihecoef];
  load_vec<float>(4, "test_data/imdihecoef.txt", nimdihecoef, (float *)imdihecoef);

  cmap_t *cmap = NULL;
  float2 *cmapcoef = NULL;

  //-------------------------------------------------------------------------------------

  const int nvdwparam = 1260;
  float* vdwparam = new float[nvdwparam];
  float* vdwparam14 = new float[nvdwparam];
  load_vec<float>(1, "test_data/vdwparam.txt", nvdwparam, vdwparam);
  load_vec<float>(1, "test_data/vdwparam14.txt", nvdwparam, vdwparam14);

  int *vdwtype = new int[ncoord];
  load_vec<int>(1, "test_data/glo_vdwtype.txt", ncoord, vdwtype);

  //-------------------------------------------------------------------------------------

  const int niblo14 = 23558;
  const int ninb14 = 34709;
  int *iblo14 = new int[niblo14];
  int *inb14 = new int[ninb14];
  load_vec<int>(1, "test_data/iblo14.txt", niblo14, iblo14);
  load_vec<int>(1, "test_data/inb14.txt", ninb14, inb14);

  //-------------------------------------------------------------------------------------
  
  const int nin14 = 6556;
  const int nex14 = 28153;
  xx14_t *in14 = new xx14_t[nin14];
  xx14_t *ex14 = new xx14_t[nex14];
  load_vec<int>(2, "test_data/in14.txt", nin14, (int *)in14);
  load_vec<int>(2, "test_data/ex14.txt", nex14, (int *)ex14);

  //-------------------------------------------------------------------------------------

  /*
  const double mO = 15.9994;
  const double mH = 1.008;
  const double rOHsq = 0.91623184;
  const double rHHsq = 2.29189321;
  const int nsolvent = 7023;
  const int npair = 458;
  const int ntrip = 233;
  const int nquad = 99;

  // Load constraint indices
  int *solvent_ind = (int *)malloc(nsolvent*3*sizeof(int));
  load_vec<int>(3, "test_data/solvent_ind.txt", nsolvent, solvent_ind);

  int *pair_ind = (int *)malloc(npair*2*sizeof(int));
  load_vec<int>(2, "test_data/pair_ind.txt", npair, pair_ind);

  int *trip_ind = (int *)malloc(ntrip*3*sizeof(int));
  load_vec<int>(3, "test_data/trip_ind.txt", ntrip, trip_ind);

  int *quad_ind = (int *)malloc(nquad*4*sizeof(int));
  load_vec<int>(4, "test_data/quad_ind.txt", nquad, quad_ind);

  // Load constraint distances and masses
  double *pair_constr = (double *)malloc(npair*sizeof(double));
  double *pair_mass = (double *)malloc(npair*2*sizeof(double));
  load_constr_mass(1, 2, "test_data/pair_constr_mass.txt", npair, pair_constr, pair_mass);

  double *trip_constr = (double *)malloc(ntrip*2*sizeof(double));
  double *trip_mass = (double *)malloc(ntrip*5*sizeof(double));
  load_constr_mass(2, 5, "test_data/trip_constr_mass.txt", ntrip, trip_constr, trip_mass);

  double *quad_constr = (double *)malloc(nquad*3*sizeof(double));
  double *quad_mass = (double *)malloc(nquad*7*sizeof(double));
  load_constr_mass(3, 7, "test_data/quad_constr_mass.txt", nquad, quad_constr, quad_mass);

  HoloConst holoconst;
  holoconst.setup_solvent_parameters(mO, mH, rOHsq, rHHsq);
  holoconst.setup_ind_mass_constr(npair, (int2 *)pair_ind, pair_constr, pair_mass,
				  ntrip, (int3 *)trip_ind, trip_constr, trip_mass,
				  nquad, (int4 *)quad_ind, quad_constr, quad_mass,
				  nsolvent, (int3 *)solvent_ind);
  */

  //-------------------------------------------------------------------------------------

  CudaLeapfrogIntegrator leapfrog(NULL /*&holoconst*/);

  // Neighborlist
  NeighborList<32> nlist(ncoord, iblo14, inb14);

  // Setup domain decomposition
  CudaDomdecBonded domdec_bonded(nbond, bond, nureyb, ureyb, nangle, angle,
				 ndihe, dihe, nimdihe, imdihe, ncmap, cmap,
				 nin14, in14, nex14, ex14);
  CudaDomdec domdec(ncoord, boxx, boxy, boxz, rnl, 1, 1, 1, 0);

  // Charges
  float *q = new float[ncoord];
  load_vec<float>(1, "test_data/q.txt", ncoord, q);

  // Setup PME force field
  CudaPMEForcefield forcefield(// Domain decomposition
			       &domdec, &domdec_bonded,
			       // Neighborlist
			       &nlist,
			       // Bonded
			       nbondcoef, bondcoef, nureybcoef, ureybcoef, nanglecoef, anglecoef,
			       ndihecoef, dihecoef, nimdihecoef, imdihecoef, ncmapcoef, cmapcoef,
			       // Direct non-bonded
			       roff, ron, kappa, e14fac, VDW_VSH, EWALD,
			       nvdwparam, vdwparam, vdwparam14, vdwtype, q,
			       // Recip non-bonded
			       nfftx, nffty, nfftz, forder);

  delete [] q;

  leapfrog.set_forcefield(&forcefield);

  // Masses
  double *mass = new double[ncoord];
  load_vec<double>(1, "test_data/mass.txt", ncoord, mass);

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

  double *fx = new double[ncoord];
  double *fy = new double[ncoord];
  double *fz = new double[ncoord];

  leapfrog.init(ncoord, x, y, z, dx, dy, dz, mass);
  leapfrog.set_coord_buffers(x, y, z);
  leapfrog.set_step_buffers(dx, dy, dz);
  leapfrog.set_force_buffers(fx, fy, fz);
  leapfrog.set_timestep(1.0);
  int nstep = 100;
  int print_freq = 1000;
  int restart_freq = 10000;
  leapfrog.run(nstep, print_freq, restart_freq);

  write_xyz(ncoord, x, y, z, "coord.txt");
  write_xyz(ncoord, dx, dy, dz, "step.txt");
  write_xyz(ncoord, fx, fy, fz, "force.txt");

  delete [] mass;

  delete [] x;
  delete [] y;
  delete [] z;

  delete [] dx;
  delete [] dy;
  delete [] dz;

  delete [] fx;
  delete [] fy;
  delete [] fz;

  //-------------------------------------------------------------------------------------

  if (bond != NULL) delete [] bond;
  delete [] bondcoef;
  
  if (ureyb != NULL) delete [] ureyb;
  delete [] ureybcoef;
  
  if (angle != NULL) delete [] angle;
  delete [] anglecoef;

  if (dihe != NULL) delete [] dihe;
  delete [] dihecoef;
  
  if (imdihe != NULL) delete [] imdihe;
  delete [] imdihecoef;

  //delete [] cmap;
  //delete [] cmapcoef;

  //-------------------------------------------------------------------------------------

  delete [] vdwparam;
  delete [] vdwparam14;
  delete [] vdwtype;

  //-------------------------------------------------------------------------------------

  delete [] iblo14;
  delete [] inb14;

  //-------------------------------------------------------------------------------------

  delete [] in14;
  delete [] ex14;

  //-------------------------------------------------------------------------------------

  /*
  delete [] solvent_ind;

  delete [] pair_ind;
  delete [] trip_ind;
  delete [] quad_ind;

  delete [] pair_constr;
  delete [] pair_mass;
  delete [] trip_constr;
  delete [] trip_mass;
  delete [] quad_constr;
  delete [] quad_mass;
  */

  return;
}
