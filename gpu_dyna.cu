#include <iostream>
#include <fstream>
#include <cuda.h>
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "mpi_utils.h"
#include "CudaLeapfrogIntegrator.h"
#include "CudaDomdec.h"
#include "CudaDomdecBonded.h"
#include "CudaPMEForcefield.h"
#include "CudaDomdecRecipLooper.h"

int numnode=1, mynode=0;

void test();

int main(int argc, char *argv[]) {

  // Get the local rank within this node from environmental variables
  int local_rank = get_env_local_rank();

  std::cout << "local_rank = " << local_rank << std::endl;

  start_gpu(1, local_rank);
  start_mpi(argc, argv, numnode, mynode);

  test();

  stop_mpi();
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
  const bool cudaAware = false;

  // Very simple node setup
  int nx = 1;
  int ny = 1;
  int nz = numnode;
  bool isDirect = true;
  bool isRecip = (mynode == numnode-1) ? true : false;
  std::vector<int> direct_nodes(numnode);
  std::vector<int> recip_nodes(1);
  for (int i=0;i < numnode;i++) direct_nodes[i] = i;
  recip_nodes[0] = numnode-1;

  // MPI communicators
  MPI_Comm comm_direct;
  MPI_Comm comm_recip;
  MPI_Comm comm_direct_recip = MPI_COMM_WORLD;
  
  MPI_Group group_world;
  MPI_Group group_direct;
  MPI_Group group_recip;
  
  // Get handle to the entire domain
  MPICheck(MPI_Comm_group(MPI_COMM_WORLD, &group_world));
  
  if (isDirect) {
    MPICheck(MPI_Group_incl(group_world, numnode, direct_nodes.data(), &group_direct));
    MPICheck(MPI_Comm_create(MPI_COMM_WORLD, group_direct, &comm_direct));
  }
  
  if (isRecip) {
    MPICheck(MPI_Group_incl(group_world, 1, recip_nodes.data(), &group_recip));
    MPICheck(MPI_Comm_create(MPI_COMM_WORLD, group_recip, &comm_recip));
  }

  CudaDomdecRecip *recip = NULL;
  CudaDomdecRecipComm recipComm(comm_recip, comm_direct_recip,
				mynode, direct_nodes, recip_nodes, cudaAware);
  
  // Create reciprocal calculator
  if (isRecip) {
    recip = new CudaDomdecRecip(nfftx, nffty, nfftz, forder, kappa);
  }

  if (isDirect) {
    // --------------------------
    // Direct node
    // --------------------------

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
    load_vec<int>(3, "test_data/global_solvent_ind.txt", nsolvent, solvent_ind);

    int *pair_ind = (int *)malloc(npair*2*sizeof(int));
    load_vec<int>(2, "test_data/global_pair_ind.txt", npair, pair_ind);

    int *trip_ind = (int *)malloc(ntrip*3*sizeof(int));
    load_vec<int>(3, "test_data/global_trip_ind.txt", ntrip, trip_ind);

    int *quad_ind = (int *)malloc(nquad*4*sizeof(int));
    load_vec<int>(4, "test_data/global_quad_ind.txt", nquad, quad_ind);

    // Load constraint distances and masses
    double *pair_constr = (double *)malloc(npair*sizeof(double));
    double *pair_mass = (double *)malloc(npair*2*sizeof(double));
    load_constr_mass(1, 2, "test_data/global_pair_constr_mass.txt", npair, pair_constr, pair_mass);

    double *trip_constr = (double *)malloc(ntrip*2*sizeof(double));
    double *trip_mass = (double *)malloc(ntrip*5*sizeof(double));
    load_constr_mass(2, 5, "test_data/global_trip_constr_mass.txt", ntrip, trip_constr, trip_mass);

    double *quad_constr = (double *)malloc(nquad*3*sizeof(double));
    double *quad_mass = (double *)malloc(nquad*7*sizeof(double));
    load_constr_mass(3, 7, "test_data/global_quad_constr_mass.txt", nquad, quad_constr, quad_mass);

    HoloConst holoconst;
    holoconst.setup_solvent_parameters(mO, mH, rOHsq, rHHsq);

    //-------------------------------------------------------------------------------------

    cudaStream_t integrator_stream;
    cudaCheck(cudaStreamCreate(&integrator_stream));

    CudaLeapfrogIntegrator leapfrog(&holoconst,
				    npair, (int2 *)pair_ind, pair_constr, pair_mass,
				    ntrip, (int3 *)trip_ind, trip_constr, trip_mass,
				    nquad, (int4 *)quad_ind, quad_constr, quad_mass,
				    nsolvent, (int3 *)solvent_ind, 0);

    // Neighborlist
    NeighborList<32> nlist(ncoord, iblo14, inb14);

    // Setup domain decomposition
    CudaDomdecBonded domdec_bonded(nbond, bond, nureyb, ureyb, nangle, angle,
				   ndihe, dihe, nimdihe, imdihe, ncmap, cmap,
				   nin14, in14, nex14, ex14);

    CudaMPI cudaMPI(cudaAware, comm_direct);

    CudaDomdec domdec(ncoord, boxx, boxy, boxz, rnl, nx, ny, nz, mynode, cudaMPI);

    // Charges
    float *q = new float[ncoord];
    load_vec<float>(1, "test_data/q.txt", ncoord, q);

    // Setup PME force field
    CudaPMEForcefield forcefield(// Domain decomposition
				 domdec, &domdec_bonded,
				 // Neighborlist
				 &nlist,
				 // Bonded
				 nbondcoef, bondcoef, nureybcoef, ureybcoef, nanglecoef, anglecoef,
				 ndihecoef, dihecoef, nimdihecoef, imdihecoef, ncmapcoef, cmapcoef,
				 // Direct non-bonded
				 roff, ron, kappa, e14fac, VDW_VSH, EWALD,
				 nvdwparam, vdwparam, vdwparam14, vdwtype, q,
				 // Recip non-bonded
				 recip, recipComm);

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
    leapfrog.set_timestep(2.0);
    int nstep = 100;
    //int print_freq = 1000;
    //int restart_freq = 10000;
    leapfrog.run(nstep);

    write_xyz(ncoord, x, y, z, "coord.txt");
    write_xyz(ncoord, dx, dy, dz, "step.txt");
    write_xyz(ncoord, fx, fy, fz, "force.txt");

    cudaCheck(cudaStreamDestroy(integrator_stream));

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

  } else {
    // ------------------------------------------------------------
    // Pure recip node, loop here until Direct nodes say were done
    // ------------------------------------------------------------
    CudaDomdecRecipLooper looper(*recip, recipComm);
    looper.run();
  }

  if (recip != NULL) delete recip;

  MPICheck(MPI_Group_free(&group_direct));
  MPICheck(MPI_Group_free(&group_recip));

  return;
}
