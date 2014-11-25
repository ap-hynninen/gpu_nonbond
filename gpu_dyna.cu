#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cuda.h>
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "mpi_utils.h"
#include "CudaLeapfrogIntegrator.h"
#include "CudaDomdec.h"
#include "CudaDomdecGroups.h"
#include "CudaPMEForcefield.h"
#include "CudaDomdecRecipLooper.h"

int numnode=1, mynode=0;

bool test_cudaAware(const int mynode, const int numnode);
void test(const int nstep, const bool use_holoconst, const bool cudaAware, const bool use_pure_recip);

int main(int argc, char *argv[]) {

  int nstep = 1;
  bool use_holoconst = false;
  bool cudaAware = false;
  bool use_pure_recip = false;
  std::vector<int> devices;

  int iarg = 1;
  bool arg_ok = true;
  while (iarg < argc) {
    if (strcmp(argv[iarg],"-nstep")==0) {
      iarg++;
      if (iarg == argc) {
	arg_ok = false;
	break;
      }
      sscanf(argv[iarg],"%d",&nstep);
      iarg++;
    } else if (strcmp(argv[iarg],"-holoconst")==0) {
      iarg++;
      if (iarg == argc) {
	arg_ok = false;
	break;
      }
      if (strcmp(argv[iarg],"yes")==0) {
	use_holoconst = true;
      } else if (strcmp(argv[iarg],"no")==0) {
	use_holoconst = false;
      } else {
	arg_ok = false;
	break;
      }
      iarg++;
    } else if (strcmp(argv[iarg],"-cuda-aware")==0) {
      iarg++;
      if (iarg == argc) {
	arg_ok = false;
	break;
      }
      if (strcmp(argv[iarg],"yes")==0) {
	cudaAware = true;
      } else if (strcmp(argv[iarg],"no")==0) {
	cudaAware = false;
      } else {
	arg_ok = false;
	break;
      }
      iarg++;
    } else if (strcmp(argv[iarg],"-use-pure-recip")==0) {
      iarg++;
      if (iarg == argc) {
	arg_ok = false;
	break;
      }
      if (strcmp(argv[iarg],"yes")==0) {
	use_pure_recip = true;
      } else if (strcmp(argv[iarg],"no")==0) {
	use_pure_recip = false;
      } else {
	arg_ok = false;
	break;
      }
      iarg++;
    } else if (strcmp(argv[iarg],"-devices")==0) {
      iarg++;
      if (iarg == argc) {
	arg_ok = false;
	break;
      }
      while (iarg < argc && isdigit(argv[iarg][0])) {
	//fprintf(stderr,"%d\n",atoi(argv[iarg]));
	devices.push_back(atoi(argv[iarg]));
	iarg++;
      }
    } else {
      std::cout << "Invalid option " << argv[iarg] << std::endl;
      arg_ok = false;
      break;
    }
  }

  if (arg_ok) {
    // Get the local rank within this node from environmental variables
    int local_rank = get_env_local_rank();
    int local_size = get_env_local_size();

    if (local_rank < 0 && cudaAware) {
      std::cout << "Requesting CUDA aware MPI but local rank not defined by environment" << std::endl;
      std::cout << "=> defaulting back to non CUDA aware MPI" << std::endl;
      cudaAware = false;
    }

    std::cout << "local_rank=" << local_rank << " local_size=" << local_size << std::endl;
    if (local_rank >= 0 && local_size > 0) {
      start_gpu(local_size, local_rank, devices);
      start_mpi(argc, argv, numnode, mynode);
    } else {
      start_mpi(argc, argv, numnode, mynode);
      start_gpu(numnode, mynode, devices);
    }

    if (cudaAware) {
      if (!test_cudaAware(mynode, numnode)) {
	std::cout << "CUDA aware test FAILED => defaulting back to non CUDA aware MPI" << std::endl;
	cudaAware = false;
      } else {
	if (mynode == 0) {
	  std::cout << "CUDA aware tested OK" << std::endl;
	}
      }
    }

  }

  int ret_val = 0;

  if (arg_ok) {
    test(nstep, use_holoconst, cudaAware, use_pure_recip);
  } else {
    std::cout << "Usage: mpirun -n X gpu_dyna OPTIONS" << std::endl;
    std::cout << "OPTIONS:" << std::endl;
    std::cout << "-nstep N" << std::endl;
    std::cout << "-holoconst <yes|no> " << std::endl;
    std::cout << "-cuda-aware <yes|no> " << std::endl;
    std::cout << "-use-pure-recip <yes|no>" << std::endl;
    std::cout << "-devices 0 1 2 3" << std::endl;
    ret_val = 1;
  }

  if (arg_ok) {
    stop_mpi();
    stop_gpu();
  }

  return ret_val;
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
    file.precision(16);
    file.setf(std::ios::fixed, std::ios::floatfield);
    for (int i=0;i < n;i++) {
      file << x[i] << " " << y[i] << " " << z[i] << std::endl;
    }
  } else {
    std::cout << "write_xyz: Error opening file " << filename << std::endl;
    exit(1);
  }
}

//
// Reads (x, y, z) from a file
//
void read_xyz(const int n, double *x, double *y, double *z, const char *filename) {
  std::ifstream file(filename);
  if (file.is_open()) {
    for (int i=0;i < n;i++) {
      file >> x[i] >> y[i] >> z[i];
    }
  } else {
    std::cout << "write_xyz: Error opening file " << filename << std::endl;
    exit(1);
  }
}

//
// Checks holonomic constraints
//
void check_holoconst(const double* x, const double* y, const double* z,
		     const int npair, const bond_t* h_pair_indtype, const double* h_pair_constr, 
		     const int ntrip, const angle_t* h_trip_indtype, const double* h_trip_constr,
		     const int nquad, const dihe_t* h_quad_indtype, const double* h_quad_constr,
		     const int nsolvent, const solvent_t* h_solvent_ind,
		     const double rOHsq, const double rHHsq) {
  double tol = 1.0e-8;
  double max_err = 0.0;

  for (int i=0;i < npair;i++) {
    bond_t bond = h_pair_indtype[i];
    double dx = x[bond.i] - x[bond.j];
    double dy = y[bond.i] - y[bond.j];
    double dz = z[bond.i] - z[bond.j];
    double rsq = dx*dx + dy*dy + dz*dz;
    double err = fabs(rsq - h_pair_constr[bond.itype]);
    max_err = max(max_err, err);
    if (err > tol) {
      std::cout << "Error in PAIR: err = " << err << std::endl;
      return;
    }
  }

  for (int i=0;i < ntrip;i++) {
    angle_t angle = h_trip_indtype[i];
    double dx = x[angle.i] - x[angle.j];
    double dy = y[angle.i] - y[angle.j];
    double dz = z[angle.i] - z[angle.j];
    double rsq = dx*dx + dy*dy + dz*dz;
    double err = fabs(rsq - h_trip_constr[angle.itype*2]);
    max_err = max(max_err, err);
    if (err > tol) {
      std::cout << "Error in TRIP (i-j): err = " << err << std::endl;
      return;
    }
    dx = x[angle.i] - x[angle.k];
    dy = y[angle.i] - y[angle.k];
    dz = z[angle.i] - z[angle.k];
    rsq = dx*dx + dy*dy + dz*dz;
    err = fabs(rsq - h_trip_constr[angle.itype*2+1]);
    max_err = max(max_err, err);
    if (err > tol) {
      std::cout << "Error in TRIP (i-k): err = " << err << std::endl;
      return;
    }
  }

  for (int i=0;i < nquad;i++) {
    dihe_t dihe = h_quad_indtype[i];
    double dx = x[dihe.i] - x[dihe.j];
    double dy = y[dihe.i] - y[dihe.j];
    double dz = z[dihe.i] - z[dihe.j];
    double rsq = dx*dx + dy*dy + dz*dz;
    double err = fabs(rsq - h_quad_constr[dihe.itype*3]);
    max_err = max(max_err, err);
    if (err > tol) {
      std::cout << "Error in QUAD (i-j): err = " << err << std::endl;
      return;
    }
    dx = x[dihe.i] - x[dihe.k];
    dy = y[dihe.i] - y[dihe.k];
    dz = z[dihe.i] - z[dihe.k];
    rsq = dx*dx + dy*dy + dz*dz;
    err = fabs(rsq - h_quad_constr[dihe.itype*3+1]);
    max_err = max(max_err, err);
    if (err > tol) {
      std::cout << "Error in QUAD (i-k): err = " << err << std::endl;
      return;
    }
    dx = x[dihe.i] - x[dihe.l];
    dy = y[dihe.i] - y[dihe.l];
    dz = z[dihe.i] - z[dihe.l];
    rsq = dx*dx + dy*dy + dz*dz;
    err = fabs(rsq - h_quad_constr[dihe.itype*3+2]);
    max_err = max(max_err, err);
    if (err > tol) {
      std::cout << "Error in QUAD (i-l): err = " << err << std::endl;
      return;
    }
  }

  for (int i=0;i < nsolvent;i++) {
    solvent_t solvent = h_solvent_ind[i];
    double dx = x[solvent.i] - x[solvent.j];
    double dy = y[solvent.i] - y[solvent.j];
    double dz = z[solvent.i] - z[solvent.j];
    double rsq = dx*dx + dy*dy + dz*dz;
    double err = fabs(rsq - rOHsq);
    max_err = max(max_err, err);
    if (err > tol) {
      std::cout << "Error in SOLVENT (O-H1): err = " << err << std::endl;
      return;
    }
    dx = x[solvent.i] - x[solvent.k];
    dy = y[solvent.i] - y[solvent.k];
    dz = z[solvent.i] - z[solvent.k];
    rsq = dx*dx + dy*dy + dz*dz;
    err = fabs(rsq - rOHsq);
    max_err = max(max_err, err);
    if (err > tol) {
      std::cout << "Error in SOLVENT (O-H2): err = " << err << std::endl;
      return;
    }
    dx = x[solvent.j] - x[solvent.k];
    dy = y[solvent.j] - y[solvent.k];
    dz = z[solvent.j] - z[solvent.k];
    rsq = dx*dx + dy*dy + dz*dz;
    err = fabs(rsq - rHHsq);
    max_err = max(max_err, err);
    if (err > tol) {
      std::cout << "Error in SOLVENT (H1-H2): err = " << err << std::endl;
      return;
    }
  }

  std::cout << "check_holoconst OK (max_err = " << max_err << ")" << std::endl;
}

//
// Test the code using data in test_data/ -directory
//
void test(const int nstep, const bool use_holoconst, const bool cudaAware, const bool use_pure_recip) {

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

  // Very simple node setup
  int nx;
  int ny;
  int nz;
  bool isDirect;
  bool isRecip;
  std::vector<int> direct_nodes;
  std::vector<int> recip_nodes;
  if (use_pure_recip && numnode > 1) {
    // Separate Recip node, last node
    direct_nodes.resize(numnode-1);
    recip_nodes.resize(1);
    nx = 1;
    ny = 1;
    nz = numnode-1;
    for (int i=0;i < numnode-1;i++) direct_nodes.at(i) = i;
    recip_nodes.at(0) = numnode-1;
    isDirect = false;
    isRecip = false;
    if (mynode == recip_nodes.at(0)) {
      isRecip = true;
    } else {
      isDirect = true;
    }
  } else {
    // Mixed Direct+Recip node, first node
    direct_nodes.resize(numnode);
    recip_nodes.resize(1);
    nx = 1;
    ny = 1;
    nz = numnode;
    isDirect = true;
    isRecip = (mynode == 0) ? true : false;
    for (int i=0;i < numnode;i++) direct_nodes.at(i) = i;
    recip_nodes.at(0) = 0;
  }

  if (isDirect && isRecip) {
    std::cout << "Node " << mynode << " is Direct+Recip" << std::endl;
  } else if (isDirect) {
    std::cout << "Node " << mynode << " is Direct" << std::endl;
  } else if (isRecip) {
    std::cout << "Node " << mynode << " is Recip" << std::endl;
  }

  // MPI communicators
  MPI_Comm comm_direct;
  MPI_Comm comm_recip;
  MPI_Comm comm_direct_recip = MPI_COMM_WORLD;
  
  MPI_Group group_world;
  MPI_Group group_direct;
  MPI_Group group_recip;
  
  // Get handle to the entire domain
  MPICheck(MPI_Comm_group(MPI_COMM_WORLD, &group_world));
  
  //if (isDirect) {
  MPICheck(MPI_Group_incl(group_world, direct_nodes.size(), direct_nodes.data(), &group_direct));
  MPICheck(MPI_Comm_create(MPI_COMM_WORLD, group_direct, &comm_direct));
  //}
  
  //if (isRecip) {
  MPICheck(MPI_Group_incl(group_world, recip_nodes.size(), recip_nodes.data(), &group_recip));
  MPICheck(MPI_Comm_create(MPI_COMM_WORLD, group_recip, &comm_recip));
  //}

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

    bond_t *h_bond = new bond_t[nbond];
    load_vec<int>(3, "test_data/bond.txt", nbond, (int *)h_bond);
    float2 *h_bondcoef = new float2[nbondcoef];
    load_vec<float>(2, "test_data/bondcoef.txt", nbondcoef, (float *)h_bondcoef);

    bond_t *h_ureyb = new bond_t[nureyb];
    load_vec<int>(3, "test_data/ureyb.txt", nureyb, (int *)h_ureyb);
    float2 *h_ureybcoef = new float2[nureybcoef];
    load_vec<float>(2, "test_data/ureybcoef.txt", nureybcoef, (float *)h_ureybcoef);

    angle_t *h_angle = new angle_t[nangle];
    load_vec<int>(4, "test_data/angle.txt", nangle, (int *)h_angle);
    float2 *h_anglecoef = new float2[nanglecoef];
    load_vec<float>(2, "test_data/anglecoef.txt", nanglecoef, (float *)h_anglecoef);

    dihe_t *h_dihe = new dihe_t[ndihe];
    load_vec<int>(5, "test_data/dihe.txt", ndihe, (int *)h_dihe);
    float4 *h_dihecoef = new float4[ndihecoef];
    load_vec<float>(4, "test_data/dihecoef.txt", ndihecoef, (float *)h_dihecoef);

    dihe_t *h_imdihe = new dihe_t[nimdihe];
    load_vec<int>(5, "test_data/imdihe.txt", nimdihe, (int *)h_imdihe);
    float4 *h_imdihecoef = new float4[nimdihecoef];
    load_vec<float>(4, "test_data/imdihecoef.txt", nimdihecoef, (float *)h_imdihecoef);

    //-------------------------------------------------------------------------------------

    const int nvdwparam = 1260;
    float* h_vdwparam = new float[nvdwparam];
    float* h_vdwparam14 = new float[nvdwparam];
    load_vec<float>(1, "test_data/vdwparam.txt", nvdwparam, h_vdwparam);
    load_vec<float>(1, "test_data/vdwparam14.txt", nvdwparam, h_vdwparam14);

    int *h_vdwtype = new int[ncoord];
    load_vec<int>(1, "test_data/glo_vdwtype.txt", ncoord, h_vdwtype);

    //-------------------------------------------------------------------------------------

    const int niblo14 = 23558;
    const int ninb14 = 34709;
    int *h_iblo14 = new int[niblo14];
    int *h_inb14 = new int[ninb14];
    load_vec<int>(1, "test_data/iblo14.txt", niblo14, h_iblo14);
    load_vec<int>(1, "test_data/inb14.txt", ninb14, h_inb14);

    //-------------------------------------------------------------------------------------
  
    const int nin14 = 6556;
    const int nex14 = 28153;
    xx14_t *h_in14 = new xx14_t[nin14];
    xx14_t *h_ex14 = new xx14_t[nex14];
    load_vec<int>(2, "test_data/in14.txt", nin14, (int *)h_in14);
    load_vec<int>(2, "test_data/ex14.txt", nex14, (int *)h_ex14);

    //-------------------------------------------------------------------------------------

    const double mO = 15.9994;
    const double mH = 1.008;
    const double rOHsq = 0.91623184;
    const double rHHsq = 2.29189321;
    const int nsolvent = 7023;
    const int npair = 458;
    const int ntrip = 233;
    const int nquad = 99;
    const int npair_type = 9;
    const int ntrip_type = 3;
    const int nquad_type = 2;

    double *h_pair_constr = new double[npair_type];
    double *h_pair_mass = new double[npair_type*2];
    load_constr_mass(1, 2, "test_data/pair_types.txt", npair_type, h_pair_constr, h_pair_mass);
    bond_t* h_pair_indtype = new bond_t[npair];
    load_vec<int>(3, "test_data/pair_indtype.txt", npair, (int *)h_pair_indtype);

    double *h_trip_constr = new double[ntrip_type*2];
    double *h_trip_mass = new double[ntrip_type*5];
    load_constr_mass(2, 5, "test_data/trip_types.txt", ntrip_type, h_trip_constr, h_trip_mass);
    angle_t* h_trip_indtype = new angle_t[ntrip];
    load_vec<int>(4, "test_data/trip_indtype.txt", ntrip, (int *)h_trip_indtype);

    double *h_quad_constr = new double[nquad_type*3];
    double *h_quad_mass = new double[nquad_type*7];
    load_constr_mass(3, 7, "test_data/quad_types.txt", nquad_type, h_quad_constr, h_quad_mass);
    dihe_t* h_quad_indtype = new dihe_t[nquad];
    load_vec<int>(5, "test_data/quad_indtype.txt", nquad, (int *)h_quad_indtype);

    // Load constraint indices
    solvent_t *h_solvent_ind = new solvent_t[nsolvent];
    load_vec<int>(3, "test_data/solvent_ind.txt", nsolvent, (int *)h_solvent_ind);

    HoloConst* holoconst = NULL;
    if (use_holoconst) {
      holoconst = new HoloConst;;
      holoconst->setup_solvent_parameters(mO, mH, rOHsq, rHHsq);
      holoconst->setup_types(npair_type, h_pair_constr, h_pair_mass,
			     ntrip_type, h_trip_constr, h_trip_mass,
			     nquad_type, h_quad_constr, h_quad_mass);
    }
    //-------------------------------------------------------------------------------------

    // Topological exclusions
    CudaTopExcl topExcl(ncoord, h_iblo14, h_inb14);

    // Setup domain decomposition
    CudaMPI cudaMPI(cudaAware, comm_direct);
    CudaDomdec domdec(ncoord, boxx, boxy, boxz, rnl, nx, ny, nz, mynode, cudaMPI);

    // Setup "bonded" groups
    CudaDomdecGroups domdecGroups(domdec);

    CudaAtomGroup<bond_t> bondGroup(nbond, h_bond, "BOND");
    CudaAtomGroup<bond_t> ureybGroup(nureyb, h_ureyb, "UREYB");
    CudaAtomGroup<angle_t> angleGroup(nangle, h_angle, "ANGLE");
    CudaAtomGroup<dihe_t> diheGroup(ndihe, h_dihe, "DIHE");
    CudaAtomGroup<dihe_t> imdiheGroup(nimdihe, h_imdihe, "IMDIHE");
    CudaAtomGroup<xx14_t> in14Group(nin14, h_in14, "IN14");
    CudaAtomGroup<xx14_t> ex14Group(nex14, h_ex14, "EX14");
    CudaAtomGroup<bond_t>    pairGroup(npair, h_pair_indtype, "PAIR");
    CudaAtomGroup<angle_t>   tripGroup(ntrip, h_trip_indtype, "TRIP");
    CudaAtomGroup<dihe_t>    quadGroup(nquad, h_quad_indtype, "QUAD");
    CudaAtomGroup<solvent_t> solventGroup(nsolvent, h_solvent_ind, "SOLVENT");
    // Register groups
    // NOTE: the register IDs (BOND, UREYB, ...) must be unique
    domdecGroups.beginGroups();
    domdecGroups.insertGroup(BOND, bondGroup, h_bond);
    domdecGroups.insertGroup(UREYB, ureybGroup, h_ureyb);
    domdecGroups.insertGroup(ANGLE, angleGroup, h_angle);
    domdecGroups.insertGroup(DIHE, diheGroup, h_dihe);
    domdecGroups.insertGroup(IMDIHE, imdiheGroup, h_imdihe);
    domdecGroups.insertGroup(IN14, in14Group, h_in14);
    domdecGroups.insertGroup(EX14, ex14Group, h_ex14);
    if (use_holoconst) {
      domdecGroups.insertGroup(PAIR,    pairGroup, h_pair_indtype);
      domdecGroups.insertGroup(TRIP,    tripGroup, h_trip_indtype);
      domdecGroups.insertGroup(QUAD,    quadGroup, h_quad_indtype);
      domdecGroups.insertGroup(SOLVENT, solventGroup, h_solvent_ind);
    }
    domdecGroups.finishGroups();

    CudaLeapfrogIntegrator leapfrog(holoconst);

    // Charges
    float *h_q = new float[ncoord];
    load_vec<float>(1, "test_data/q.txt", ncoord, h_q);

    // Setup PME force field
    CudaPMEForcefield forcefield(// Domain decomposition
				 domdec, domdecGroups,
				 // Non-bonded topological exclusions
				 topExcl,
				 // Bonded
				 nbondcoef, h_bondcoef, nureybcoef, h_ureybcoef, nanglecoef, h_anglecoef,
				 ndihecoef, h_dihecoef, nimdihecoef, h_imdihecoef, 0, NULL,
				 // Direct non-bonded
				 roff, ron, kappa, e14fac, VDW_VSH, EWALD,
				 nvdwparam, h_vdwparam, h_vdwparam14, h_vdwtype, h_q,
				 // Recip non-bonded
				 recip, recipComm);

    delete [] h_q;

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
    double timestep = use_holoconst ? 2.0 : 1.0;
    leapfrog.set_timestep(timestep);
    MPICheck(MPI_Barrier( MPI_COMM_WORLD));
    double begin = MPI_Wtime();
    leapfrog.run(nstep, 0);
    double end = MPI_Wtime();
    double time_spent = end - begin;
    if (mynode == 0) {
      std::cout << "time_spent (sec) = " << time_spent << " ns/day = "
		<< (double)nstep*timestep*0.000001/(time_spent/(double)(60*60*24)) << std::endl;
    }

    cudaCheck(cudaDeviceSynchronize());

    if (mynode == 0) {
      if (nstep == 100 || nstep == 20 || nstep == 10 || nstep == 2 
	  || (nstep == 1 && !use_holoconst)) {
	double* fxref = new double[ncoord];
	double* fyref = new double[ncoord];
	double* fzref = new double[ncoord];
	char filename[256];
	if (nstep == 100 && use_holoconst) {
	  sprintf(filename,"test_data/force_dyn%d_holoconst.txt",nstep);
	} else {
	  sprintf(filename,"test_data/force_dyn%d.txt",nstep);
	}
	read_xyz(ncoord, fxref, fyref, fzref, filename);
	double max_err = 0.0;
	double err_tol = 5.0e-6;
	for (int i=0;i < ncoord;i++) {
	  double dfx = fx[i] - fxref[i];
	  double dfy = fy[i] - fyref[i];
	  double dfz = fz[i] - fzref[i];
	  double err = dfx*dfx + dfy*dfy + dfz*dfz;
	  max_err = max(max_err, err);
	  if (err > err_tol) {
	    std::cout << "i = " << i << " err = " << err << std::endl;
	    break;
	  }
	}
	if (max_err < err_tol) {
	  std::cout << "Test OK, maximum error = " << max_err << std::endl;
	} else {
	  std::cout << "Test FAILED" << std::endl;
	}
	delete [] fxref;
	delete [] fyref;
	delete [] fzref;
      } else {
	std::cout << "Test NOT performed (nstep != 100)" << std::endl;
      }

      write_xyz(ncoord, x, y, z, "coord.txt");
      write_xyz(ncoord, dx, dy, dz, "step.txt");
      write_xyz(ncoord, fx, fy, fz, "force.txt");

      if (nstep != 1 && use_holoconst) {
	check_holoconst(x, y, z,
			npair, h_pair_indtype, h_pair_constr, 
			ntrip, h_trip_indtype, h_trip_constr,
			nquad, h_quad_indtype, h_quad_constr,
			nsolvent, h_solvent_ind, rOHsq, rHHsq);
      }
    }

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

    if (h_bond != NULL) delete [] h_bond;
    delete [] h_bondcoef;
  
    if (h_ureyb != NULL) delete [] h_ureyb;
    delete [] h_ureybcoef;
  
    if (h_angle != NULL) delete [] h_angle;
    delete [] h_anglecoef;

    if (h_dihe != NULL) delete [] h_dihe;
    delete [] h_dihecoef;
  
    if (h_imdihe != NULL) delete [] h_imdihe;
    delete [] h_imdihecoef;

    //-------------------------------------------------------------------------------------

    delete [] h_vdwparam;
    delete [] h_vdwparam14;
    delete [] h_vdwtype;

    //-------------------------------------------------------------------------------------

    delete [] h_iblo14;
    delete [] h_inb14;

    //-------------------------------------------------------------------------------------

    delete [] h_in14;
    delete [] h_ex14;

    //-------------------------------------------------------------------------------------

    delete [] h_solvent_ind;

    delete [] h_pair_indtype;
    delete [] h_trip_indtype;
    delete [] h_quad_indtype;

    delete [] h_pair_constr;
    delete [] h_pair_mass;
    delete [] h_trip_constr;
    delete [] h_trip_mass;
    delete [] h_quad_constr;
    delete [] h_quad_mass;

    //-------------------------------------------------------------------------------------
    if (holoconst != NULL) delete holoconst;

  } else {
    // ------------------------------------------------------------
    // Pure recip node, loop here until Direct nodes say were done
    // ------------------------------------------------------------
    CudaDomdecRecipLooper looper(*recip, recipComm);
    looper.run();
  }

  if (recip != NULL) delete recip;

  if (isDirect) {
    MPICheck(MPI_Group_free(&group_direct));
  }

  if (isRecip) {
    MPICheck(MPI_Group_free(&group_recip));
  }

  return;
}

//
// Test if MPI is truly cuda aware by sending and receiving device buffer
//
bool test_cudaAware(const int mynode, const int numnode) {
  const int ncomm = 10;
  int *sendbuf, *recvbuf;
  allocate<int>(&sendbuf, ncomm);
  allocate<int>(&recvbuf, ncomm);
  int *h_sendbuf = new int[ncomm];
  int *h_recvbuf = new int[ncomm];
  for (int i=0;i < ncomm;i++) {
    h_sendbuf[i] = i;
    h_recvbuf[i] = 0;
  }
  copy_HtoD_sync<int>(h_sendbuf, sendbuf, ncomm);
  copy_HtoD_sync<int>(h_recvbuf, recvbuf, ncomm);

  const int TAG=1;
  MPICheck(MPI_Sendrecv(sendbuf, ncomm, MPI_INT, (mynode+1) % numnode, TAG,
			recvbuf, ncomm, MPI_INT, (mynode-1+numnode) % numnode, TAG,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE));

  copy_DtoH_sync<int>(recvbuf, h_recvbuf, ncomm);
  bool ok=true;
  for (int i=0;i < ncomm;i++) {
    if (h_recvbuf[i] != i) ok = false;
  }

  deallocate<int>(&sendbuf);
  deallocate<int>(&recvbuf);
  delete [] h_sendbuf;
  delete [] h_recvbuf;

  return ok;
}
