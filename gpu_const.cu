#include <iostream>
#include <fstream>
#include <cuda.h>
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "HoloConst.h"
#include "hostXYZ.h"
#include "Bonded_struct.h"

void test();

//
// Main
//
int main(int argc, char *argv[]) {

  int numnode = 1;
  int mynode = 0;

  start_gpu(numnode, mynode);
  
  test();

  return 0;
}

//
// Loads (x, y, z) coordinates from file
//
void load_coord(const char *filename, const int n, double *x, double *y, double *z) {

  std::ifstream file(filename);
  if (file.is_open()) {

    int i = 0;
    while (file >> x[i] >> y[i] >> z[i]) i++;

    if (i > n) {
      std::cerr<<"Too many lines in file "<<filename<<std::endl;
      exit(1);
    }

  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }

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
// Checks SETTLE and SHAKE results
//
bool check_result(const int nind, const int n, const int *ind,
		  const double *x, const double *y, const double *z,
		  const double *x_ref, const double *y_ref, const double *z_ref,
		  const double tol, double &max_diff) {

  double x1, y1, z1;
  double x2, y2, z2;
  double diff;
  int imol, j, i;

  for (imol=0;imol < n;imol++) {
    for (j=0;j < nind;j++) {
      i = ind[imol*nind+j];
      x1 = x[i];
      y1 = y[i];
      z1 = z[i];
      x2 = x_ref[i];
      y2 = y_ref[i];
      z2 = z_ref[i];
      bool ok = true;
      if (isnan(x1) || isnan(y1) || isnan(z1) || isnan(x2) || isnan(y2) || isnan(z2)) ok = false;
      if (ok) {
	diff = max(fabs(x1-x2), max(fabs(y1-y2), fabs(z1-z2)));
	max_diff = max(diff, max_diff);
	if (diff > tol) ok = false;
      }
      if (!ok) {
	std::cout << "comparison FAILED, imol=" << imol << " diff=" << diff << std::endl;
	std::cout << "ind =";
	for (j=0;j < nind;j++) {
	  std::cout << " " << ind[imol*nind+j];
	}
	std::cout << std::endl;
	std::cout << x1 << " "<< y1 << " "<< z1 << std::endl;
	std::cout << x2 << " "<< y2 << " "<< z2 << std::endl;
	return false;
      }
    }
  }

  return true;
}

//
// Check results
//
void check_results(cudaXYZ<double>& xyz1, hostXYZ<double>& h_xyz_ref,
		   const int nsolvent, const int* h_solvent_ind,
		   const int npair, const int* h_pair_ind,
		   const int ntrip, const int* h_trip_ind,
		   const int nquad, const int* h_quad_ind) {

  hostXYZ<double> h_xyz1(xyz1);

  double max_diff;
  double tol = 5.0e-14;

  max_diff = 0.0;
  if (check_result(3, nsolvent, h_solvent_ind, h_xyz1.x(), h_xyz1.y(), h_xyz1.z(),
		   h_xyz_ref.x(), h_xyz_ref.y(), h_xyz_ref.z(), tol, max_diff)) {
    std::cout<<"solvent SETTLE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

  max_diff = 0.0;
  if (check_result(2, npair, h_pair_ind, h_xyz1.x(), h_xyz1.y(), h_xyz1.z(),
		   h_xyz_ref.x(), h_xyz_ref.y(), h_xyz_ref.z(), tol, max_diff)) {
    std::cout<<"pair SHAKE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

  max_diff = 0.0;
  if (check_result(3, ntrip, h_trip_ind, h_xyz1.x(), h_xyz1.y(), h_xyz1.z(),
		   h_xyz_ref.x(), h_xyz_ref.y(), h_xyz_ref.z(), tol, max_diff)) {
    std::cout<<"trip SHAKE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

  max_diff = 0.0;
  if (check_result(4, nquad, h_quad_ind, h_xyz1.x(), h_xyz1.y(), h_xyz1.z(),
		   h_xyz_ref.x(), h_xyz_ref.y(), h_xyz_ref.z(), tol, max_diff)) {
    std::cout<<"quad SHAKE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

}

//
// Test parametric version
//
void test_parametric(const double mO, const double mH, const double rOHsq, const double rHHsq,
		     const int npair, const int* h_pair_ind,
		     const int ntrip, const int* h_trip_ind,
		     const int nquad, const int* h_quad_ind,
		     const int nsolvent, const int* h_solvent_ind,
		     cudaXYZ<double>& xyz0, cudaXYZ<double>& xyz1, hostXYZ<double>& h_xyz1) {
  //---------------------------------------------------------------------------
  // Load constraint distances and masses
  double *h_pair_constr = (double *)malloc(npair*sizeof(double));
  double *h_pair_mass = (double *)malloc(npair*2*sizeof(double));
  load_constr_mass(1, 2, "test_data/pair_constr_mass.txt", npair, h_pair_constr, h_pair_mass);

  double *h_trip_constr = (double *)malloc(ntrip*2*sizeof(double));
  double *h_trip_mass = (double *)malloc(ntrip*5*sizeof(double));
  load_constr_mass(2, 5, "test_data/trip_constr_mass.txt", ntrip, h_trip_constr, h_trip_mass);

  double *h_quad_constr = (double *)malloc(nquad*3*sizeof(double));
  double *h_quad_mass = (double *)malloc(nquad*7*sizeof(double));
  load_constr_mass(3, 7, "test_data/quad_constr_mass.txt", nquad, h_quad_constr, h_quad_mass);
  //---------------------------------------------------------------------------

  HoloConst holoconst;

  // Setup
  holoconst.setup_solvent_parameters(mO, mH, rOHsq, rHHsq);
  holoconst.setup_ind_mass_constr(npair, (int2 *)h_pair_ind, h_pair_constr, h_pair_mass,
				  ntrip, (int3 *)h_trip_ind, h_trip_constr, h_trip_mass,
				  nquad, (int4 *)h_quad_ind, h_quad_constr, h_quad_mass,
				  nsolvent, (int3 *)h_solvent_ind);

  // Apply holonomic constraints
  xyz1.set_data_sync(h_xyz1);
  holoconst.apply(xyz0, xyz1);
  xyz1.set_data_sync(h_xyz1);
  holoconst.apply(xyz0, xyz1);
  xyz1.set_data_sync(h_xyz1);
  holoconst.apply(xyz0, xyz1);
  cudaCheck(cudaDeviceSynchronize());

  free(h_pair_constr);
  free(h_pair_mass);
  free(h_trip_constr);
  free(h_trip_mass);
  free(h_quad_constr);
  free(h_quad_mass);
}

//
// Test indexed version
//
void test_indexed(const double mO, const double mH, const double rOHsq, const double rHHsq,
		  const int npair, const int ntrip, const int nquad,
		  const int nsolvent, const int* h_solvent_ind,
		  cudaXYZ<double>& xyz0, cudaXYZ<double>& xyz1, hostXYZ<double>& h_xyz1) {

  const int npair_type = 9;
  const int ntrip_type = 3;
  const int nquad_type = 2;

  double *h_pair_constr = new double[npair];
  double *h_pair_mass = new double[npair*2];
  load_constr_mass(1, 2, "test_data/pair_types.txt", npair_type, h_pair_constr, h_pair_mass);
  bond_t* h_pair_indtype = new bond_t[npair];
  load_vec<int>(3, "test_data/pair_indtype.txt", npair, (int *)h_pair_indtype);

  double *h_trip_constr = new double[ntrip*2];
  double *h_trip_mass = new double[ntrip*5];
  load_constr_mass(2, 5, "test_data/trip_types.txt", ntrip_type, h_trip_constr, h_trip_mass);
  angle_t* h_trip_indtype = new angle_t[ntrip];
  load_vec<int>(4, "test_data/trip_indtype.txt", ntrip, (int *)h_trip_indtype);

  double *h_quad_constr = new double[nquad*3];
  double *h_quad_mass = new double[nquad*7];
  load_constr_mass(3, 7, "test_data/quad_types.txt", nquad_type, h_quad_constr, h_quad_mass);
  dihe_t* h_quad_indtype = new dihe_t[nquad];
  load_vec<int>(5, "test_data/quad_indtype.txt", nquad, (int *)h_quad_indtype);

  // Setup
  HoloConst holoconst;
  holoconst.setup_solvent_parameters(mO, mH, rOHsq, rHHsq);
  holoconst.setup_indexed(npair, h_pair_indtype, npair_type, h_pair_constr, h_pair_mass,
			  ntrip, h_trip_indtype, ntrip_type, h_trip_constr, h_trip_mass,
			  nquad, h_quad_indtype, nquad_type, h_quad_constr, h_quad_mass,
			  nsolvent, (int3 *)h_solvent_ind);

  // Apply holonomic constraints
  xyz1.set_data_sync(h_xyz1);
  holoconst.apply(xyz0, xyz1);
  xyz1.set_data_sync(h_xyz1);
  holoconst.apply(xyz0, xyz1);
  xyz1.set_data_sync(h_xyz1);
  holoconst.apply(xyz0, xyz1);
  cudaCheck(cudaDeviceSynchronize());

  delete [] h_pair_indtype;
  delete [] h_trip_indtype;
  delete [] h_quad_indtype;

  delete [] h_pair_constr;
  delete [] h_pair_mass;
  delete [] h_trip_constr;
  delete [] h_trip_mass;
  delete [] h_quad_constr;
  delete [] h_quad_mass;
}

//
// Test the code using data in test_data/ -directory
//
void test() {

  // Settings for the data:
  const double mO = 15.9994;
  const double mH = 1.008;
  const double rOHsq = 0.91623184;
  const double rHHsq = 2.29189321;
  const int ncoord = 23558;
  const int nsolvent = 7023;
  const int npair = 458;
  const int ntrip = 233;
  const int nquad = 99;

  cudaXYZ<double> xyz0(ncoord);
  cudaXYZ<double> xyz1(ncoord);

  // Load coordinates
  hostXYZ<double> h_xyz0(ncoord, NON_PINNED);
  hostXYZ<double> h_xyz1(ncoord, NON_PINNED);
  hostXYZ<double> h_xyz_ref(ncoord, NON_PINNED);
  load_coord("test_data/xyz0.txt", h_xyz0.size(), h_xyz0.x(), h_xyz0.y(), h_xyz0.z());
  load_coord("test_data/xyz1.txt", h_xyz1.size(), h_xyz1.x(), h_xyz1.y(), h_xyz1.z());
  load_coord("test_data/xyz_ref.txt", h_xyz_ref.size(), h_xyz_ref.x(), h_xyz_ref.y(), h_xyz_ref.z());

  xyz0.set_data_sync(h_xyz0);

  // Load constraint indices
  int *h_solvent_ind = (int *)malloc(nsolvent*3*sizeof(int));
  load_vec<int>(3, "test_data/solvent_ind.txt", nsolvent, h_solvent_ind);

  int *h_pair_ind = (int *)malloc(npair*2*sizeof(int));
  load_vec<int>(2, "test_data/pair_ind.txt", npair, h_pair_ind);

  int *h_trip_ind = (int *)malloc(ntrip*3*sizeof(int));
  load_vec<int>(3, "test_data/trip_ind.txt", ntrip, h_trip_ind);

  int *h_quad_ind = (int *)malloc(nquad*4*sizeof(int));
  load_vec<int>(4, "test_data/quad_ind.txt", nquad, h_quad_ind);

  //-------------------------
  // Test parametric
  //-------------------------
  test_parametric(mO, mH, rOHsq, rHHsq, npair, h_pair_ind, ntrip, h_trip_ind, nquad, h_quad_ind,
		  nsolvent, h_solvent_ind, xyz0, xyz1, h_xyz1);
  check_results(xyz1, h_xyz_ref, nsolvent, h_solvent_ind, npair, h_pair_ind,
		ntrip, h_trip_ind, nquad, h_quad_ind);

  //hostXYZ<double> h_xyz_tmp(xyz1);

  //-------------------------
  // Test indexed
  //-------------------------
  xyz1.set_data_sync(h_xyz1);
  test_indexed(mO, mH, rOHsq, rHHsq, npair, ntrip, nquad, nsolvent, h_solvent_ind, xyz0, xyz1, h_xyz1);
  check_results(xyz1, h_xyz_ref, nsolvent, h_solvent_ind, npair, h_pair_ind,
  		ntrip, h_trip_ind, nquad, h_quad_ind);
  //check_results(xyz1, h_xyz_tmp, nsolvent, h_solvent_ind, npair, h_pair_ind,
  //		ntrip, h_trip_ind, nquad, h_quad_ind);

  free(h_solvent_ind);

  free(h_pair_ind);
  free(h_trip_ind);
  free(h_quad_ind);


}
