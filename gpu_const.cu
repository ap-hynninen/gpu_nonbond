#include <iostream>
#include <fstream>
#include <cuda.h>
#include "cuda_utils.h"
#include "HoloConst.h"

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
void load_coord(const char *filename, const int stride, double *xyz) {

  std::ifstream file(filename);
  if (file.is_open()) {

    int i = 0;
    while (file >> xyz[i] >> xyz[i+stride] >> xyz[i+stride*2]) i++;

    if (i > stride) {
      std::cerr<<"Too many lines in file "<<filename<<std::endl;
      exit(1);
    }

  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }

}

//
// Loads indices from file
//
void load_ind(const int nind, const char *filename, const int n, int *ind) {

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
		  const double *xyz, const double *xyz_ref, const int stride,
		  const double tol, double &max_diff) {

  double x1, y1, z1;
  double x2, y2, z2;
  double diff;
  int imol, j, i;

  try {
    for (imol=0;imol < n;imol++) {

      for (j=0;j < nind;j++) {
	i = ind[imol*nind+j];
	x1 = xyz[i];
	y1 = xyz[i + stride];
	z1 = xyz[i + 2*stride];
	x2 = xyz_ref[i];
	y2 = xyz_ref[i + stride];
	z2 = xyz_ref[i + 2*stride];
	
	if (isnan(x1) || isnan(y1) || isnan(z1) || isnan(x2) || isnan(y2) || isnan(z2)) throw 1;
	diff = max(fabs(x1-x2), max(fabs(y1-y2), fabs(z1-z2)));
	max_diff = max(diff, max_diff);
	if (diff > tol) throw 1;
      }
    }
  }
  catch (int a) {
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

  return true;
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
  const int stride = ((ncoord-1)/32+1)*32;
  const int nsolvent = 7023;
  const int npair = 458;
  const int ntrip = 233;
  const int nquad = 99;

  // Load coordinates
  double *h_xyz0 = (double *)malloc(stride*3*sizeof(double));
  double *h_xyz1 = (double *)malloc(stride*3*sizeof(double));
  double *h_xyz_ref = (double *)malloc(stride*3*sizeof(double));
  load_coord("test_data/xyz0.txt", stride, h_xyz0);
  load_coord("test_data/xyz1.txt", stride, h_xyz1);
  load_coord("test_data/xyz_ref.txt", stride, h_xyz_ref);

  double *xyz0;
  double *xyz1;
  allocate<double>(&xyz0, stride*3);
  allocate<double>(&xyz1, stride*3);

  copy_HtoD<double>(h_xyz0, xyz0, stride*3);
  copy_HtoD<double>(h_xyz1, xyz1, stride*3);

  // Load constraint indices
  int *h_solvent_ind = (int *)malloc(nsolvent*3*sizeof(int));
  load_ind(3, "test_data/solvent_ind.txt", nsolvent, h_solvent_ind);

  int *h_pair_ind = (int *)malloc(npair*2*sizeof(int));
  load_ind(2, "test_data/pair_ind.txt", npair, h_pair_ind);

  int *h_trip_ind = (int *)malloc(ntrip*3*sizeof(int));
  load_ind(3, "test_data/trip_ind.txt", ntrip, h_trip_ind);

  int *h_quad_ind = (int *)malloc(nquad*4*sizeof(int));
  load_ind(4, "test_data/quad_ind.txt", nquad, h_quad_ind);

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

  //--------------------------------------------------------------------------
  // Setup & Apply holonomic constraints
  //--------------------------------------------------------------------------

  HoloConst holoconst;

  // Setup
  holoconst.setup(mO, mH, rOHsq, rHHsq);
  holoconst.set_solvent_ind(nsolvent, (int3 *)h_solvent_ind);
  holoconst.set_pair_ind(npair, (int2 *)h_pair_ind, h_pair_constr, h_pair_mass);
  holoconst.set_trip_ind(ntrip, (int3 *)h_trip_ind, h_trip_constr, h_trip_mass);
  holoconst.set_quad_ind(nquad, (int4 *)h_quad_ind, h_quad_constr, h_quad_mass);
  
  // Apply holonomic constraints
  holoconst.apply(xyz0, xyz1, stride);

  copy_HtoD<double>(h_xyz1, xyz1, stride*3);

  holoconst.apply2(xyz0, xyz1, stride);

  copy_HtoD<double>(h_xyz1, xyz1, stride*3);

  holoconst.apply2(xyz0, xyz1, stride);


  //--------------------------------------------------------------------------
  // Check result
  //--------------------------------------------------------------------------
  copy_DtoH<double>(xyz1, h_xyz1, stride*3);

  double max_diff;
  double tol = 5.0e-14;

  max_diff = 0.0;
  if (check_result(3, nsolvent, h_solvent_ind, h_xyz1, h_xyz_ref, stride, tol, max_diff)) {
    std::cout<<"solvent SETTLE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

  max_diff = 0.0;
  if (check_result(2, npair, h_pair_ind, h_xyz1, h_xyz_ref, stride, tol, max_diff)) {
    std::cout<<"pair SHAKE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

  max_diff = 0.0;
  if (check_result(3, ntrip, h_trip_ind, h_xyz1, h_xyz_ref, stride, tol, max_diff)) {
    std::cout<<"trip SHAKE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

  max_diff = 0.0;
  if (check_result(4, nquad, h_quad_ind, h_xyz1, h_xyz_ref, stride, tol, max_diff)) {
    std::cout<<"quad SHAKE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

  free(h_xyz0);
  free(h_xyz1);
  free(h_xyz_ref);

  free(h_solvent_ind);

  free(h_pair_ind);
  free(h_trip_ind);
  free(h_quad_ind);

  free(h_pair_constr);
  free(h_pair_mass);
  free(h_trip_constr);
  free(h_trip_mass);
  free(h_quad_constr);
  free(h_quad_mass);

  deallocate<double>(&xyz0);
  deallocate<double>(&xyz1);
  
}
