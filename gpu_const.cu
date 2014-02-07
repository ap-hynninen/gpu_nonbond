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
// Loads solvent_ind from file
//
void load_solvent_ind(const char *filename, const int nsolvent, int3 *solvent_ind) {

  std::ifstream file(filename);
  if (file.is_open()) {

    int i = 0;
    while (file >> solvent_ind[i].x >> solvent_ind[i].y >> solvent_ind[i].z) i++;

    if (i != nsolvent) {
      std::cerr<<"Incorrect number of lines in file "<<filename<<std::endl;
      exit(1);
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
  const double mO = 15.9994;
  const double mH = 1.008;
  const double rOHsq = 0.91623184;
  const double rHHsq = 2.29189321;
  const int ncoord = 23558;
  const int stride = ((ncoord-1)/32+1)*32;
  const int nsolvent = 7023;

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

  // Load solvent_ind
  int3 *h_solvent_ind = (int3 *)malloc(nsolvent*sizeof(int3));
  load_solvent_ind("test_data/solvent_ind.txt", nsolvent, h_solvent_ind);

  //--------------------------------------------------------------------------
  // Setup & Apply holonomic constraints
  //--------------------------------------------------------------------------

  HoloConst holoconst;

  // Setup
  holoconst.setup(mO, mH, rOHsq, rHHsq);
  holoconst.set_solvent_ind(nsolvent, h_solvent_ind);
  
  // Apply holonomic constraints
  holoconst.apply(xyz0, xyz1, stride);

  //--------------------------------------------------------------------------
  // Check result
  //--------------------------------------------------------------------------
  copy_DtoH<double>(xyz1, h_xyz1, stride*3);

  double max_diff;
  double tol = 1.0e-10;
  bool ok = true;
  for (int i=0;i < ncoord;i++) {
    double x1 = h_xyz1[i];
    double y1 = h_xyz1[i + stride];
    double z1 = h_xyz1[i + 2*stride];
    double x2 = h_xyz_ref[i];
    double y2 = h_xyz_ref[i + stride];
    double z2 = h_xyz_ref[i + 2*stride];

    std::cout << x1 << " "<< y1 << " "<< z1 << std::endl;
    std::cout << x2 << " "<< y2 << " "<< z2 << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    if (isnan(x1) || isnan(y1) || isnan(z1) || isnan(x2) || isnan(y2) || isnan(z2)) {
      std::cout << "NaN at i=" << i << std::endl;
      ok = false;
      break;
    }
    double diff = max(fabs(x1-x2), max(fabs(y1-y2), fabs(z1-z2)));
    max_diff = max(diff, max_diff);
    if (max_diff > tol) {
      std::cout << "force comparison FAILED, i,diff=" << i << " " << diff << std::endl;
      std::cout << x1 << " "<< y1 << " "<< z1 << std::endl;
      std::cout << x2 << " "<< y2 << " "<< z2 << std::endl;
      ok = false;
      break;
    }
  }

  if (ok) {
    std::cout<<"coordinate comparison OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

  free(h_xyz0);
  free(h_xyz1);
  free(h_xyz_ref);
  free(h_solvent_ind);
  deallocate<double>(&xyz0);
  deallocate<double>(&xyz1);
  
}
