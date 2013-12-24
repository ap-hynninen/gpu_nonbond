#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include "gpu_utils.h"
#include "reduce.h"
#include "cuda_utils.h"
#include "Force.h"

//
// Class creator
//
template <typename T>
Force<T>::Force(const int ncoord) : ncoord(ncoord) {
  stride = ((ncoord*sizeof(T) - 1)/256 + 1)*256/sizeof(T);
  allocate<T>(&data, 3*stride);
}

template <typename T>
Force<T>::Force(const char *filename) {
  std::ifstream file(filename);
  if (file.is_open()) {
    
    T fx, fy, fz;
    
    // Count number of coordinates
    ncoord = 0;
    while (file >> fx >> fy >> fz) ncoord++;

    stride = ((ncoord*sizeof(T) - 1)/256 + 1)*256/sizeof(T);

    // Rewind
    file.clear();
    file.seekg(0, std::ios::beg);
    
    // Allocate CPU memory
    T *data_cpu = new T[3*stride];
    
    // Read coordinates
    int i=0;
    while (file >> data_cpu[i] >> data_cpu[i+stride] >> data_cpu[i+stride*2]) i++;
    
    // Allocate GPU memory
    allocate<T>(&data, 3*stride);

    // Copy coordinates from CPU to GPU
    copy_HtoD<T>(data_cpu, data, 3*stride);

    // Deallocate CPU memory
    delete [] data_cpu;
    
  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }

}

//
// Class destructor
//
template <typename T>
Force<T>::~Force() {
  deallocate<T>(&data);
}

//
// Sets force data to zero
//
template <typename T>
void Force<T>::setzero() {
  clear_gpu_array<T>(data, 3*stride);
}

//
// Compares two force arrays, returns true if the difference is within tolerance
// NOTE: Comparison is done in double precision
//
template <typename T>
bool Force<T>::compare(Force<T>* force, const double tol, double& max_diff) {

  assert(force->ncoord == ncoord);

  T *h_data1 = new T[3*stride];
  T *h_data2 = new T[3*force->stride];

  copy_DtoH<T>(data,        h_data1, 3*stride);
  copy_DtoH<T>(force->data, h_data2, 3*force->stride);

  bool ok = true;

  max_diff = 0.0;

  int i;
  double fx1, fy1, fz1;
  double fx2, fy2, fz2;
  double diff;
  try {
    for (i=0;i < ncoord;i++) {
      fx1 = (double)h_data1[i];
      fy1 = (double)h_data1[i + stride];
      fz1 = (double)h_data1[i + 2*stride];
      fx2 = (double)h_data2[i];
      fy2 = (double)h_data2[i + force->stride];
      fz2 = (double)h_data2[i + 2*force->stride];
      if (isnan(fx1) || isnan(fy1) || isnan(fz1) || isnan(fx2) || isnan(fy2) || isnan(fz2)) throw 1;
      diff = max(fabs(fx1-fx2), max(fabs(fy1-fy2), fabs(fz1-fz2)));
      max_diff = max(diff, max_diff);
      if (diff > tol) throw 2;
    }
  }
  catch (int a) {
    std::cout << "i = "<< i << std::endl;
    std::cout << "fx1 fy1 fz1 = " << fx1 << " "<< fy1 << " "<< fz1 << std::endl;
    std::cout << "fx2 fy2 fz2 = " << fx2 << " "<< fy2 << " "<< fz2 << std::endl;
    if (a == 2) std::cout << "difference: " << diff << std::endl;
    ok = false;
  }

  delete [] h_data1;
  delete [] h_data2;
  
  return ok;
}

//
// Converts one type of force array to another. Result is in "force"
//
template <typename T>
template <typename T2>
void Force<T>::convert(Force<T2>* force) {

  assert(force->ncoord == ncoord);
  assert(force->stride == stride);

  int nthread = 512;
  int nblock = (3*stride - 1)/nthread + 1;

  reduce_data<T, T2> <<< nblock, nthread >>>(3*stride,
					     this->data,
					     force->data);
}

template class Force<long long int>;
template class Force<double>;
template class Force<float>;
