#include <iostream>
#include <fstream>
#include <cassert>
#include "gpu_utils.h"
#include "reduce.h"
#include "cuda_utils.h"
#include "Force.h"
#include "hostXYZ.h"

//
// Class creators
//
template <typename T>
Force<T>::Force() {
}

template <typename T>
Force<T>::Force(const int ncoord) {
  xyz.resize(ncoord);
}

template <typename T>
Force<T>::Force(const char *filename) {
  std::ifstream file(filename);
  if (file.is_open()) {
    
    T fx, fy, fz;
    
    // Count number of coordinates
    int ncoord = 0;
    while (file >> fx >> fy >> fz) ncoord++;

    // Rewind
    file.clear();
    file.seekg(0, std::ios::beg);
    
    // Allocate CPU memory
    hostXYZ<T> xyz_cpu(ncoord, NON_PINNED);
    
    // Read coordinates
    int i=0;
    while (file >> xyz_cpu.data[i] 
	   >> xyz_cpu.data[i+xyz_cpu.stride] 
	   >> xyz_cpu.data[i+xyz_cpu.stride*2]) i++;

    // Allocate GPU memory
    xyz.resize(ncoord);

    // Copy coordinates from CPU to GPU
    xyz.set_data_sync(xyz_cpu);

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
}

//
// Sets force data to zero
//
template <typename T>
void Force<T>::clear(cudaStream_t stream) {
  xyz.clear(stream);
}

//
// Compares two force arrays, returns true if the difference is within tolerance
// NOTE: Comparison is done in double precision
//
template <typename T>
bool Force<T>::compare(Force<T>* force, const double tol, double& max_diff) {

  //assert(force->ncoord == ncoord);
  assert(force->xyz.n == xyz.n);

  hostXYZ<T> xyz1(xyz.n, NON_PINNED);
  hostXYZ<T> xyz2(force->xyz.n, NON_PINNED);

  xyz1.set_data(xyz);
  xyz2.set_data(force->xyz);

  bool ok = true;

  max_diff = 0.0;

  int i;
  double fx1, fy1, fz1;
  double fx2, fy2, fz2;
  double diff;
  try {
    for (i=0;i < xyz.n;i++) {
      fx1 = (double)xyz1.data[i];
      fy1 = (double)xyz1.data[i + xyz1.stride];
      fz1 = (double)xyz1.data[i + xyz1.stride*2];
      fx2 = (double)xyz2.data[i];
      fy2 = (double)xyz2.data[i + xyz2.stride];
      fz2 = (double)xyz2.data[i + xyz2.stride*2];
      if (isnan(fx1) || isnan(fy1) || isnan(fz1) || isnan(fx2) || isnan(fy2) || isnan(fz2)) throw 1;
      diff = max(fabs(fx1-fx2), max(fabs(fy1-fy2), fabs(fz1-fz2)));
      max_diff = max(diff, max_diff);
      if (diff > tol) throw 2;
    }
  }
  catch (int a) {
    std::cout << "i = "<< i << std::endl;
    std::cout << "this: fx1 fy1 fz1 = " << fx1 << " "<< fy1 << " "<< fz1 << std::endl;
    std::cout << "force:fx2 fy2 fz2 = " << fx2 << " "<< fy2 << " "<< fz2 << std::endl;
    if (a == 2) std::cout << "difference: " << diff << std::endl;
    ok = false;
  }

  return ok;
}

//
// Sets the size of the force array
//
template <typename T>
void Force<T>::set_ncoord(int ncoord, float fac) {
  xyz.resize(ncoord, fac);
}

//
// Returns stride
//
template <typename T>
int Force<T>::get_stride() {
  return xyz.stride;
}

//
// Copies data to host
//
template <typename T>
void Force<T>::get_data_sync(T *fx, T *fy, T *fz) {
  
}

//
// Converts one type of force array to another. Result is in "force"
//
template <typename T>
template <typename T2>
void Force<T>::convert(Force<T2>* force, cudaStream_t stream) {

  assert(force->xyz.n == xyz.n);

  if (force->xyz.stride == xyz.stride) {
    int nthread = 512;
    int nblock = (3*xyz.stride - 1)/nthread + 1;
    reduce_force<T, T2>
      <<< nblock, nthread, 0, stream >>>(3*xyz.stride, xyz.data, force->xyz.data);
    cudaCheck(cudaGetLastError());
  } else {
    int nthread = 512;
    int nblock = (xyz.n - 1)/nthread + 1;
    reduce_force<T, T2>
      <<< nblock, nthread, 0, stream >>>(xyz.n, xyz.stride, xyz.data,
					 force->xyz.stride, force->xyz.data);
    cudaCheck(cudaGetLastError());
  }
}

//
// Converts one type of force array to another. Result is in "force"
//
template <typename T>
template <typename T2, typename T3>
void Force<T>::convert_to(Force<T3>* force, cudaStream_t stream) {

  assert(force->xyz.n == xyz.n);
  assert(force->xyz.stride == xyz.stride);
  assert(sizeof(T2) == sizeof(T3));

  int nthread = 512;
  int nblock = (3*xyz.stride - 1)/nthread + 1;

  reduce_force<T, T2>
    <<< nblock, nthread, 0, stream >>>(3*xyz.stride, xyz.data, (T2 *)force->xyz.data);
  cudaCheck(cudaGetLastError());
}

//
// Converts one type of force array to another. Result is in "this"
// NOTE: Only works when the size of the types T and T2 match
//
template <typename T>
template <typename T2>
void Force<T>::convert(cudaStream_t stream) {

  assert(sizeof(T) == sizeof(T2));

  int nthread = 512;
  int nblock = (3*xyz.stride - 1)/nthread + 1;

  reduce_force<T, T2>
    <<< nblock, nthread, 0, stream >>>(3*xyz.stride, xyz.data);
  cudaCheck(cudaGetLastError());
}

//
// Converts one type of force array to another and adds force to the result.
// Result is in "this"
// NOTE: Only works when the size of the types T and T2 match
//
template <typename T>
template <typename T2, typename T3>
void Force<T>::convert_add(Force<T3> *force, cudaStream_t stream) {

  assert(force->xyz.stride == xyz.stride);
  assert(sizeof(T) == sizeof(T2));

  int nthread = 512;
  int nblock = (3*xyz.stride - 1)/nthread + 1;

  reduce_add_force<T, T2, T3>
    <<< nblock, nthread, 0, stream >>>(3*xyz.stride, force->xyz.data, xyz.data);
  cudaCheck(cudaGetLastError());
}

//
// Adds non-strided force_data
//
template <typename T>
template <typename T2>
void Force<T>::add(float3 *force_data, int force_n, cudaStream_t stream) {

  assert(force_n <= xyz.n);
  assert(sizeof(T) == sizeof(T2));

  int nthread = 512;
  int nblock = (force_n - 1)/nthread + 1;

  add_nonstrided_force<<< nblock, nthread, 0, stream >>>
    (force_n, force_data, xyz.stride, (double *)xyz.data);
  cudaCheck(cudaGetLastError());
}

//
// Explicit instances of Force class
//
template class Force<long long int>;
template class Force<double>;
template class Force<float>;
template void Force<long long int>::convert<double>(cudaStream_t stream);
template void Force<long long int>::convert_add<double>(Force<float> *force, cudaStream_t stream);
template void Force<long long int>::convert<float>(Force<float>* force, cudaStream_t stream);
template void Force<long long int>::convert<double>(Force<double>* force, cudaStream_t stream);
template void Force<float>::convert_to<double>(Force<long long int> *force, cudaStream_t stream);
template void Force<long long int>::add<double>(float3 *force_data, int force_n, cudaStream_t stream);
