#include <iostream>
#include <fstream>
#include <cassert>
#include "gpu_utils.h"
#include "reduce.h"
#include "cuda_utils.h"
#include "Force.h"
#include "hostXYZ.h"

template <typename T>
Force<T>::Force(const char *filename) {
  _size = 0;
  _stride = 0;
  _capacity = 0;
  _xyz = NULL;

  std::ifstream file(filename);
  if (file.is_open()) {
    
    T fx, fy, fz;
    
    // Count number of coordinates
    int nforce = 0;
    while (file >> fx >> fy >> fz) nforce++;

    // Rewind
    file.clear();
    file.seekg(0, std::ios::beg);
    
    // Allocate CPU memory
    hostXYZ<T> xyz_cpu(nforce, NON_PINNED);

    // Read coordinates
    int i=0;
    while (file >> xyz_cpu.x()[i] >> xyz_cpu.y()[i] >> xyz_cpu.z()[i]) i++;

    // Allocate GPU memory
    this->resize(nforce);

    // Copy coordinates from CPU to GPU
    copy_HtoD_sync<T>(xyz_cpu.x(), this->x(), nforce);
    copy_HtoD_sync<T>(xyz_cpu.y(), this->y(), nforce);
    copy_HtoD_sync<T>(xyz_cpu.z(), this->z(), nforce);

  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }

}

//
// Compares two force arrays, returns true if the difference is within tolerance
// NOTE: Comparison is done in double precision
//
template <typename T>
bool Force<T>::compare(Force<T>& force, const double tol, double& max_diff) {
  assert(force.size() == this->size());

  hostXYZ<T> xyz1(this->size(), NON_PINNED);
  hostXYZ<T> xyz2(force.size(), NON_PINNED);
  xyz1.set_data_sync(force.size(), force.x(), force.y(), force.z());
  xyz2.set_data_sync(this->size(), this->x(), this->y(), this->z());

  bool ok = true;

  max_diff = 0.0;

  int i;
  double fx1, fy1, fz1;
  double fx2, fy2, fz2;
  double diff;
  try {
    for (i=0;i < this->size();i++) {
      fx1 = (double)(xyz1.x()[i]);
      fy1 = (double)(xyz1.y()[i]);
      fz1 = (double)(xyz1.z()[i]);
      fx2 = (double)(xyz2.x()[i]);
      fy2 = (double)(xyz2.y()[i]);
      fz2 = (double)(xyz2.z()[i]);
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
// Converts one type of force array to another. Result is in "force"
//
template <typename T>
template <typename T2>
void Force<T>::convert(Force<T2>& force, cudaStream_t stream) {

  assert(force.size() == this->size());

  if (force.stride() == this->stride()) {
    int nthread = 512;
    int nblock = (3*this->stride() - 1)/nthread + 1;
    reduce_force<T, T2>
      <<< nblock, nthread, 0, stream >>>(3*this->stride(), this->xyz(), force.xyz());
    cudaCheck(cudaGetLastError());
  } else {
    int nthread = 512;
    int nblock = (this->size() - 1)/nthread + 1;
    reduce_force<T, T2>
      <<< nblock, nthread, 0, stream >>>(this->size(), this->stride(), this->xyz(),
					 force.stride(), force.xyz());
    cudaCheck(cudaGetLastError());
  }
}

//
// Converts one type of force array to another. Result is in "force"
//
template <typename T>
template <typename T2, typename T3>
void Force<T>::convert_to(Force<T3>& force, cudaStream_t stream) {

  assert(force.size() == this->size());
  assert(force.stride() == this->stride());
  assert(sizeof(T2) == sizeof(T3));

  int nthread = 512;
  int nblock = (3*this->stride() - 1)/nthread + 1;

  reduce_force<T, T2>
    <<< nblock, nthread, 0, stream >>>(3*this->stride(), this->xyz(), (T2 *)force.xyz());
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
  int nblock = (3*this->stride() - 1)/nthread + 1;

  reduce_force<T, T2>
    <<< nblock, nthread, 0, stream >>>(3*this->stride(), this->xyz());
  cudaCheck(cudaGetLastError());
}

//
// Converts one type of force array to another and adds force to the result.
// Result is in "this"
// NOTE: Only works when the size of the types T and T2 match
//
template <typename T>
template <typename T2, typename T3>
void Force<T>::convert_add(Force<T3>& force, cudaStream_t stream) {
  assert(force.stride() == this->stride());
  assert(sizeof(T) == sizeof(T2));

  int nthread = 512;
  int nblock = (3*this->stride() - 1)/nthread + 1;

  reduce_add_force<T, T2, T3>
    <<< nblock, nthread, 0, stream >>>(3*this->stride(), force.xyz(), this->xyz());
  cudaCheck(cudaGetLastError());
}

//
// Adds non-strided force_data
//
template <typename T>
template <typename T2>
void Force<T>::add(float3 *force_data, int force_n, cudaStream_t stream) {

  assert(force_n <= this->size());
  assert(sizeof(T) == sizeof(T2));

  int nthread = 512;
  int nblock = (force_n - 1)/nthread + 1;

  add_nonstrided_force<<< nblock, nthread, 0, stream >>>
    (force_n, force_data, this->stride(), (double *)this->xyz());
  cudaCheck(cudaGetLastError());
}

//
// Explicit instances of Force class
//
template class Force<long long int>;
template class Force<double>;
template class Force<float>;
template void Force<long long int>::convert<double>(cudaStream_t stream);
template void Force<long long int>::convert_add<double>(Force<float>& force, cudaStream_t stream);
template void Force<long long int>::convert<float>(Force<float>& force, cudaStream_t stream);
template void Force<long long int>::convert<double>(Force<double>& force, cudaStream_t stream);
template void Force<float>::convert_to<double>(Force<long long int>& force, cudaStream_t stream);
template void Force<long long int>::add<double>(float3 *force_data, int force_n, cudaStream_t stream);
