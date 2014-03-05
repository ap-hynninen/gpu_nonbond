#ifndef CUDAXYZ_H
#define CUDAXYZ_H
#include <cassert>
#include <iostream>
#include "cuda_utils.h"
#include "XYZ.h"

// Forward declaration of hostXYZ
template<typename T> class hostXYZ;

//
// CUDA XYZ strided array class
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//

template <typename T>
class cudaXYZ : public XYZ<T> {

public:

  cudaXYZ() { }

  cudaXYZ(int n) {
    this->resize(n);
  }

  ~cudaXYZ() {
    this->n = 0;
    this->stride = 0;
    this->size = 0;
    if (this->data != NULL) deallocate<T>(&this->data);
  }

  void resize(int n, float fac=1.0f) {
    this->n = n;
    this->stride = calc_stride<T>(this->n);
    reallocate<T>(&this->data, &this->size, 3*this->stride, fac);
  }

  // Clears the data array
  void clear(cudaStream_t stream=0) {
    clear_gpu_array<T>(this->data, 3*this->stride, stream);
  }

  //--------------------------------------------------------------------------

  // Sets data from hostXYZ
  void set_data(hostXYZ<T> &xyz, cudaStream_t stream=0) {
    assert(this->match(xyz));
    copy_HtoD<T>(xyz.data, this->data, 3*this->stride, stream);
  }

  // Sets data from cudaXYZ
  void set_data(cudaXYZ<T> &xyz, cudaStream_t stream=0) {
    assert(this->match(xyz));
    copy_DtoD<T>(xyz.data, this->data, 3*this->stride, stream);
  }

  // Sets data from hostXYZ
  void set_data_sync(hostXYZ<T> &xyz) {
    assert(this->match(xyz));
    copy_HtoD_sync<T>(xyz.data, this->data, 3*this->stride);
  }

  // Sets data from cudaXYZ
  void set_data_sync(cudaXYZ<T> &xyz) {
    assert(this->match(xyz));
    copy_DtoD_sync<T>(xyz.data, this->data, 3*this->stride);
  }

  //--------------------------------------------------------------------------

  // Sets data from (int n, int stride, T *xyz)
  void set_data_sync(int n, int stride, T *xyz) {
    assert(this->n == n);
    assert(this->stride == stride);
    copy_HtoD_sync<T>(xyz, this->data, 3*this->stride);
  }

};

#endif // CUDAXYZ_H
