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

  cudaXYZ(int n, int stride, T *data) {
    this->n = n;
    this->stride = stride;
    this->data = data;
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
  template <typename P>
  void set_data(hostXYZ<P> &xyz, cudaStream_t stream=0) {
    assert(this->match(xyz));
    copy_HtoD<T>((T *)xyz.data, this->data, 3*this->stride, stream);
  }

  // Sets data from cudaXYZ
  template <typename P>
  void set_data(cudaXYZ<P> &xyz, cudaStream_t stream=0) {
    assert(this->match(xyz));
    copy_DtoD<T>((T *)xyz.data, this->data, 3*this->stride, stream);
  }

  // Sets data from hostXYZ
  template <typename P>
  void set_data_sync(hostXYZ<P> &xyz) {
    assert(this->match(xyz));
    copy_HtoD_sync<T>((T *)xyz.data, this->data, 3*this->stride);
  }

  // Sets data from cudaXYZ
  template <typename P>
  void set_data_sync(cudaXYZ<P> &xyz) {
    assert(this->match(xyz));
    copy_DtoD_sync<T>((T *)xyz.data, this->data, 3*this->stride);
  }

  // Sets data from list of numbers on host
  void set_data_sync(const int n, const T *h_x, const T *h_y, const T *h_z) {
    resize(n);
    copy_HtoD_sync<T>(h_x, this->data, this->n);
    copy_HtoD_sync<T>(h_y, &this->data[this->stride], this->n);
    copy_HtoD_sync<T>(h_z, &this->data[this->stride*2], this->n);
  }

  //--------------------------------------------------------------------------

};

#endif // CUDAXYZ_H
