#ifndef CUDAXYZ_H
#define CUDAXYZ_H
#include <cassert>
#include <iostream>
#include <vector>
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

  template <typename P>
  cudaXYZ(hostXYZ<P> &xyz) {
    this->resize(xyz.n);
    this->set_data(xyz);
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

  // Sets data from cudaXYZ pointer
  template <typename P>
  void set_data(cudaXYZ<P> *xyz, cudaStream_t stream=0) {
    assert(this->match(xyz));
    copy_DtoD<T>((T *)xyz->data, this->data, 3*this->stride, stream);
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

  // Sets data from cudaXYZ
  template <typename P>
  void set_data_sync(cudaXYZ<P> *xyz) {
    assert(this->match(xyz));
    copy_DtoD_sync<T>((T *)xyz->data, this->data, 3*this->stride);
  }

  // Sets data from host arrays
  void set_data_sync(const int n, const T *h_x, const T *h_y, const T *h_z) {
    assert(this->n == n);
    copy_HtoD_sync<T>(h_x, this->data, this->n);
    copy_HtoD_sync<T>(h_y, &this->data[this->stride], this->n);
    copy_HtoD_sync<T>(h_z, &this->data[this->stride*2], this->n);
  }

  // Sets data from host arrays with indexing
  void set_data_sync(const std::vector<int>& h_loc2glo, const T *h_x, const T *h_y, const T *h_z) {
    assert(this->n == h_loc2glo.size());

    T *h_data = new T[this->stride*3];
    for (int i=0;i < h_loc2glo.size();i++) {
      int j = h_loc2glo[i];
      h_data[i]                = h_x[j];
      h_data[i+this->stride]   = h_y[j];
      h_data[i+this->stride*2] = h_z[j];
    }

    copy_HtoD_sync<T>(h_data, this->data, this->stride*3);

    delete [] h_data;
  }

  //--------------------------------------------------------------------------

  // Copies data to host buffers (x, y, z)
  void get_data_sync(double *h_x, double *h_y, double *h_z) {
    copy_DtoH_sync<T>(this->data,                  h_x, this->n);
    copy_DtoH_sync<T>(&this->data[this->stride],   h_y, this->n);
    copy_DtoH_sync<T>(&this->data[this->stride*2], h_z, this->n);
  }

  //--------------------------------------------------------------------------

  void print(const int start, const int end, std::ostream& out) {
    assert((start >= 0) && (end >= start) && (end < this->n));
    T *h_data = new T[this->stride*3];
    copy_DtoH<T>(this->data, h_data, this->stride*3);

    for (int i=start;i <= end;i++) {
      out << i << " " << h_data[i] << " " << h_data[i+this->stride] << " "
	  << h_data[i+this->stride*2] << std::endl;
    }
    
    delete [] h_data;
  }

};

#endif // CUDAXYZ_H
