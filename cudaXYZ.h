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
// CUDA XYZ array class
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//

template <typename T>
class cudaXYZ : public XYZ<T> {

public:

  cudaXYZ() { }

  cudaXYZ(int size){
    this->resize(size);
  }

  template <typename P>
    cudaXYZ(hostXYZ<P> &xyz){
    this->resize(xyz.size());
    this->set_data(xyz);
  }

  ~cudaXYZ() {
    this->_size = 0;
    this->_capacity = 0;
    if (this->_x != NULL) deallocate<T>(&this->_x);
    if (this->_y != NULL) deallocate<T>(&this->_y);
    if (this->_z != NULL) deallocate<T>(&this->_z);
  }

  void realloc_array(T** array, int* capacity, float fac) {
    reallocate<T>(array, capacity, this->_size, fac);
  }

  // Clears the data array
  void clear(cudaStream_t stream=0) {
    clear_gpu_array<T>(this->_x, this->_size, stream);
    clear_gpu_array<T>(this->_y, this->_size, stream);
    clear_gpu_array<T>(this->_z, this->_size, stream);
  }

  //--------------------------------------------------------------------------

  // Sets data from hostXYZ
  template <typename P>
  void set_data(hostXYZ<P> &xyz, cudaStream_t stream=0) {
    assert(this->match(xyz));
    copy_HtoD<T>((T *)xyz.x(), this->_x, this->_size, stream);
    copy_HtoD<T>((T *)xyz.y(), this->_y, this->_size, stream);
    copy_HtoD<T>((T *)xyz.z(), this->_z, this->_size, stream);
  }

  // Sets data from hostXYZ synchroniously
  template <typename P>
  void set_data_sync(hostXYZ<P> &xyz) {
    assert(this->match(xyz));
    copy_HtoD_sync<T>((T *)xyz.x(), this->_x, this->_size);
    copy_HtoD_sync<T>((T *)xyz.y(), this->_y, this->_size);
    copy_HtoD_sync<T>((T *)xyz.z(), this->_z, this->_size);
  }

  // Sets data from cudaXYZ
  template <typename P>
  void set_data(cudaXYZ<P> &xyz, cudaStream_t stream=0) {
    assert(this->match(xyz));
    copy_DtoD<T>((T *)xyz._x, this->_x, this->_size, stream);
    copy_DtoD<T>((T *)xyz._y, this->_y, this->_size, stream);
    copy_DtoD<T>((T *)xyz._z, this->_z, this->_size, stream);
  }

  // Sets data from cudaXYZ synchroniously
  template <typename P>
  void set_data_sync(cudaXYZ<P> &xyz) {
    assert(this->match(xyz));
    copy_DtoD_sync<T>((T *)xyz._x, this->_x, this->_size);
    copy_DtoD_sync<T>((T *)xyz._y, this->_y, this->_size);
    copy_DtoD_sync<T>((T *)xyz._z, this->_z, this->_size);
  }

  /*
  // Sets data from cudaXYZ pointer
  template <typename P>
  void set_data(cudaXYZ<P> *xyz, cudaStream_t stream=0) {
    assert(this->match(xyz));
    copy_DtoD<T>((T *)xyz->data, this->data, 3*this->stride, stream);
  }
  */

  /*
  // Sets data from cudaXYZ
  template <typename P>
  void set_data_sync(cudaXYZ<P> *xyz) {
    assert(this->match(xyz));
    copy_DtoD_sync<T>((T *)xyz->data, this->data, 3*this->stride);
  }
  */

  // Sets data from host arrays
  void set_data_sync(const int size, const T *h_x, const T *h_y, const T *h_z) {
    assert(this->_size == size);
    copy_HtoD_sync<T>(h_x, this->_x, this->_size);
    copy_HtoD_sync<T>(h_y, this->_y, this->_size);
    copy_HtoD_sync<T>(h_z, this->_z, this->_size);
  }

  // Sets data from host arrays with indexing
  void set_data_sync(const std::vector<int>& h_loc2glo, const T *h_x, const T *h_y, const T *h_z) {
    assert(this->_size == h_loc2glo.size());

    T *h_xt = new T[this->_size];
    T *h_yt = new T[this->_size];
    T *h_zt = new T[this->_size];

    for (int i=0;i < h_loc2glo.size();i++) {
      int j = h_loc2glo[i];
      h_xt[i] = h_x[j];
      h_yt[i] = h_y[j];
      h_zt[i] = h_z[j];
    }

    this->set_data_sync(this->_size, h_xt, h_yt, h_zt);

    delete [] h_xt;
    delete [] h_yt;
    delete [] h_zt;
  }

  //--------------------------------------------------------------------------

  // Copies data to host buffers (x, y, z)
  void get_data_sync(const int size, double *h_x, double *h_y, double *h_z) {
    assert(size == this->_size);
    copy_DtoH_sync<T>(this->_x, h_x, this->_size);
    copy_DtoH_sync<T>(this->_y, h_y, this->_size);
    copy_DtoH_sync<T>(this->_z, h_z, this->_size);
  }

  //--------------------------------------------------------------------------

  /*
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
  */

};

#endif // CUDAXYZ_H
