#ifndef HOSTXYZ_H
#define HOSTXYZ_H
#include <cassert>
#include "cuda_utils.h"
#include "XYZ.h"

// Forward declaration of cudaXYZ
template<typename T> class cudaXYZ;

//
// Host XYZ strided array class
// By default host array is allocated pinned (PINNED)
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//

enum {NON_PINNED, PINNED};

template <typename T>
class hostXYZ : public XYZ<T> {

private:
  int type;

public:

  hostXYZ() {
    this->type = PINNED;
  }

  hostXYZ(int n, int type=PINNED) {
    this->type = type;
    this->resize(n);
  }

  hostXYZ(int n, int stride, T *data, int type=PINNED) {
    this->n = n;
    this->stride = stride;
    this->data = data;
    this->type = type;
  }

  ~hostXYZ() {
    this->n = 0;
    this->stride = 0;
    this->size = 0;
    if (this->data != NULL) {
      if (type == PINNED) {
	deallocate_host<T>(&this->data);
      } else {
	delete [] this->data;
      }
    }
  }

  void resize(int n, float fac=1.0f) {
    this->n = n;
    this->stride = calc_stride<T>(this->n);
    if (type == PINNED) {
      reallocate_host<T>(&this->data, &this->size, 3*this->stride, fac);
    } else {
      if (this->data != NULL && this->size < 3*this->stride) {
	delete [] this->data;
	this->data = NULL;
      }
      if (this->data == NULL) {
	if (fac > 1.0f) {
	  this->size = (int)(((double)(3*this->stride))*(double)(fac));
	} else {
	  this->size = 3*this->stride;
	}
	this->data = new T[this->size];
      }
    }
  }

  // Sets data from cudaXYZ object
  //  void set_data(cudaXYZ<T> &xyz, cudaStream_t stream=0) {
  //    assert(this->stride == xyz.stride);
  //    copy_DtoH<T>(xyz.data, this->data, 3*this->stride, stream);
  //  }

  // Sets data from cudaXYZ object
  template <typename P>
  void set_data(cudaXYZ<P> &xyz, cudaStream_t stream=0) {
    assert(this->match(xyz));
    copy_DtoH<T>((T *)xyz.data, this->data, 3*this->stride, stream);
  }

  // Sets data from cudaXYZ object
  template <typename P>
  void set_data_sync(cudaXYZ<P> &xyz) {
    assert(this->match(xyz));
    copy_DtoH_sync<T>((T *)xyz.data, this->data, 3*this->stride);
  }

  // Sets data from list of numbers on device
  void set_data_sync(const T *d_x, const T *d_y, const T *d_z) {
    copy_DtoH_sync<T>(d_x, this->data,                  this->n);
    copy_DtoH_sync<T>(d_y, &this->data[this->stride],   this->n);
    copy_DtoH_sync<T>(d_z, &this->data[this->stride*2], this->n);
  }

};


#endif // HOSTXYZ_H
