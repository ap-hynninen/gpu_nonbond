#ifndef HOSTXYZ_H
#define HOSTXYZ_H
#include <cassert>
#include "cuda_utils.h"
#include "XYZ.h"

// Forward declaration of cudaXYZ
template<typename T> class cudaXYZ;

//
// Host XYZ strided array class
// By default host array is allocated non-pinned (NON_PINNED)
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
    this->type = NON_PINNED;
  }

  hostXYZ(int n, int type=NON_PINNED) {
    this->type = type;
    this->resize(n);
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
  void set_data(cudaXYZ<T> &xyz, cudaStream_t stream=0) {
    assert(this->stride == xyz.stride);
    copy_DtoH<T>(xyz.data, this->data, 3*this->stride, stream);
  }

  // Sets data from cudaXYZ object
  void set_data_sync(cudaXYZ<T> &xyz) {
    assert(this->stride == xyz.stride);
    copy_DtoH_sync<T>(xyz.data, this->data, 3*this->stride);
  }

};


#endif // HOSTXYZ_H
