#ifndef HOSTXYZ_H
#define HOSTXYZ_H
#include <cassert>
#include "cuda_utils.h"
#include "XYZ.h"

// Forward declaration of cudaXYZ
template<typename T> class cudaXYZ;

//
// Host XYZ array class
// By default host array is allocated pinned (PINNED)
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//

enum {NON_PINNED, PINNED};

template <typename T>
class hostXYZ : public XYZ<T> {

private:
  int _type;

public:

 hostXYZ() : _type(PINNED) {}

 hostXYZ(int size, int type=PINNED) : _type(type) {
    this->resize(size);
  }
  
 hostXYZ(int size, int capacity, T *x, T *y, T *z, int type=PINNED) : 
  _type(type), XYZ<T>(size, capacity, x, y, z) {}

  template <typename P>
    hostXYZ(cudaXYZ<P> &xyz, int type=PINNED) : _type(type) {
    this->resize(xyz.size());
    this->set_data_sync(xyz);
  }

  ~hostXYZ() {
    this->_size = 0;
    this->_capacity = 0;
    if (this->_x != NULL) {
      if (this->_type == PINNED) {
	deallocate_host<T>(&this->_x);
	deallocate_host<T>(&this->_y);
	deallocate_host<T>(&this->_z);
      } else {
	delete [] this->_x;
	delete [] this->_y;
	delete [] this->_z;
      }
    }
  }

  void realloc_array(T** array, int* capacity, float fac) {
    if (this->_type == PINNED) {
      reallocate_host<T>(array, capacity, this->_size, fac);
    } else {
      if (*array != NULL && this->_capacity < *capacity) {
	delete [] *array;
	*array = NULL;
      }
      if (*array == NULL) {
	if (fac > 1.0f) {
	  *capacity = (int)(((double)(this->_size))*(double)(fac));
	} else {
	  *capacity = this->_size;
	}
	*array = new T[*capacity];
      }      
    }
  }

  // Sets data from cudaXYZ object
  template <typename P>
  void set_data(cudaXYZ<P> &xyz, cudaStream_t stream=0) {
    assert(this->match(xyz));
    copy_DtoH<T>((T *)xyz.x(), this->_x, this->_size, stream);
    copy_DtoH<T>((T *)xyz.y(), this->_y, this->_size, stream);
    copy_DtoH<T>((T *)xyz.z(), this->_z, this->_size, stream);
  }

  // Sets data from cudaXYZ object with sync
  template <typename P>
  void set_data_sync(cudaXYZ<P> &xyz) {
    assert(this->match(xyz));
    copy_DtoH_sync<T>((T *)xyz.x(), this->_x, this->_size);
    copy_DtoH_sync<T>((T *)xyz.y(), this->_y, this->_size);
    copy_DtoH_sync<T>((T *)xyz.z(), this->_z, this->_size);
  }

  // Sets data from device arrays
  void set_data_sync(const int size, const T *d_x, const T *d_y, const T *d_z) {
    assert(size == this->_size);
    copy_DtoH_sync<T>(d_x, this->_x, this->_size);
    copy_DtoH_sync<T>(d_y, this->_y, this->_size);
    copy_DtoH_sync<T>(d_z, this->_z, this->_size);
  }

  // Sets data from host arrays
  void set_data_fromhost(const int size, const T *h_x, const T *h_y, const T *h_z) {
    assert(size == this->_size);
    for (int i=0;i < this->_size;i++) {
      this->_x[i] = h_x[i];
      this->_y[i] = h_y[i];
      this->_z[i] = h_z[i];
    }
  }

};


#endif // HOSTXYZ_H
