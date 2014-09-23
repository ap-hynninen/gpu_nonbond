#ifndef FORCE_H
#define FORCE_H

#include <cuda.h>
#include "cudaXYZ.h"
#include "hostXYZ.h"

//
// Simple storage class for forces
//
template <typename T>
class Force {

public:

  // Number of coordinates in the force array
  //int ncoord;

  // Stride of the force data:
  // x data is in data[0...ncoord-1];
  // y data is in data[stride...stride+ncoord-1];
  // z data is in data[stride*2...stride*2+ncoord-1];
  //int stride;

  // Force data
  //int data_len;
  //T *data;

  cudaXYZ<T> xyz;

  Force();
  Force(const int ncoord);
  Force(const char *filename);
  ~Force();


  void clear(cudaStream_t stream=0);
  bool compare(Force<T>* force, const double tol, double& max_diff);

  void set_ncoord(int ncoord, float fac=1.0f);
  int get_stride();

  void get_data_sync(T *fx, T *fy, T *fz);

  template <typename T2> void convert(Force<T2>* force, cudaStream_t stream=0);
  template <typename T2> void convert(cudaStream_t stream=0);
  template <typename T2, typename T3> void convert_to(Force<T3> *force, cudaStream_t stream=0);
  template <typename T2, typename T3> void convert_add(Force<T3> *force, cudaStream_t stream=0);
  template <typename T2, typename T3> void add(Force<T3> *force, cudaStream_t stream=0);
};


#endif // FORCE_H
