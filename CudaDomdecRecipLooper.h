#ifndef CUDADOMDECRECIPLOOPER_H
#define CUDADOMDECRECIPLOOPER_H

#include <cuda.h>
#include "CudaDomdecRecip.h"
#include "CudaDomdecRecipComm.h"

class CudaDomdecRecipLooper {

  CudaDomdecRecip& recip;
  CudaDomdecRecipComm& recipComm;

  // Coordinates in (X, Y, Z, Q) format
  XYZQ xyzq;

  // Forces
  int force_len;
  float3 *force;

  // Stream
  cudaStream_t stream;

 public:
 CudaDomdecRecipLooper(CudaDomdecRecip& recip, CudaDomdecRecipComm& recipComm) : 
  recip(recip), recipComm(recipComm), force_len(0), force(NULL) {
   cudaCheck(cudaStreamCreate(&stream));
   recip.set_stream(stream);
 }

  ~CudaDomdecRecipLooper() {
    if (force != NULL) deallocate<float3>(&force);
    cudaCheck(cudaStreamDestroy(stream));
  }

  void run();

};

#endif // CUDADOMDECRECIPLOOPER_H
