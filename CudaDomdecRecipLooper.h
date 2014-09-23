#ifndef CUDADOMDECRECIPLOOPER_H
#define CUDADOMDECRECIPLOOPER_H

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

 public:
 CudaDomdecRecipLooper(CudaDomdecRecip& recip, CudaDomdecRecipComm& recipComm) : 
  recip(recip), recipComm(recipComm), force_len(0), force(NULL) {}

  ~CudaDomdecRecipLooper() {
    if (force != NULL) deallocate<float3>(&force);
  }

  void run();

};

#endif // CUDADOMDECRECIPLOOPER_H
