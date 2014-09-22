#ifndef CUDADOMDECRECIPLOOPER_H
#define CUDADOMDECRECIPLOOPER_H

#include "CudaDomdecRecip.h"
#include "CudaDomdecRecipComm.h"

class CudaDomdecRecipLooper {

  CudaDomdecRecip& recip;
  CudaDomdecRecipComm& recipComm;

  int force_len;
  float* force;

 public:
 CudaDomdecRecipLooper(CudaDomdecRecip& recip, CudaDomdecRecipComm& recipComm) : 
  recip(recip), recipComm(recipComm) {
    force_len = 0;
    force = NULL;
  }

  void run();

};

#endif // CUDADOMDECRECIPLOOPER_H
