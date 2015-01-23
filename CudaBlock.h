#ifndef CUDABLOCK_H
#define CUDABLOCK_H
//
// BLOCK / lambda dynamics storage class
//
// (c) Antti-Pekka Hynninen 2015
//
#include <cuda.h>

class CudaBlock {
private:

  // Number of blocks
  const int numBlock;

  // Block type for each atom (ncoord -size), values in range 0...numBlock-1
  int blockTypeLen;
  int *blockType;

  // parameter (lambda) for each block pair, size numBlock*(numBlock+1)/2
  float *blockParam;
#ifdef USE_TEXTURE_OBJECTS
  cudaTextureObject_t blockParamTexObj;
#endif

  // Coupling coefficients for sites (size numBlock)
  float *bixlam;
  
  // Results (size numBlock each)
  double *biflam;
  double *biflam2;
  
public:
  CudaBlock(const int numBlock);
  ~CudaBlock();

  void setBlockType(const int ncoord, const int *h_blockType);
  void setBlockParam(const double *h_blockParam);
  void setBixlam(const double *h_bixlam);
  
  int getNumBlock() {return numBlock;}
  int *getBlockType() {return blockType;}
  float *getBixlam() {return bixlam;}
  double *getBiflam() {return biflam;}
  double *getBiflam2() {return biflam2;}
  void getBiflam(double *h_biflam, double *h_biflam2);
#ifdef USE_TEXTURE_OBJECTS
  cudaTextureObject_t *getBlockParamTexObj() {return blockParamTexObj;}
#endif
  
};

#endif //CUDABLOCK_H
