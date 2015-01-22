#ifndef CUDAPMEDIRECTFORCEBLOCK_H
#define CUDAPMEDIRECTFORCEBLOCK_H
#include <cuda.h>
#include "CudaPMEDirectForce.h"

//
// Calculates direct non-bonded interactions on GPU using BLOCK
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//
// AT = accumulation type
// CT = calculation type
//

template <typename AT, typename CT>
  class CudaPMEDirectForceBlock : public CudaPMEDirectForce<AT,CT> {

 private:

  // Number of blocks
  int numBlock;

  // block type for each atom (ncoord -size)
  int blocktype_len;
  int *blocktype;

  // parameter (lambda) for each block pair, size numBlock*(numBlock+1)/2
  float *blockparam;
#ifdef USE_TEXTURE_OBJECTS
  cudaTextureObject_t blockparam_tex;
#endif

  // Coupling coefficients for sites (size numBlock)
  float *bixlam;

  // Force coefficients (size numBlock each)
  AT *biflam;
  AT *biflam2;
  
 public:

  CudaPMEDirectForceBlock(int numBlock);
  ~CudaPMEDirectForceBlock();

  void set_blocktype(const int ncoord, const int *h_blocktype);

  void set_blockparam(const double *h_blockparam);

  void set_bixlam(const double *h_bixlam);

  void clear_biflam(cudaStream_t stream=0);
  void get_biflam(double *h_biflam, double *h_biflam2);
  
  void calc_14_force(const float4 *xyzq,
		     const bool calc_energy, const bool calc_virial,
		     const int stride, AT *force, cudaStream_t stream=0);

  void calc_force(const float4 *xyzq,
		  const CudaNeighborListBuild<32> *nlist,
		  const bool calc_energy,
		  const bool calc_virial,
		  const int stride, AT *force, cudaStream_t stream=0);

};

#endif // CUDAPMEDIRECTFORCEBLOCK_H
