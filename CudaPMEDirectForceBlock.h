#ifndef CUDAPMEDIRECTFORCEBLOCK_H
#define CUDAPMEDIRECTFORCEBLOCK_H
#include <cuda.h>
#include "CudaPMEDirectForce.h"
#include "CudaBlock.h"

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

  CudaBlock &cudaBlock;
  
  // Local (fixed precision versions) force coefficients (size numBlock each)
  int biflamLen;
  AT *biflam;
  int biflam2Len;
  AT *biflam2;
  
 public:

  CudaPMEDirectForceBlock(CudaEnergyVirial &energyVirial,
			  const char *nameVdw, const char *nameElec, const char *nameExcl,
			  CudaBlock &cudaBlock);
  ~CudaPMEDirectForceBlock();

  //void calc_14_force(const float4 *xyzq,
  //		     const bool calc_energy, const bool calc_virial,
  //		     const int stride, AT *force, cudaStream_t stream=0);

  void calc_force(const float4 *xyzq,
		  const CudaNeighborListBuild<32>& nlist,
		  const bool calc_energy,
		  const bool calc_virial,
		  const int stride, AT *force,
		  cudaStream_t stream=0);

};

#endif // CUDAPMEDIRECTFORCEBLOCK_H
