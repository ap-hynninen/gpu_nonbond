#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "CudaPMEDirectForceBlock.h"

extern __constant__ DirectSettings_t d_setup;
extern __device__ DirectEnergyVirial_t d_energy_virial;

#ifndef USE_TEXTURE_OBJECTS
// VdW parameter texture reference
static texture<float2, 1, cudaReadModeElementType> vdwparam_block_texref;
static bool vdwparam_block_texref_bound = false;
static texture<float2, 1, cudaReadModeElementType> vdwparam14_block_texref;
static bool vdwparam14_block_texref_bound = false;
static texture<float, 1, cudaReadModeElementType> blockparam_texref;
#endif

#ifndef USE_TEXTURE_OBJECTS
#define VDWPARAM_TEXREF vdwparam_block_texref
#define VDWPARAM14_TEXREF vdwparam14_block_texref
#endif

//#define NUMBLOCK_LARGE

#define USE_BLOCK
#include "CudaDirectForce_util.h"
#undef USE_BLOCK


__global__ void convert_biflam_to_DP(const int numBlock,
				     long long int* __restrict__ biflam,
				     long long int* __restrict__ biflam2) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < numBlock) {
    long long int val1 = biflam[tid];
    long long int val2 = biflam2[tid];
    biflam[tid]  = ((double)val1)*INV_FORCE_SCALE_VIR;
    biflam2[tid] = ((double)val2)*INV_FORCE_SCALE_VIR;
  }
}


//########################################################################################
//########################################################################################
//########################################################################################

//
// Class creator
//
template <typename AT, typename CT>
CudaPMEDirectForceBlock<AT, CT>::CudaPMEDirectForceBlock(int numBlock) {
  assert(numBlock >= 1);
  blocktype_len = 0;
  blocktype = NULL;
  allocate<float>(&blockparam, numBlock*(numBlock+1)/2);
#ifdef USE_TEXTURE_OBJECTS
  blockparam_tex = 0;

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = blockparam;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = sizeof(CT)*8;
  resDesc.res.linear.sizeInBytes = numBlock*(numBlock+1)/2*sizeof(CT);

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  cudaCreateTextureObject(&blockparam_tex, &resDesc, &texDesc, NULL);
#else
  // Bind blockparam texture
  memset(&blockparam_texref, 0, sizeof(blockparam_texref));
  blockparam_texref.normalized = 0;
  blockparam_texref.filterMode = cudaFilterModePoint;
  blockparam_texref.addressMode[0] = cudaAddressModeClamp;
  blockparam_texref.channelDesc.x = 32;
  blockparam_texref.channelDesc.y = 0;
  blockparam_texref.channelDesc.z = 0;
  blockparam_texref.channelDesc.w = 0;
  blockparam_texref.channelDesc.f = cudaChannelFormatKindFloat;
  cudaCheck(cudaBindTexture(NULL, blockparam_texref, blockparam, numBlock*(numBlock+1)/2*sizeof(float)));
#endif
  allocate<float>(&bixlam, numBlock);
  allocate<AT>(&biflam, numBlock);
  allocate<AT>(&biflam2, numBlock);
}

//
// Class destructor
//
template <typename AT, typename CT>
CudaPMEDirectForceBlock<AT, CT>::~CudaPMEDirectForceBlock() {
  if (blocktype != NULL) deallocate<int>(&blocktype);
#ifdef USE_TEXTURE_OBJECTS
  if (blockparam_tex != 0) cudaDestroyTextureObject(blockparam_tex);
#else
  cudaCheck(cudaUnbindTexture(blockparam_texref));
#endif
  deallocate<float>(&blockparam);
  deallocate<float>(&bixlam);
  deallocate<AT>(&biflam);
  deallocate<AT>(&biflam2);
}

//
// Sets blocktype array from host memory
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::set_blocktype(const int ncoord, const int *h_blocktype) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord-1)/warpsize+1)*warpsize;
  reallocate<int>(&blocktype, &blocktype_len, ncoord_aligned, 1.2f);
  copy_HtoD_sync<int>(h_blocktype, blocktype, ncoord);
}

//
// Sets block parameters by copying them from CPU
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::set_blockparam(const double *h_blockparam) {
  float* h_blockparam_f = new float[numBlock*(numBlock+1)/2];
  for (int i=0;i < numBlock*(numBlock+1)/2;i++) h_blockparam_f[i] = (float)h_blockparam[i];
  copy_HtoD_sync<float>(h_blockparam_f, blockparam, numBlock*(numBlock+1)/2);
  delete [] h_blockparam_f;
}

//
// Sets bixlam by copying them from CPU
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::set_bixlam(const double *h_bixlam) {
  float* h_bixlam_f = new float[numBlock];
  for (int i=0;i < numBlock;i++) h_bixlam_f[i] = (float)h_bixlam[i];
  copy_HtoD_sync<CT>(h_bixlam_f, bixlam, numBlock);
  delete [] h_bixlam_f;
}

//
// Calculates 1-4 exclusions and interactions
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::calc_14_force(const float4 *xyzq,
						    const bool calc_energy, const bool calc_virial,
						    const int stride, AT *force, cudaStream_t stream) {

#ifdef USE_TEXTURE_OBJECTS
  if (this->vdwparam14_tex == 0) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_14_force, vdwparam14_tex must be created" << std::endl;
    exit(1);
  }
  //if (blockparam_tex == 0) {
  //std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_14_force, blockparam_tex must be created" << std::endl;
  //exit(1);
  //}
#else
  if (!vdwparam14_block_texref_bound) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_14_force, vdwparam14_block_texref must be bound"
	      << std::endl;
    exit(1);
  }
#endif

  int nthread = 512;
  int nin14block = (this->nin14list - 1)/nthread + 1;
  int nex14block = (this->nex14list - 1)/nthread + 1;
  int nblock = nin14block + nex14block;
  int shmem_size = 0;
  if (calc_energy) {
    shmem_size = nthread*sizeof(double2);
  }

  int vdw_model_loc = this->calc_vdw ? this->vdw_model : NONE;
  int elec_model_loc = this->calc_elec ? this->elec_model : NONE;
  if (elec_model_loc == NONE && vdw_model_loc == NONE) return;

#ifdef USE_TEXTURE_OBJECTS
  CREATE_KERNELS(CREATE_KERNEL14, calc_14_force_kernel, this->vdwparam14_tex,
		 this->nin14list, this->nex14list, nin14block, this->in14list, this->ex14list,
		 this->vdwtype, this->vdwparam14, xyzq, stride, force);
#else
  CREATE_KERNELS(CREATE_KERNEL14, calc_14_force_kernel,
		 this->nin14list, this->nex14list, nin14block, this->in14list, this->ex14list,
		 this->vdwtype, this->vdwparam14, xyzq, stride, force);
#endif

  cudaCheck(cudaGetLastError());
}

//
// Clears biflam and biflam2 arrays
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::clear_biflam(cudaStream_t stream) {
  clear_gpu_array<AT>(biflam, numBlock, stream);
  clear_gpu_array<AT>(biflam2, numBlock, stream);
}

//
// Copies biflam and biflam2 to CPU arrays
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::get_biflam(double *h_biflam, double *h_biflam2) {
  copy_DtoH_sync<double>((double *)biflam, h_biflam, numBlock);
  copy_DtoH_sync<double>((double *)biflam2, h_biflam2, numBlock);
}

//
// Calculates direct force
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::calc_force(const float4 *xyzq,
						 const CudaNeighborListBuild<32> *nlist,
						 const bool calc_energy,
						 const bool calc_virial,
						 const int stride, AT *force, cudaStream_t stream) {

  const int tilesize = 32;

#ifdef USE_TEXTURE_OBJECTS
  if (this->vdwparam_tex == 0) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_force, vdwparam_tex must be created" << std::endl;
    exit(1);
  }
  if (blockparam_tex == 0) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_force, blockparam_tex must be created" << std::endl;
    exit(1);
  }
#else
  if (!vdwparam_block_texref_bound) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_force, vdwparam_block_texref must be bound"
	      << std::endl;
    exit(1);
  }
#endif

  if (numBlock > 512) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_force, Larger than 512 number of blocks not currently allowed" << std::endl;
    exit(1);
  }
  
  if (nlist->get_n_ientry() == 0) return;
  int vdw_model_loc = this->calc_vdw ? this->vdw_model : NONE;
  int elec_model_loc = this->calc_elec ? this->elec_model : NONE;
  if (elec_model_loc == NONE && vdw_model_loc == NONE) return;

  int nwarp = 2;
  if (get_cuda_arch() < 300) {
    nwarp = 2;
  } else {
    nwarp = 4;
  }
  int nthread = warpsize*nwarp;
  int nblock_tot = (nlist->get_n_ientry()-1)/(nthread/warpsize)+1;

  int shmem_size = 0;
  // (sh_xi, sh_yi, sh_zi, sh_qi, sh_vdwtypei, sh_blocktypei)
  if (get_cuda_arch() < 300)
    shmem_size += (nthread/warpsize)*tilesize*(sizeof(float)*4 + sizeof(int) + sizeof(int));
  // (sh_fix, sh_fiy, sh_fiz)
  shmem_size += (nthread/warpsize)*warpsize*sizeof(AT)*3;
  // If no texture fetch for vdwparam:
  //shmem_size += nvdwparam*sizeof(float);

  if (calc_energy) shmem_size = max(shmem_size, (int)(nthread*sizeof(double)*2));
  if (calc_virial) shmem_size = max(shmem_size, (int)(nthread*sizeof(double)*3));

  int3 max_nblock3 = get_max_nblock();
  unsigned int max_nblock = max_nblock3.x;
  unsigned int base = 0;

  while (nblock_tot != 0) {

    int nblock = (nblock_tot > max_nblock) ? max_nblock : nblock_tot;
    nblock_tot -= nblock;

#ifdef USE_TEXTURE_OBJECTS
    CREATE_KERNELS(CREATE_KERNEL, calc_force_kernel, this->vdwparam_tex,
		   base, nlist->get_n_ientry(), nlist->get_ientry(), nlist->get_tile_indj(),
		   nlist->get_tile_excl(), stride, this->vdwparam, this->nvdwparam, xyzq, this->vdwtype,
		   this->numBlock, this->bixlam, this->blocktype, this->biflam, this->biflam2, blockparam_tex, force);
#else
    CREATE_KERNELS(CREATE_KERNEL, calc_force_kernel,
		   base, nlist->get_n_ientry(), nlist->get_ientry(), nlist->get_tile_indj(),
		   nlist->get_tile_excl(), stride, this->vdwparam, this->nvdwparam, xyzq, this->vdwtype,
		   this->numBlock, this->bixlam, this->blocktype, this->biflam, this->biflam2, force);
#endif

    base += (nthread/warpsize)*nblock;

    cudaCheck(cudaGetLastError());
  }

  // Convert biflam and biflam2 into double precision
  convert_biflam_to_DP<<< (numBlock-1)/64+1, 64, 0, stream >>>(numBlock, biflam, biflam2);
  cudaCheck(cudaGetLastError());
  
}

//
// Explicit instances of CudaPMEDirectForceBlock
//
template class CudaPMEDirectForceBlock<long long int, float>;

#ifndef USE_TEXTURE_OBJECTS
#undef VDWPARAM_TEXREF
#undef VDWPARAM14_TEXREF
#endif
