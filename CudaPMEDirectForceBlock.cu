#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "NeighborList.h"
#include "CudaPMEDirectForceBlock.h"

extern __constant__ DirectSettings_t d_setup;
extern __device__ DirectEnergyVirial_t d_energy_virial;

#define USE_BLOCK
#include "CudaDirectForce_util.h"
#undef USE_BLOCK

//########################################################################################
//########################################################################################
//########################################################################################

//
// Class creator
//
template <typename AT, typename CT>
CudaPMEDirectForceBlock<AT, CT>::CudaPMEDirectForceBlock(int nblock) {
  assert(nblock >= 1);
  blocktype_len = 0;
  blocktype = NULL;
  blockparam_tex = 0;
  allocate<float>(&blockparam, nblock*(nblock+1)/2);

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = blockparam;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = sizeof(CT)*8;
  resDesc.res.linear.sizeInBytes = nblock*(nblock+1)/2*sizeof(CT);

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  cudaCreateTextureObject(&blockparam_tex, &resDesc, &texDesc, NULL);

  // Bind blockparam texture
  //blockparam_texref.normalized = 0;
  //blockparam_texref.filterMode = cudaFilterModePoint;
  //blockparam_texref.addressMode[0] = cudaAddressModeClamp;
  //blockparam_texref.channelDesc.x = 32;
  //blockparam_texref.channelDesc.y = 0;
  //blockparam_texref.channelDesc.z = 0;
  //blockparam_texref.channelDesc.w = 0;
  //blockparam_texref.channelDesc.f = cudaChannelFormatKindFloat;
  //cudaCheck(cudaBindTexture(NULL, blockparam_texref, blockparam, nblock*(nblock+1)/2*sizeof(float)));
}

//
// Class destructor
//
template <typename AT, typename CT>
CudaPMEDirectForceBlock<AT, CT>::~CudaPMEDirectForceBlock() {
  if (blocktype != NULL) deallocate<int>(&blocktype);
  if (blockparam_tex != 0) cudaDestroyTextureObject(blockparam_tex);
  // Unbind texture
  //cudaCheck(cudaUnbindTexture(blockparam_texref));
  deallocate<float>(&blockparam);
}

//
// Sets blocktype array from host memory
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::set_blocktype(const int ncoord, const int *h_blocktype) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord-1)/warpsize+1)*warpsize;
  reallocate<int>(&blocktype, &blocktype_len, ncoord_aligned, 1.2f);
  copy_HtoD<int>(h_blocktype, blocktype, ncoord);
}

//
// Sets block parameters by copying them from CPU
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::set_blockparam(const CT *h_blockparam) {
  copy_HtoD<CT>(h_blockparam, blockparam, nblock*(nblock+1)/2);
}

//
// Calculates 1-4 exclusions and interactions
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::calc_14_force(const float4 *xyzq,
						    const bool calc_energy, const bool calc_virial,
						    const int stride, AT *force, cudaStream_t stream) {

  if (this->vdwparam14_tex == 0) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_14_force, vdwparam14_tex must be created" << std::endl;
    exit(1);
  }

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

  CREATE_KERNELS(CREATE_KERNEL14, calc_14_force_kernel, this->vdwparam14_tex,
		 this->nin14list, this->nex14list, nin14block, this->in14list, this->ex14list,
		 this->vdwtype, this->vdwparam14, xyzq, stride, force);

  cudaCheck(cudaGetLastError());
}

//
// Calculates direct force
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::calc_force(const float4 *xyzq,
						 const NeighborList<32> *nlist,
						 const bool calc_energy,
						 const bool calc_virial,
						 const int stride, AT *force, cudaStream_t stream) {

  const int tilesize = 32;

  if (this->vdwparam_tex == 0) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_force, vdwparam_tex must be created" << std::endl;
    exit(1);
  }

  if (blockparam_tex == 0) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_force, blockparam_tex must be created" << std::endl;
    exit(1);
  }

  if (nlist->n_ientry == 0) return;
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
  int nblock_tot = (nlist->n_ientry-1)/(nthread/warpsize)+1;

  int shmem_size = 0;
  // (sh_xi, sh_yi, sh_zi, sh_qi, sh_vdwtypei, sh_blocktypei)
  if (get_cuda_arch() < 300)
    shmem_size += (nthread/warpsize)*tilesize*(sizeof(float)*4 + sizeof(int) + sizeof(float));
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

    CREATE_KERNELS(CREATE_KERNEL, calc_force_kernel, this->vdwparam_tex,
		   base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
		   nlist->tile_excl, stride, this->vdwparam, this->nvdwparam, xyzq, this->vdwtype,
		   this->blocktype, blockparam_tex, force);

    base += (nthread/warpsize)*nblock;

    cudaCheck(cudaGetLastError());
  }

}

//
// Explicit instances of CudaPMEDirectForceBlock
//
template class CudaPMEDirectForceBlock<long long int, float>;
