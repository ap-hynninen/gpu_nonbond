#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "CudaPMEDirectForceBlock.h"
#include "CudaDirectForceKernels.h"

__global__ void merge_biflam_kernel(const int numBlock,
				    const long long int* __restrict__ biflam_in,
				    const long long int* __restrict__ biflam2_in,
				    double* __restrict__ biflam,
				    double* __restrict__ biflam2) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < numBlock) {
    long long int val1 = biflam_in[tid];
    long long int val2 = biflam2_in[tid];
    biflam[tid]  += ((double)val1)*INV_FORCE_SCALE_VIR;
    biflam2[tid] += ((double)val2)*INV_FORCE_SCALE_VIR;
  }
}


//########################################################################################
//########################################################################################
//########################################################################################

//
// Class creator
//
template <typename AT, typename CT>
CudaPMEDirectForceBlock<AT, CT>::CudaPMEDirectForceBlock(CudaBlock &cudaBlock) : cudaBlock(cudaBlock) {
  biflamLen = 0;
  biflam = NULL;
  
  biflam2Len = 0;
  biflam2 = NULL;
}

//
// Class destructor
//
template <typename AT, typename CT>
CudaPMEDirectForceBlock<AT, CT>::~CudaPMEDirectForceBlock() {
  if (biflam != NULL) deallocate<AT>(&biflam);
  if (biflam2 != NULL) deallocate<AT>(&biflam2);
}

/*
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
*/

//
// Calculates direct force
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::calc_force(const float4 *xyzq,
						 const CudaNeighborListBuild<32>& nlist,
						 const bool calc_energy,
						 const bool calc_virial,
						 const int stride, AT *force,
						 cudaStream_t stream) {

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
  if (!get_vdwparam_texref_bound()) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_force, vdwparam_texref must be bound"
  	      << std::endl;
    exit(1);
  }
#endif

#ifndef NUMBLOCK_LARGE
  if (cudaBlock.getNumBlock() > 512) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_force, numBlock > 512 is not currently allowed" << std::endl;
    exit(1);
  }
#endif

  // Re-allocate biflam and biflam2
  reallocate<AT>(&biflam, &biflamLen, cudaBlock.getNumBlock());
  reallocate<AT>(&biflam2, &biflam2Len, cudaBlock.getNumBlock());

  // Clear biflam and biflam2
  clear_gpu_array<AT>(biflam, cudaBlock.getNumBlock(), stream);
  clear_gpu_array<AT>(biflam2, cudaBlock.getNumBlock(), stream);

  if (nlist.get_n_ientry() == 0) return;
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
  int nblock_tot = (nlist.get_n_ientry()-1)/(nthread/warpsize)+1;

  int shmem_size = 0;
  // (sh_xi, sh_yi, sh_zi, sh_qi, sh_vdwtypei, sh_blocktypei)
  if (get_cuda_arch() < 300)
    shmem_size += (nthread/warpsize)*tilesize*(sizeof(float)*4 + sizeof(int) + sizeof(int));
#ifndef NUMBLOCK_LARGE
  shmem_size += cudaBlock.getNumBlock()*sizeof(float);
#endif
  // (sh_fix, sh_fiy, sh_fiz)
  shmem_size += (nthread/warpsize)*warpsize*sizeof(AT)*3;
  // If no texture fetch for vdwparam:
  //shmem_size += nvdwparam*sizeof(float);

  if (calc_energy) shmem_size = max(shmem_size, (int)(nthread*sizeof(double)*2));
  if (calc_virial) shmem_size = max(shmem_size, (int)(nthread*sizeof(double)*3));

  calcForceKernelChoice<AT,CT>(nblock_tot, nthread, shmem_size, stream,
			       vdw_model_loc, elec_model_loc, calc_energy, calc_virial,
			       nlist, stride, this->vdwparam, this->nvdwparam, xyzq, this->vdwtype,
			       this->d_energy_virial, force, &cudaBlock, this->biflam, this->biflam2);

  /*
  int3 max_nblock3 = get_max_nblock();
  unsigned int max_nblock = max_nblock3.x;
  unsigned int base = 0;
  
  while (nblock_tot != 0) {

    int nblock = (nblock_tot > max_nblock) ? max_nblock : nblock_tot;
    nblock_tot -= nblock;

#ifdef USE_TEXTURE_OBJECTS
    CREATE_KERNELS(CREATE_KERNEL, calc_force_kernel, this->vdwparam_tex,
		   base, nlist.get_n_ientry(), nlist.get_ientry(), nlist.get_tile_indj(),
		   nlist.get_tile_excl(), stride, this->vdwparam, this->nvdwparam, xyzq, this->vdwtype,
		   cudaBlock.getNumBlock(), cudaBlock.getBixlam(), cudaBlock.getBlockType(),
		   this->biflam, this->biflam2, cudaBlock.getBlockParamTexObj(), force);
#else
    CREATE_KERNELS(CREATE_KERNEL, calc_force_kernel,
		   base, nlist.get_n_ientry(), nlist.get_ientry(), nlist.get_tile_indj(),
		   nlist.get_tile_excl(), stride, this->vdwparam, this->nvdwparam, xyzq, this->vdwtype,
		   cudaBlock.getNumBlock(), cudaBlock.getBixlam(), cudaBlock.getBlockType(),
		   this->biflam, this->biflam2, force);
#endif

    base += (nthread/warpsize)*nblock;

    cudaCheck(cudaGetLastError());
  }
  */
  
  // Convert biflam and biflam2 into double precision and add to cudaBlock.biflam -arrays
  merge_biflam_kernel<<< (cudaBlock.getNumBlock()-1)/64+1, 64, 0, stream >>>
    (cudaBlock.getNumBlock(), biflam, biflam2, cudaBlock.getBiflam(), cudaBlock.getBiflam2());
  cudaCheck(cudaGetLastError());
  
}

//
// Explicit instances of CudaPMEDirectForceBlock
//
template class CudaPMEDirectForceBlock<long long int, float>;
