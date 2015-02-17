#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "CudaPMEDirectForceBlock.h"
#include "CudaDirectForceKernels.h"

//
// Merge results from calc_force
//
__global__ void mergeNonbondResultsKernel(const int numBlock,
					  const long long int* __restrict__ biflam_in,
					  const long long int* __restrict__ biflam2_in,
					  double* __restrict__ biflam,
					  double* __restrict__ biflam2) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < numBlock) {
    long long int val1 = biflam_in[tid];
    long long int val2 = biflam2_in[tid];
    atomicAdd(&biflam[tid], ((double)val1)*INV_FORCE_SCALE_VIR);
    atomicAdd(&biflam2[tid], ((double)val2)*INV_FORCE_SCALE_VIR);
  }
}

//
// merge results from calc_14_force
//
__global__ void merge14ResultsKernel(const int m,
				     const int* __restrict__ lowTriangleIJ,
				     const float* __restrict__ blockParam,
				     const int* __restrict__ siteMLD,
				     const float* __restrict__ bixlam,
				     const double* __restrict__ energyVdw14Block,
				     const double* __restrict__ energyElec14Block,
				     const double* __restrict__ energyExcl14Block,
				     double* __restrict__ biflam,
				     double* __restrict__ biflam2,
				     double* __restrict__ energyVdw,
				     double* __restrict__ energyElec,
				     double* __restrict__ energyExcl) {
  // Shared memory required: min(m, blockDim.x)*3*sizeof(double)
  extern __shared__ double sh_energyBuf[];
  // Limit for shared memory access
  const int shlim = min(m, blockDim.x);
  volatile double* sh_energyVdw = &sh_energyBuf[0];
  volatile double* sh_energyElec = &sh_energyBuf[shlim];
  volatile double* sh_energyExcl = &sh_energyBuf[shlim*2];

  const int k = threadIdx.x + blockIdx.x*blockDim.x;

  if (threadIdx.x < shlim) {
    sh_energyVdw[threadIdx.x] = 0.0;
    sh_energyElec[threadIdx.x] = 0.0;
    sh_energyExcl[threadIdx.x] = 0.0;
  }
    
  if (k < m) {
    // lower triangle indices ib and jb could be calculated from: ceil((ceil(sqrt(1+8*k))-1)/2)-1;
    // However, I'm worried about rounding errors so I'll use pre-calculated table here
    // lowTriangleIbJB = (jb << 16) | ib
    int ib = lowTriangleIJ[k];
    int jb = ib;
    ib &= 0xffff;
    jb >>= 16;
    float fscale = blockParam[k];

    double energyVdw14BlockVal = energyVdw14Block[k];
    double energyElec14BlockVal = energyElec14Block[k];
    
    if (fscale != 1.0f && fscale > 0.0f) {
      int ib_site = siteMLD[ib];
      int jb_site = siteMLD[jb];
      int ibb = (ib == jb) ? ib : ( ib == 0 ? jb : (jb == 0 ? ib : -1) );
      double energyTot = energyVdw14BlockVal + energyElec14BlockVal;
      if (ibb >= 0) {
	atomicAdd(&biflam[ibb], energyTot);
      } else if (ib_site != jb_site) {
	atomicAdd(&biflam2[ib], ((double)bixlam[ib])*energyTot);
	atomicAdd(&biflam2[jb], ((double)bixlam[jb])*energyTot);
      }
    }
    
    //if (fscale /= one .and. fscale > zero) then
    //   call msld_lambdaforce(ibl, jbl, vdwpot_block, biflam_loc, biflam2_loc)
    //   call msld_lambdaforce(ibl, jbl, coulpot_block, biflam_loc, biflam2_loc)
    //endif

    /*
    subroutine msld_lambdaforce(ibl,jbl,energy,biflam_loc,biflam2_loc)
      integer, intent(in) :: ibl, jbl
      real(chm_real), intent(in) :: energy
      real(chm_real), intent(inout), optional :: biflam_loc(:), biflam2_loc(:)

      if (present(biflam_loc) .and. present(biflam2_loc)) then
         if (ibl.eq.jbl) then
            biflam_loc(ibl) = biflam_loc(ibl) + energy
         elseif (ibl.eq.1) then
            biflam_loc(jbl) = biflam_loc(jbl) + energy
         elseif (jbl.eq.1) then
            biflam_loc(ibl) = biflam_loc(ibl) + energy
         elseif (isitemld(ibl).ne.isitemld(jbl)) then
            biflam2_loc(jbl) = biflam2_loc(jbl) + bixlam(ibl)*energy
            biflam2_loc(ibl) = biflam2_loc(ibl) + bixlam(jbl)*energy
         endif
    */
    
    // Store energy into shared memory
    double fscaled = (double)fscale;
    sh_energyVdw[threadIdx.x]  = fscaled*energyVdw14BlockVal;
    sh_energyElec[threadIdx.x] = fscaled*energyElec14BlockVal;
    sh_energyExcl[threadIdx.x] = fscaled*energyExcl14Block[k];    
  }

  // Reduce energies within thread block
  __syncthreads();
  for (int d=1;d < shlim;d *= 2) {
      int pos = threadIdx.x + d;
      double energyVdw_val  = (pos < shlim) ? sh_energyVdw[pos] : 0.0;
      double energyElec_val = (pos < shlim) ? sh_energyElec[pos] : 0.0;
      double energyExcl_val = (pos < shlim) ? sh_energyExcl[pos] : 0.0;
      __syncthreads();
      if (threadIdx.x < shlim) {
	sh_energyVdw[threadIdx.x]  += energyVdw_val;
	sh_energyElec[threadIdx.x] += energyElec_val;
	sh_energyExcl[threadIdx.x] += energyExcl_val;
      }
      __syncthreads();
  }
  
  // Write to global memory
  if (threadIdx.x == 0) {
    atomicAdd(energyVdw, sh_energyVdw[0]);
    atomicAdd(energyElec, sh_energyElec[0]);
    atomicAdd(energyExcl, sh_energyExcl[0]);
  }

}

//########################################################################################
//########################################################################################
//########################################################################################

//
// Class creator
//
template <typename AT, typename CT>
CudaPMEDirectForceBlock<AT, CT>::CudaPMEDirectForceBlock(CudaEnergyVirial &energyVirial,
							 const char *nameVdw, const char *nameElec, const char *nameExcl,
							 CudaBlock &cudaBlock) :
  CudaPMEDirectForce<AT,CT>(energyVirial, nameVdw, nameElec, nameExcl), cudaBlock(cudaBlock) {
  
  biflamLen = 0;
  biflam = NULL;
  
  biflam2Len = 0;
  biflam2 = NULL;

  energy14BlockBuffer = NULL;

  h_in14TblBlockPos = NULL;
  h_ex14TblBlockPos = NULL;
  
  // lowTriangleIbJB = (jb << 16) | ib
  int m = cudaBlock.getNumBlock()*(cudaBlock.getNumBlock()+1)/2;
  int *h_lowTriangleIJ = new int[m];
  int k = 0;
  for (int jb=0;jb < cudaBlock.getNumBlock();jb++) {
    for (int ib=jb;ib < cudaBlock.getNumBlock();ib++) {
      h_lowTriangleIJ[k] = (jb << 16) | ib;
      k++;
    }
  }
  allocate<int>(&lowTriangleIJ, m);
  copy_HtoD_sync<int>(h_lowTriangleIJ, lowTriangleIJ, m);
  delete [] h_lowTriangleIJ;
}

//
// Class destructor
//
template <typename AT, typename CT>
CudaPMEDirectForceBlock<AT, CT>::~CudaPMEDirectForceBlock() {
  if (biflam != NULL) deallocate<AT>(&biflam);
  if (biflam2 != NULL) deallocate<AT>(&biflam2);
  if (energy14BlockBuffer != NULL) deallocate<double>(&energy14BlockBuffer);
  if (h_in14TblBlockPos != NULL) delete [] h_in14TblBlockPos;
  if (h_ex14TblBlockPos != NULL) delete [] h_ex14TblBlockPos;
  deallocate<int>(&lowTriangleIJ);
}

//
// Set values for 1-4 block position tables
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::set14BlockPos(int *h_in14TblBlockPos_in, int *h_ex14TblBlockPos_in) {
  int m = cudaBlock.getNumBlock()*(cudaBlock.getNumBlock()+1)/2;
  if (h_in14TblBlockPos == NULL) h_in14TblBlockPos = new int[m+1];
  if (h_ex14TblBlockPos == NULL) h_ex14TblBlockPos = new int[m+1];
  for (int i=0;i < m+1;i++) {
    h_in14TblBlockPos[i] = h_in14TblBlockPos_in[i];
    h_ex14TblBlockPos[i] = h_ex14TblBlockPos_in[i];
  }
}

//
// Calculates 1-4 exclusions and interactions
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::calc_14_force(const float4 *xyzq,
						    const bool calc_energy, const bool calc_virial,
						    const int stride, AT *force, cudaStream_t stream) {
  if (this->use_tex_vdwparam14) {
#ifdef USE_TEXTURE_OBJECTS
    if (!this->vdwParam14TexObjActive) {
      std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_14_force, vdwParam14TexObj must be created" << std::endl;
      exit(1);
    }
#else
    if (!get_vdwparam14_texref_bound()) {
      std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_14_force, vdwparam14_texref must be bound" << std::endl;
      exit(1);
    }
#endif
  }

  int nthread = 512;
  int shmem_size = 0;
  if (calc_energy) {
    shmem_size = nthread*sizeof(double2);
  }

  int vdw_model_loc = this->calc_vdw ? this->vdw_model : NONE;
  int elec_model_loc = this->calc_elec ? this->elec_model : NONE;
  if (elec_model_loc == NONE && vdw_model_loc == NONE) return;

  int m = cudaBlock.getNumBlock()*(cudaBlock.getNumBlock()+1)/2;

  if (calc_energy) {
    if (energy14BlockBuffer == NULL) {
      allocate<double>(&energy14BlockBuffer, m*3);
      energyVdw14Block = &energy14BlockBuffer[0];
      energyElec14Block = &energy14BlockBuffer[m];
      energyExcl14Block = &energy14BlockBuffer[m*2];
    }
    clear_gpu_array<double>(energy14BlockBuffer, m*3, stream);
  }
  
  for (int k=0;k < m;k++) {
    float fscale = cudaBlock.getBlockParamValue(k);
    int pos_in14 = h_in14TblBlockPos[k];
    int num_in14 = h_in14TblBlockPos[k+1] - h_in14TblBlockPos[k];
    int pos_ex14 = h_ex14TblBlockPos[k];
    int num_ex14 = h_ex14TblBlockPos[k+1] - h_ex14TblBlockPos[k];
    int nin14block = (num_in14 - 1)/nthread + 1;
    int nex14block = (num_ex14 - 1)/nthread + 1;
    int nblock = nin14block + nex14block;
    calcForce14KernelChoice<AT,CT>(nblock, nthread, shmem_size, stream,
				   vdw_model_loc, elec_model_loc, calc_energy, calc_virial,
				   num_in14, &this->in14list[pos_in14],
				   num_ex14, &this->ex14list[pos_ex14],
				   nin14block, this->vdwtype, this->vdwparam14,
#ifdef USE_TEXTURE_OBJECTS
				   this->vdwParam14TexObj,
#endif
				   xyzq, fscale, stride, force,
				   this->energyVirial.getVirialPointer(),
				   &energyVdw14Block[k], &energyElec14Block[k],
				   &energyExcl14Block[k]);
  }
  if (calc_energy) {
    nthread = min( ((m-1)/warpsize+1)*warpsize, get_max_nthread());
    shmem_size = min(m, nthread)*3*sizeof(double);
    // Check if we want too much shared memory (this should not happen)
    if (shmem_size > get_max_shmem_size()) {
      std::cout << "CudaPMEDirectForceBlock::calc_14_force, amount of shared memory exceeded" << std::endl;
      exit(1);
    }
    int nblock = (m - 1)/nthread + 1;
    merge14ResultsKernel<<< nblock, nthread, shmem_size, stream >>>
      (m, lowTriangleIJ, cudaBlock.getBlockParam(), cudaBlock.getSiteMLD(), cudaBlock.getBixlam(),
       energyVdw14Block, energyElec14Block, energyExcl14Block,
       cudaBlock.getBiflam(), cudaBlock.getBiflam2(),
       this->energyVirial.getEnergyPointer(this->strVdw),
       this->energyVirial.getEnergyPointer(this->strElec),
       this->energyVirial.getEnergyPointer(this->strExcl));
    cudaCheck(cudaGetLastError());
  }
}

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
  if (!this->vdwParamTexObjActive) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_force, vdwParamTexObj must be created" << std::endl;
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
			       nlist, this->vdwparam, this->nvdwparam, this->vdwtype,
#ifdef USE_TEXTURE_OBJECTS
			       this->vdwParamTexObj,
#endif
			       xyzq, stride, force,
			       this->energyVirial.getVirialPointer(),
			       this->energyVirial.getEnergyPointer(this->strVdw),
			       this->energyVirial.getEnergyPointer(this->strElec),
			       &cudaBlock, this->biflam, this->biflam2);
  
  // Convert biflam and biflam2 into double precision and add to cudaBlock.biflam -arrays
  if (calc_energy) {
    mergeNonbondResultsKernel<<< (cudaBlock.getNumBlock()-1)/64+1, 64, 0, stream >>>
      (cudaBlock.getNumBlock(), biflam, biflam2, cudaBlock.getBiflam(), cudaBlock.getBiflam2());
    cudaCheck(cudaGetLastError());
  }
  
}

//
// Explicit instances of CudaPMEDirectForceBlock
//
template class CudaPMEDirectForceBlock<long long int, float>;
