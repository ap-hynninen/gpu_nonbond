#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "NeighborList.h"
#include "CudaPMEDirectForceBlock.h"

static __constant__ const float ccelec = 332.0716;

// Settings for direct computation in device memory
static __constant__ DirectSettings_t d_setup;

// Energy and virial in device memory
static __device__ DirectEnergyVirial_t d_energy_virial;

// VdW parameter texture reference
static texture<float2, 1, cudaReadModeElementType> vdwparam_texref;
static bool vdwparam_texref_bound = false;
static texture<float2, 1, cudaReadModeElementType> vdwparam14_texref;
static bool vdwparam14_texref_bound = false;

/*
//
// 1-4 exclusion and interaction calculation kernel
//
template <typename AT, typename CT, int vdw_model, int elec_model, 
	  bool calc_energy, bool calc_virial, bool tex_vdwparam>
__global__ void calc_14_force_kernel(const int nin14list, const int nex14list,
				     const int nin14block,
				     const xx14list_t* in14list, const xx14list_t* ex14list,
				     const int* vdwtype, const float* vdwparam14,
				     const float4* xyzq, const int stride, AT *force) {
  // Amount of shared memory required:
  // blockDim.x*sizeof(double2)
  extern __shared__ double2 shpot[];

  if (blockIdx.x < nin14block) {
    double vdw_pot, elec_pot;
    if (calc_energy) {
      vdw_pot = 0.0;
      elec_pot = 0.0;
    }

    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < nin14list) {
      calc_in14_force_device<AT, CT, vdw_model, elec_model, calc_energy, calc_virial, tex_vdwparam>
	(pos, in14list, vdwtype, vdwparam14, xyzq, stride, force, vdw_pot, elec_pot);
    }

    if (calc_energy) {
      shpot[threadIdx.x].x = vdw_pot;
      shpot[threadIdx.x].y = elec_pot;
      __syncthreads();
      for (int i=1;i < blockDim.x;i *= 2) {
	int t = threadIdx.x + i;
	double val1 = (t < blockDim.x) ? shpot[t].x : 0.0;
	double val2 = (t < blockDim.x) ? shpot[t].y : 0.0;
	__syncthreads();
	shpot[threadIdx.x].x += val1;
	shpot[threadIdx.x].y += val2;
	__syncthreads();
      }
      if (threadIdx.x == 0) {
	atomicAdd(&d_energy_virial.energy_vdw,  shpot[0].x);
	atomicAdd(&d_energy_virial.energy_elec, shpot[0].y);
      }
    }

  } else {
    double excl_pot;
    if (calc_energy) excl_pot = 0.0;

    int pos = threadIdx.x + (blockIdx.x-nin14block)*blockDim.x;
    if (pos < nex14list) {
      calc_ex14_force_device<AT, CT, elec_model, calc_energy, calc_virial>
	(pos, ex14list, xyzq, stride, force, excl_pot);
    }

    if (calc_energy) {
      shpot[threadIdx.x].x = excl_pot;
      __syncthreads();
      for (int i=1;i < blockDim.x;i *= 2) {
	int t = threadIdx.x + i;
	double val = (t < blockDim.x) ? shpot[t].x : 0.0;
	__syncthreads();
	shpot[threadIdx.x].x += val;
	__syncthreads();
      }
      if (threadIdx.x == 0) {
	atomicAdd(&d_energy_virial.energy_excl,  shpot[0].x);
      }
    }

  }

}

//
// Nonbonded force kernel
//
template <typename AT, typename CT, int tilesize, int vdw_model, int elec_model,
	  bool calc_energy, bool calc_virial, bool tex_vdwparam>
__global__ void calc_force_kernel(const int base,
				  const int n_ientry, const ientry_t* __restrict__ ientry,
				  const int* __restrict__ tile_indj,
				  const tile_excl_t<tilesize>* __restrict__ tile_excl,
				  const int stride,
				  const float* __restrict__ vdwparam, const int nvdwparam,
				  const float4* __restrict__ xyzq, const int* __restrict__ vdwtype,
				  AT* __restrict__ force) {

  // Pre-computed constants
  const int num_excl = ((tilesize*tilesize-1)/32 + 1);
  const int num_thread_per_excl = (32/num_excl);

  //
  // Shared data, common for the entire block
  //
  extern __shared__ char shmem[];
  
  //const unsigned int sh_start = tilesize*threadIdx.y;

  // Warp index (0...warpsize-1)
  const int wid = threadIdx.x % warpsize;

  // Load index (0...15 or 0...31)
  const int lid = (tilesize == 16) ? (wid % tilesize) : wid;

  int shmem_pos = 0;
  //
  // Shared memory requirements:
  // sh_xi, sh_yi, sh_zi, sh_qi: (blockDim.x/warpsize)*tilesize*sizeof(float)
  // sh_vdwtypei               : (blockDim.x/warpsize)*tilesize*sizeof(int)
  // sh_fix, sh_fiy, sh_fiz    : (blockDim.x/warpsize)*warpsize*sizeof(AT)
  // sh_vdwparam               : nvdwparam*sizeof(float)
  //
  // (x_i, y_i, z_i, q_i, vdwtype_i) are private to each warp
  // (fix, fiy, fiz) are private for each warp
  // vdwparam_sh is for the entire thread block
#if __CUDA_ARCH__ < 300
  float *sh_xi = (float *)&shmem[shmem_pos + (threadIdx.x/warpsize)*tilesize*sizeof(float)];
  shmem_pos += (blockDim.x/warpsize)*tilesize*sizeof(float);
  float *sh_yi = (float *)&shmem[shmem_pos + (threadIdx.x/warpsize)*tilesize*sizeof(float)];
  shmem_pos += (blockDim.x/warpsize)*tilesize*sizeof(float);
  float *sh_zi = (float *)&shmem[shmem_pos + (threadIdx.x/warpsize)*tilesize*sizeof(float)];
  shmem_pos += (blockDim.x/warpsize)*tilesize*sizeof(float);
  float *sh_qi = (float *)&shmem[shmem_pos + (threadIdx.x/warpsize)*tilesize*sizeof(float)];
  shmem_pos += (blockDim.x/warpsize)*tilesize*sizeof(float);
  int *sh_vdwtypei = (int *)&shmem[shmem_pos + (threadIdx.x/warpsize)*tilesize*sizeof(int)];
  shmem_pos += (blockDim.x/warpsize)*tilesize*sizeof(int);
#endif

  volatile AT *sh_fix = (AT *)&shmem[shmem_pos + (threadIdx.x/warpsize)*warpsize*sizeof(AT)];
  shmem_pos += (blockDim.x/warpsize)*warpsize*sizeof(AT);
  volatile AT *sh_fiy = (AT *)&shmem[shmem_pos + (threadIdx.x/warpsize)*warpsize*sizeof(AT)];
  shmem_pos += (blockDim.x/warpsize)*warpsize*sizeof(AT);
  volatile AT *sh_fiz = (AT *)&shmem[shmem_pos + (threadIdx.x/warpsize)*warpsize*sizeof(AT)];
  shmem_pos += (blockDim.x/warpsize)*warpsize*sizeof(AT);

  float *sh_vdwparam;
  if (!tex_vdwparam) {
    sh_vdwparam = (float *)&shmem[shmem_pos];
    shmem_pos += nvdwparam*sizeof(float);
  }

  // Load ientry. Single warp takes care of one ientry
  const int ientry_ind = (threadIdx.x + blockDim.x*blockIdx.x)/warpsize + base;

  int indi, ish, startj, endj;
  if (ientry_ind < n_ientry) {
    indi   = ientry[ientry_ind].indi;
    ish    = ientry[ientry_ind].ish;
    startj = ientry[ientry_ind].startj;
    endj   = ientry[ientry_ind].endj;
  } else {
    indi = 0;
    ish  = 1;
    startj = 1;
    endj = 0;
  }

  // Calculate shift for i-atom
  // ish = 0...26
  int ish_tmp = ish;
  float shz = (ish_tmp/9 - 1)*d_setup.boxz;
  ish_tmp -= (ish_tmp/9)*9;
  float shy = (ish_tmp/3 - 1)*d_setup.boxy;
  ish_tmp -= (ish_tmp/3)*3;
  float shx = (ish_tmp - 1)*d_setup.boxx;

  // Load i-atom data to shared memory (and shift coordinates)
  float4 xyzq_tmp = xyzq[indi + lid];
#if __CUDA_ARCH__ >= 300
  float xi = xyzq_tmp.x + shx;
  float yi = xyzq_tmp.y + shy;
  float zi = xyzq_tmp.z + shz;
  float qi = xyzq_tmp.w*ccelec;
  int vdwtypei = vdwtype[indi + lid];
#else
  sh_xi[lid] = xyzq_tmp.x + shx;
  sh_yi[lid] = xyzq_tmp.y + shy;
  sh_zi[lid] = xyzq_tmp.z + shz;
  sh_qi[lid] = xyzq_tmp.w*ccelec;
  sh_vdwtypei[lid] = vdwtype[indi + lid];
#endif

  sh_fix[wid] = (AT)0;
  sh_fiy[wid] = (AT)0;
  sh_fiz[wid] = (AT)0;

  if (!tex_vdwparam) {
    // Copy vdwparam to shared memory
    for (int i=threadIdx.x;i < nvdwparam;i+=blockDim.x)
      sh_vdwparam[i] = vdwparam[i];
    __syncthreads();
  }

  double vdwpotl;
  double coulpotl;
  if (calc_energy) {
    vdwpotl = 0.0;
    coulpotl = 0.0;
  }

  for (int jtile=startj;jtile <= endj;jtile++) {

    // Load j-atom starting index and exclusion mask
    unsigned int excl;
    if (tilesize == 16) {
      // For 16x16 tile, the exclusion mask per is 8 bits per thread:
      // NUM_THREAD_PER_EXCL = 4
      excl = tile_excl[jtile].excl[wid/num_thread_per_excl] >> 
	((wid % num_thread_per_excl)*num_excl);
    } else {
      excl = tile_excl[jtile].excl[wid];
    }
    int indj = tile_indj[jtile];

    // Skip empty tile
    if (__all(~excl == 0)) continue;

    float4 xyzq_j = xyzq[indj + lid];
    int ja = vdwtype[indj + lid];

    // Clear j forces
    AT fjx = (AT)0;
    AT fjy = (AT)0;
    AT fjz = (AT)0;

    for (int t=0;t < num_excl;t++) {
      
      int ii;
      if (tilesize == 16) {
	ii = (wid + t*2 + (wid/tilesize)*(tilesize-1)) % tilesize;
      } else {
	ii = ((wid + t) % tilesize);
      }

#if __CUDA_ARCH__ >= 300
      float dx = __shfl(xi, ii) - xyzq_j.x;
      float dy = __shfl(yi, ii) - xyzq_j.y;
      float dz = __shfl(zi, ii) - xyzq_j.z;
#else
      float dx = sh_xi[ii] - xyzq_j.x;
      float dy = sh_yi[ii] - xyzq_j.y;
      float dz = sh_zi[ii] - xyzq_j.z;
#endif
	
      float r2 = dx*dx + dy*dy + dz*dz;

#if __CUDA_ARCH__ >= 300
      float qq = __shfl(qi, ii)*xyzq_j.w;
#else
      float qq = sh_qi[ii]*xyzq_j.w;
#endif

#if __CUDA_ARCH__ >= 300
      int ia = __shfl(vdwtypei, ii);
#else
      int ia = sh_vdwtypei[ii];
#endif

      if (!(excl & 1) && r2 < d_setup.roff2) {

	float rinv = rsqrtf(r2);
	float r = r2*rinv;
	
	float fij_elec = pair_elec_force<elec_model, calc_energy>(r2, r, rinv, qq, coulpotl);
	
	int aa = (ja > ia) ? ja : ia;      // aa = max(ja,ia)
	int ivdw = (aa*(aa-3) + 2*(ja + ia) - 2) >> 1;
	
	float c6, c12;
	if (tex_vdwparam) {
	  //c6 = __ldg(&vdwparam[ivdw]);
	  //c12 = __ldg(&vdwparam[ivdw+1]);
	  float2 c6c12 = tex1Dfetch(vdwparam_texref, ivdw);
	  c6  = c6c12.x;
	  c12 = c6c12.y;
	} else {
	  c6 = sh_vdwparam[ivdw];
	  c12 = sh_vdwparam[ivdw+1];
	}
	
	float rinv2 = rinv*rinv;
	float fij_vdw = pair_vdw_force<vdw_model, calc_energy>(r2, r, rinv, rinv2,
							       c6, c12, vdwpotl);
	
	float fij = (fij_vdw - fij_elec)*rinv*rinv;

	AT fxij;
	AT fyij;
	AT fzij;
	calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);
	
	fjx -= fxij;
	fjy -= fyij;
	fjz -= fzij;
	
	if (tilesize == 16) {
	  // We need to re-calculate ii because ii must be warp sized in order to
	  // prevent race condition
	  int tmp = (wid + t*2) % 16 + (wid/16)*31;
	  ii = tilesize*(threadIdx.x/warpsize)*2 + (tmp + (tmp/32)*16) % 32;
	}
	
	sh_fix[ii] += fxij;
	sh_fiy[ii] += fyij;
	sh_fiz[ii] += fzij;
      } // if (!(excl & 1) && r2 < d_setup.roff2)

      // Advance exclusion mask
      excl >>= 1;
    }

    // Dump register forces (fjx, fjy, fjz)
    write_force<AT>(fjx, fjy, fjz, indj + lid, stride, force);
  }

  // Dump shared memory force (fi)
  // NOTE: no __syncthreads() required here because sh_fix is "volatile"
  write_force<AT>(sh_fix[wid], sh_fiy[wid], sh_fiz[wid], indi + lid, stride, force);

  if (calc_virial) {
    // Virial is calculated from (sh_fix[], sh_fiy[], sh_fiz[])
    // Variable "ish" depends on warp => Reduce within warp

    // Convert into double
    volatile double *sh_sfix = (double *)sh_fix;
    volatile double *sh_sfiy = (double *)sh_fiy;
    volatile double *sh_sfiz = (double *)sh_fiz;

    sh_sfix[wid] = ((double)sh_fix[wid])*INV_FORCE_SCALE;
    sh_sfiy[wid] = ((double)sh_fiy[wid])*INV_FORCE_SCALE;
    sh_sfiz[wid] = ((double)sh_fiz[wid])*INV_FORCE_SCALE;

    for (int d=16;d >= 1;d/=2) {
      if (wid < d) {
	sh_sfix[wid] += sh_sfix[wid + d];
	sh_sfiy[wid] += sh_sfiy[wid + d];
	sh_sfiz[wid] += sh_sfiz[wid + d];
      }
    }
    if (wid == 0) {
      atomicAdd(&d_energy_virial.sforcex[ish], sh_sfix[0]);
      atomicAdd(&d_energy_virial.sforcey[ish], sh_sfiy[0]);
      atomicAdd(&d_energy_virial.sforcez[ish], sh_sfiz[0]);
    }
  }

  if (calc_energy) {
    // Reduce energies across the entire thread block
    // Shared memory required:
    // blockDim.x*sizeof(double)*2
    __syncthreads();
    double2* sh_pot = (double2 *)(shmem);
    sh_pot[threadIdx.x].x = vdwpotl;
    sh_pot[threadIdx.x].y = coulpotl;
    __syncthreads();
    for (int i=1;i < blockDim.x;i *= 2) {
      int pos = threadIdx.x + i;
      double vdwpot_val  = (pos < blockDim.x) ? sh_pot[pos].x : 0.0;
      double coulpot_val = (pos < blockDim.x) ? sh_pot[pos].y : 0.0;
      __syncthreads();
      sh_pot[threadIdx.x].x += vdwpot_val;
      sh_pot[threadIdx.x].y += coulpot_val;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicAdd(&d_energy_virial.energy_vdw,  sh_pot[0].x);
      atomicAdd(&d_energy_virial.energy_elec, sh_pot[0].y);
    }
  }

}
*/

//########################################################################################
//########################################################################################
//########################################################################################

//
// Class creator
//
template <typename AT, typename CT>
CudaPMEDirectForceBlock<AT, CT>::CudaPMEDirectForceBlock() {
}

//
// Class destructor
//
template <typename AT, typename CT>
CudaPMEDirectForceBlock<AT, CT>::~CudaPMEDirectForceBlock() {
}

#define CREATE_KERNELS(KERNEL_NAME, KERNEL_PARAM)	\
  {							\
    KERNEL_NAME <AT, CT, VDW_VSH, EWALD, true, true, true>	\
      <<< nblock, nthread, shmem_size, stream >>>		\
      (KERNEL_PARAM);						\
  }

//
// Calculates 1-4 exclusions and interactions
//
template <typename AT, typename CT>
void CudaPMEDirectForceBlock<AT, CT>::calc_14_force(const float4 *xyzq,
					const bool calc_energy, const bool calc_virial,
					const int stride, AT *force, cudaStream_t stream) {

  if (!vdwparam_texref_bound) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_14_force, vdwparam14_texref must be bound" << std::endl;
    exit(1);
  }

  /*
  int nthread = 512;
  //int nblock = (nin14list + nex14list - 1)/nthread + 1;
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

  /*
  CREATE_KERNELS("calc_14_force_kernel", "this->nin14list, this->nex14list, this->nin14block, this->in14list, this->ex14list,
	     this->vdwtype, this->vdwparam14, xyzq, stride, force");

  /*
  if (vdw_model_loc == VDW_VSH) {
    if (elec_model_loc == EWALD) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSH, EWALD, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (this->nin14list, this->nex14list, this->nin14block, this->in14list, this->ex14list,
	     this->vdwtype, this->vdwparam14, xyzq, stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSH, EWALD, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSH, EWALD, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSH, EWALD, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else if (elec_model_loc == EWALD_LOOKUP) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSH, EWALD_LOOKUP, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSH, EWALD_LOOKUP, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSH, EWALD_LOOKUP, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSH, EWALD_LOOKUP, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else if (elec_model_loc == NONE) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSH, NONE, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSH, NONE, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSH, NONE, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSH, NONE, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else {
      std::cout<<"CudaPMEDirectForceBlock<AT, CT>::calc_14_force, Invalid EWALD model "
	       <<elec_model_loc<<std::endl;
      exit(1);
    }
  } else if (vdw_model_loc == VDW_VSW) {
    if (elec_model_loc == EWALD) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSW, EWALD, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSW, EWALD, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSW, EWALD, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSW, EWALD, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else if (elec_model_loc == EWALD_LOOKUP) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSW, EWALD_LOOKUP, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSW, EWALD_LOOKUP, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSW, EWALD_LOOKUP, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSW, EWALD_LOOKUP, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else if (elec_model_loc == NONE) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSW, NONE, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSW, NONE, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSW, NONE, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VSW, NONE, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else {
      std::cout<<"CudaPMEDirectForceBlock<AT, CT>::calc_14_force, Invalid EWALD model "
	       <<elec_model_loc<<std::endl;
      exit(1);
    }
  } else if (vdw_model_loc == VDW_VFSW) {
    if (elec_model_loc == EWALD) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, EWALD, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, EWALD, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, EWALD, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, EWALD, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else if (elec_model_loc == EWALD_LOOKUP) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, EWALD_LOOKUP, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, EWALD_LOOKUP, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, EWALD_LOOKUP, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, EWALD_LOOKUP, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else if (elec_model_loc == NONE) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, NONE, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, NONE, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, NONE, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_VFSW, NONE, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else {
      std::cout<<"CudaPMEDirectForceBlock<AT, CT>::calc_14_force, Invalid EWALD model "
	       <<elec_model_loc<<std::endl;
      exit(1);
    }
  } else if (vdw_model_loc == VDW_CUT) {
    if (elec_model_loc == EWALD) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_CUT, EWALD, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_CUT, EWALD, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_CUT, EWALD, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_CUT, EWALD, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else if (elec_model_loc == EWALD_LOOKUP) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_CUT, EWALD_LOOKUP, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_CUT, EWALD_LOOKUP, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_CUT, EWALD_LOOKUP, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_CUT, EWALD_LOOKUP, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else if (elec_model_loc == NONE) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_CUT, NONE, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_CUT, NONE, true, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      } else {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_CUT, NONE, false, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	} else {
	  calc_14_force_kernel <AT, CT, VDW_CUT, NONE, false, false, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
	}
      }
    } else {
      std::cout<<"CudaPMEDirectForceBlock<AT, CT>::calc_14_force, Invalid EWALD model "
	       <<elec_model_loc<<std::endl;
      exit(1);
    }
  } else {
    std::cout<<"CudaPMEDirectForceBlock<AT, CT>::calc_14_force, Invalid VDW model"<<std::endl;
    exit(1);
  }
  */

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

  if (!vdwparam_texref_bound) {
    std::cerr << "CudaPMEDirectForceBlock<AT, CT>::calc_force, vdwparam_texref must be bound" << std::endl;
    exit(1);
  }

  /*
  if (nlist->n_ientry == 0) return;
  int vdw_model_loc = calc_vdw ? vdw_model : NONE;
  int elec_model_loc = calc_elec ? elec_model : NONE;
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
  // (sh_xi, sh_yi, sh_zi, sh_qi, sh_vdwtypei)
  if (get_cuda_arch() < 300)
    shmem_size += (nthread/warpsize)*tilesize*(sizeof(float)*4 + sizeof(int));
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

    if (vdw_model_loc == VDW_VSH) {
      if (elec_model_loc == EWALD) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else if (elec_model_loc == EWALD_LOOKUP) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD_LOOKUP, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD_LOOKUP, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD_LOOKUP, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD_LOOKUP, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else if (elec_model_loc == NONE) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, NONE, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, NONE, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, NONE, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, NONE, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else {
	std::cout<<"CudaPMEDirectForceBlock<AT, CT>::calc_force, Invalid EWALD model "<<elec_model_loc<<std::endl;
	exit(1);
      }
    } else if (vdw_model_loc == VDW_VSW) {
      if (elec_model_loc == EWALD) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else if (elec_model_loc == EWALD_LOOKUP) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD_LOOKUP, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD_LOOKUP, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD_LOOKUP, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD_LOOKUP, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else if (elec_model_loc == NONE) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, NONE, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, NONE, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, NONE, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, NONE, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else {
	std::cout<<"CudaPMEDirectForceBlock<AT, CT>::calc_force, Invalid EWALD model "<<elec_model_loc<<std::endl;
	exit(1);
      }
    } else if (vdw_model_loc == VDW_VFSW) {
      if (elec_model_loc == EWALD) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else if (elec_model_loc == EWALD_LOOKUP) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD_LOOKUP, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD_LOOKUP, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD_LOOKUP, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD_LOOKUP, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else if (elec_model_loc == NONE) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, NONE, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, NONE, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, NONE, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, NONE, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else {
	std::cout<<"CudaPMEDirectForceBlock<AT, CT>::calc_force, Invalid EWALD model "<<elec_model_loc<<std::endl;
	exit(1);
      }
    } else if (vdw_model_loc == VDW_CUT) {
      if (elec_model_loc == EWALD) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else if (elec_model_loc == EWALD_LOOKUP) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD_LOOKUP, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD_LOOKUP, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD_LOOKUP, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD_LOOKUP, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else if (elec_model_loc == NONE) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, NONE, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, NONE, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, NONE, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, NONE, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base, nlist->n_ientry, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else {
	std::cout<<"CudaPMEDirectForceBlock<AT, CT>::calc_force, Invalid EWALD model "
		 <<elec_model_loc<<std::endl;
	exit(1);
      }
    } else {
      std::cout<<"CudaPMEDirectForceBlock<AT, CT>::calc_force, Invalid VDW model"<<std::endl;
      exit(1);
    }

    base += (nthread/warpsize)*nblock;

    cudaCheck(cudaGetLastError());
  }
  */

}

//
// Explicit instances of CudaPMEDirectForceBlock
//
template class CudaPMEDirectForceBlock<long long int, float>;
