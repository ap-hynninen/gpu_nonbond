#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "NeighborList.h"
#include "DirectForce.h"

static __constant__ const float ccelec = 332.0716;

template <typename AT, typename CT>
__forceinline__ __device__ void write_force(const CT fx, const CT fy, const CT fz,
					    const int ind, const int stride,
					    AT* force) {
  // The generic version can not be used
}

// Template specialization for 64bit integer = "long long int"
template <>
__forceinline__ __device__ void write_force <long long int, float> (const float fx, const float fy, const float fz,
								    const int ind, const int stride,
								    long long int* force) {

  atomicAdd((unsigned long long int *)&force[ind           ], llitoulli(fx));
  atomicAdd((unsigned long long int *)&force[ind + stride  ], llitoulli(fy));
  atomicAdd((unsigned long long int *)&force[ind + stride*2], llitoulli(fz));
}

/*
template <>
__forceinline__ __device__ void write_force <double, float>() {
    // Reduce forces and then do atomicAdd from a single thread
    // Write to shared memory
    fj_tmp[tid].x = fjx;
    fj_tmp[tid].y = fjy;
    fj_tmp[tid].z = fjz;
    if (threadIdx.x == 0) {
      FORCE_T f_red[3] = {0.0f, 0.0f, 0.0f};
      for (int i=sh_start;i < sh_start + threadIdx.x;i++) {
	f_red[0] += fj_tmp[i].x;
	f_red[1] += fj_tmp[i].y;
	f_red[2] += fj_tmp[i].z;
      }
      atomicAdd(&force[blockIdx.x*stride3 +           indj + threadIdx.x], f_red[0]);
      atomicAdd(&force[blockIdx.x*stride3 + stride  + indj + threadIdx.x], f_red[1]);
      atomicAdd(&force[blockIdx.x*stride3 + stride2 + indj + threadIdx.x], f_red[2]);
    }
    //    force[blockIdx.x*stride3 +           indj + threadIdx.x] += fjx;
    //    force[blockIdx.x*stride3 + stride +  indj + threadIdx.x] += fjy;
    //    force[blockIdx.x*stride3 + stride2 + indj + threadIdx.x] += fjz;
}
*/

template <typename AT, typename CT>
__forceinline__ __device__
void calc_component_force(CT fij,
			  const CT dx, const CT dy, const CT dz,
			  AT &fxij, AT &fyij, AT &fzij) {
  fxij = (AT)(fij*dx);
  fyij = (AT)(fij*dy);
  fzij = (AT)(fij*dz);
}

template <>
__forceinline__ __device__
void calc_component_force<long long int, float>(float fij,
						const float dx, const float dy, const float dz,
						long long int &fxij, long long int &fyij, long long int &fzij) {
  fij *= FORCE_SCALE;
  fxij = lliroundf(fij*dx);
  fyij = lliroundf(fij*dy);
  fzij = lliroundf(fij*dz);
}

class vdw_base {
public:
  virtual void setup(float ron2, float roff2) = 0;
  //virtual float pair_force() = 0;
};

class vdw_vsh : public vdw_base {
public:
  float roffinv6;
  float roffinv12;
  float roffinv18;

  void setup(float ron2, float roff2) {
    roffinv6 = 1.0f/(roff2*roff2*roff2);
    roffinv12 = roffinv6*roffinv6;
    roffinv18 = roffinv12*roffinv6;
  }
};


//#define WARPSIZE 32                             // Number of threads per warp
//#define TILESIZE 32                             // Number of atoms per tile direction
//#define NUM_EXCL ((32*32-1)/32 + 1) // Number of exclusion mask integers
//#define NUM_THREAD_PER_EXCL (32/NUM_EXCL)       // Number of threads per exclusion mask integer

// Settings for direct computation in device memory
//static DirectSettings_t h_setup;
static __constant__ DirectSettings_t d_setup;

// Energy and virial in device memory
static __device__ DirectEnergyVirial_t d_energy_virial;

// VdW parameter texture reference
static texture<float2, 1, cudaReadModeElementType> vdwparam_texref;
static bool vdwparam_texref_bound = false;

//
// Calculates VdW pair force & energy
//
template <int vdw_model, bool calc_energy>
__forceinline__ __device__
float pair_vdw_force(float r2, float r, float rinv, float rinv2, float c6, float c12,
		     double &vdwpotl) {

  float fij_vdw;

  if (vdw_model == VDW_VSH) {
    float r6 = r2*r2*r2;
    float rinv6 = rinv2*rinv2*rinv2;
    float rinv12 = rinv6*rinv6;
	    
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      vdwpotl += (double)(c12*one_twelve*(rinv12 + 2.0f*r6*d_setup.roffinv18 - 
					  3.0f*d_setup.roffinv12)-
			  c6*one_six*(rinv6 + r6*d_setup.roffinv12 - 2.0f*d_setup.roffinv6));
    }
	  
    fij_vdw = c6*(rinv6 - r6*d_setup.roffinv12) - c12*(rinv12 + r6*d_setup.roffinv18);
  } else if (vdw_model == VDW_VSW) {
    float roff2_r2_sq = d_setup.roff2 - r2;
    roff2_r2_sq *= roff2_r2_sq;
    float sw = (r2 <= d_setup.ron2) ? 1.0f : 
      roff2_r2_sq*(d_setup.roff2 + 2.0f*r2 - 3.0f*d_setup.ron2)*d_setup.inv_roff2_ron2;
    // dsw_6 = dsw/6.0
    float dsw_6 = (r2 <= d_setup.ron2) ? 0.0f : 
      (d_setup.roff2-r2)*(d_setup.ron2-r2)*d_setup.inv_roff2_ron2;
    float rinv4 = rinv2*rinv2;
    float rinv6 = rinv4*rinv2;
    fij_vdw = rinv4*( c12*rinv6*(dsw_6 - sw*rinv2) - c6*(2.0f*dsw_6 - sw*rinv2) );
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      vdwpotl += (double)( sw*rinv6*(one_twelve*c12*rinv6 - one_six*c6) );
    }
  } else if (vdw_model == VDW_CUT) {
    float rinv6 = rinv2*rinv2*rinv2;
	  
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      float rinv12 = rinv6*rinv6;
      vdwpotl += (double)(c12*one_twelve*rinv12 - c6*one_six*rinv6);
      fij_vdw = c6*rinv6 - c12*rinv12;
    } else {
      fij_vdw = c6*rinv6 - c12*rinv6*rinv6;
    }
  } else if (vdw_model == VDW_VFSW) {
    float rinv3 = rinv*rinv2;
    float rinv6 = rinv3*rinv3;
    float A6 = (r2 > d_setup.ron2) ? d_setup.k6 : 1.0f;
    float B6 = (r2 > d_setup.ron2) ? d_setup.roffinv3  : 0.0f;
    float A12 = (r2 > d_setup.ron2) ? d_setup.k12 : 1.0f;
    float B12 = (r2 > d_setup.ron2) ? d_setup.roffinv6 : 0.0f;
    fij_vdw = c6*A6*(rinv3 - B6)*rinv3 - c12*A12*(rinv6 - B12)*rinv6;
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      float C6  = (r2 > d_setup.ron2) ? 0.0f : d_setup.dv6;
      float C12 = (r2 > d_setup.ron2) ? 0.0f : d_setup.dv12;

      float rinv3_B6_sq = rinv3 - B6;
      rinv3_B6_sq *= rinv3_B6_sq;

      float rinv6_B12_sq = rinv6 - B12;
      rinv6_B12_sq *= rinv6_B12_sq;

      vdwpotl += (double)(one_twelve*c12*(A12*rinv6_B12_sq + C12) -
			  one_six*c6*(A6*rinv3_B6_sq + C6));
    }
  } else if (vdw_model == NONE) {
    fij_vdw = 0.0f;
  }

  return fij_vdw;
}

//static texture<float, 1, cudaReadModeElementType> ewald_force_texref;

//
// Returns simple linear interpolation
// NOTE: Could the interpolation be done implicitly using the texture unit?
//
__forceinline__ __device__ float lookup_force(float r, float hinv) {
  float r_hinv = r*hinv;
  int ind = (int)r_hinv;
  float f1 = r_hinv - (float)ind;
  float f2 = 1.0f - f1;
#if __CUDA_ARCH__ < 350
  return f1*d_setup.ewald_force[ind] + f2*d_setup.ewald_force[ind+1];
#else
  return f1*__ldg(&d_setup.ewald_force[ind]) + f2*__ldg(&d_setup.ewald_force[ind+1]);
#endif
  //return f1*tex1Dfetch(ewald_force_texref, ind) + f2*tex1Dfetch(ewald_force_texref, ind+1);
}

//
// Calculates electrostatic force & energy
//
template <int elec_model, bool calc_energy>
__forceinline__ __device__
float pair_elec_force(float r2, float r, float rinv, float qi, float qj, double &coulpotl) {

  float fij_elec;

  float qq = qi*qj;

  if (elec_model == EWALD_LOOKUP) {
    fij_elec = qi*qj*lookup_force(r, d_setup.hinv);
  } else if (elec_model == EWALD) {
    float erfc_val = fasterfc(d_setup.kappa*r);
    float exp_val = expf(-d_setup.kappa2*r2);
    if (calc_energy) {
      coulpotl += (double)(qq*erfc_val*rinv);
    }
    const float two_sqrtpi = 1.12837916709551f;    // 2/sqrt(pi)
    fij_elec = qq*(two_sqrtpi*d_setup.kappa*exp_val + erfc_val*rinv);
  } else if (elec_model == NONE) {
    fij_elec = 0.0f;
  }

  return fij_elec;
}

//
// Nonbonded force kernel
//
template <typename AT, typename CT, int tilesize, int vdw_model, int elec_model,
	  bool calc_energy, bool calc_virial, bool tex_vdwparam>
__global__ void calc_force_kernel(const unsigned int base_tid,
				  const int ni, const ientry_t* __restrict__ ientry,
				  const int* __restrict__ tile_indj,
				  const tile_excl_t<tilesize>* __restrict__ tile_excl,
				  const int stride,
				  const float* __restrict__ vdwparam, const int nvdwparam,
				  const float4* __restrict__ xyzq, const int* __restrict__ vdwtype,
				  AT *force) {

  // Pre-computed constants
  const int num_excl = ((tilesize*tilesize-1)/32 + 1);
  const int num_thread_per_excl = (32/num_excl);

  //
  // Shared data, common for the entire block
  //
  extern __shared__ char shmem[];
  
  volatile float *x_i = (float *)&shmem[0];                        // tilesize*blockDim.y
  volatile float *y_i = (float *)&x_i[tilesize*blockDim.y];        // tilesize*blockDim.y
  volatile float *z_i = (float *)&y_i[tilesize*blockDim.y];        // tilesize*blockDim.y
  volatile float *q_i = (float *)&z_i[tilesize*blockDim.y];        // tilesize*blockDim.y
  volatile int *vdwtype_i = (int *)&q_i[tilesize*blockDim.y];      // tilesize*blockDim.y
  volatile AT *fix = (AT *)&vdwtype_i[tilesize*blockDim.y];        // WARPSIZE*blockDim.y
  volatile AT *fiy = &fix[warpsize*blockDim.y];                    // WARPSIZE*blockDim.y
  volatile AT *fiz = &fiy[warpsize*blockDim.y];                    // WARPSIZE*blockDim.y
  volatile float *vdwparam_sh;
  
  if (!tex_vdwparam) {
    vdwparam_sh = (float *)&fiz[warpsize*blockDim.y];
  }

  /*
#ifdef PREC_SPDP
  __shared__ FORCE3_T fj_tmp[WARPSIZE*TILEX_NBLOCK];
#endif
#ifndef TEX_FETCH_VDWPARAM
  __shared__ float vdwparam_sh[MAX_NVDWPARAM];
#endif
  */

  // Load ientry
  const unsigned int ientry_ind = threadIdx.y + blockDim.y*blockIdx.x + base_tid;

  int indi, ish, startj, endj;
  if (ientry_ind < ni) {
    indi   = ientry[ientry_ind].indi;
    ish    = ientry[ientry_ind].ish;
    startj = ientry[ientry_ind].startj;
    endj   = ientry[ientry_ind].endj;
  } else {
    indi = 0;
    ish  = 0;
    startj = 1;
    endj = 0;
  }

  // Calculate shift for i-atom
  // ish = 1...27
  int ish_tmp = ish;
  float shz = (ish_tmp/9 - 1)*d_setup.boxz;
  ish_tmp -= (ish_tmp/9)*9;
  float shy = (ish_tmp/3 - 1)*d_setup.boxy;
  ish_tmp -= (ish_tmp/3)*3;
  float shx = (ish_tmp - 1)*d_setup.boxx;

  const unsigned int sh_start = tilesize*threadIdx.y;
  // tid:
  // threadIdx.y=0: 0...31
  // threadIdx.y=1: 32...63
  const unsigned int tid = threadIdx.x + blockDim.x*threadIdx.y;

  unsigned int load_ij;
  if (tilesize == 16) {
    load_ij = threadIdx.x % tilesize;
  } else {
    load_ij = threadIdx.x;
  }

  // Load i-atom data to shared memory (and shift coordinates)
  float4 xyzq_tmp = xyzq[indi + load_ij];
  x_i[sh_start + load_ij] = xyzq_tmp.x + shx;
  y_i[sh_start + load_ij] = xyzq_tmp.y + shy;
  z_i[sh_start + load_ij] = xyzq_tmp.z + shz;
  q_i[sh_start + load_ij] = xyzq_tmp.w*ccelec;

  vdwtype_i[sh_start + load_ij] = vdwtype[indi + load_ij];

  fix[tid] = (AT)0;
  fiy[tid] = (AT)0;
  fiz[tid] = (AT)0;

  if (!tex_vdwparam) {
    // Copy vdwparam to shared memory
    if (tid < nvdwparam)
      vdwparam_sh[tid] = vdwparam[tid];
  }

  __syncthreads();

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
      excl = tile_excl[jtile].excl[threadIdx.x/num_thread_per_excl] >> 
	((threadIdx.x % num_thread_per_excl)*num_excl);
    } else {
      excl = tile_excl[jtile].excl[load_ij];
    }
    int indj = tile_indj[jtile];

    // Skip empty tile
    if (__all(~excl == 0)) continue;

    float4 xyzq_j = xyzq[indj + load_ij];
    int ja = vdwtype[indj + load_ij];

    // Clear j forces
    AT fjx = (AT)0;
    AT fjy = (AT)0;
    AT fjz = (AT)0;

    for (int t=0;t < num_excl;t++) {
      
      unsigned int excl_bit = !(excl & 1);

      if (excl_bit) {
	
	int ii;
	if (tilesize == 16) {
	  ii = sh_start + (threadIdx.x + t*2 + (threadIdx.x/tilesize)*(tilesize-1)) % tilesize;
	} else {
	  ii = sh_start + ((threadIdx.x + t) % tilesize);
	}
	
	float dx = x_i[ii] - xyzq_j.x;
	float dy = y_i[ii] - xyzq_j.y;
	float dz = z_i[ii] - xyzq_j.z;
	
	float r2 = dx*dx + dy*dy + dz*dz;

	if (r2 < d_setup.roff2) {

	  float rinv = rsqrtf(r2);
	  float rinv2 = rinv*rinv;
	  float r = r2*rinv;

	  float fij_elec = pair_elec_force<elec_model, calc_energy>(r2, r, rinv,
								    q_i[ii], xyzq_j.w, coulpotl);

	  int ia = vdwtype_i[ii];
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
	    c6 = vdwparam_sh[ivdw];
	    c12 = vdwparam_sh[ivdw+1];
	  }

	  float fij_vdw = pair_vdw_force<vdw_model, calc_energy>(r2, r, rinv, rinv2,
								 c6, c12, vdwpotl);

	  float fij = (fij_vdw - fij_elec)*rinv2;

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
	    int tmp = (threadIdx.x + t*2) % 16 + (threadIdx.x/16)*31;
	    ii = sh_start*2 + (tmp + (tmp/32)*16) % 32;
	  }

	  fix[ii] += fxij;
	  fiy[ii] += fyij;
	  fiz[ii] += fzij;
	} // if (r2 < d_setup.roff2)
      }

      // Advance exclusion mask
      excl >>= 1;
    }

    // Dump register forces (fjx, fjy, fjz)
    write_force<AT, CT>(fjx, fjy, fjz, indj+load_ij, stride, force);
  }

  // Dump shared memory force (fi)
  //__syncthreads();         // <-- Is this really needed?
  write_force<AT, CT>(fix[tid], fiy[tid], fiz[tid], indi+load_ij, stride, force);

  if (calc_virial) {
    // Value of ish depends on threadIdx.y => Reduce within warp
    __syncthreads();
    double *shmem_p = ((double *)x_i);
    volatile double *sforcex = &shmem_p[blockDim.x*threadIdx.y];
    volatile double *sforcey = &shmem_p[blockDim.x*(blockDim.y + threadIdx.y)];
    volatile double *sforcez = &shmem_p[blockDim.x*(blockDim.y*2 + threadIdx.y)];

    if (threadIdx.x < 16) {
      sforcex[threadIdx.x] += sforcex[threadIdx.x + 16];
      sforcey[threadIdx.x] += sforcey[threadIdx.x + 16];
      sforcez[threadIdx.x] += sforcez[threadIdx.x + 16];
    }

    if (threadIdx.x < 8) {
      sforcex[threadIdx.x] += sforcex[threadIdx.x + 8];
      sforcey[threadIdx.x] += sforcey[threadIdx.x + 8];
      sforcez[threadIdx.x] += sforcez[threadIdx.x + 8];
    }

    if (threadIdx.x < 4) {
      sforcex[threadIdx.x] += sforcex[threadIdx.x + 4];
      sforcey[threadIdx.x] += sforcey[threadIdx.x + 4];
      sforcez[threadIdx.x] += sforcez[threadIdx.x + 4];
    }

    if (threadIdx.x < 2) {
      sforcex[threadIdx.x] += sforcex[threadIdx.x + 2];
      sforcey[threadIdx.x] += sforcey[threadIdx.x + 2];
      sforcez[threadIdx.x] += sforcez[threadIdx.x + 2];
    }

    if (threadIdx.x < 1) {
      sforcex[threadIdx.x] += sforcex[threadIdx.x + 1];
      sforcey[threadIdx.x] += sforcey[threadIdx.x + 1];
      sforcez[threadIdx.x] += sforcez[threadIdx.x + 1];

      atomicAdd(&d_energy_virial.sforcex[ish-1], sforcex[0]);
      atomicAdd(&d_energy_virial.sforcey[ish-1], sforcey[0]);
      atomicAdd(&d_energy_virial.sforcez[ish-1], sforcez[0]);
    }

  }

  if (calc_energy) {
    // Reduce energies to
    // Reduces within thread block, uses the "xyzq_i" shared memory buffer
    __syncthreads();          // NOTE: this makes sure we can write to x_i 
    double2 *potbuf = (double2 *)(x_i);
    potbuf[tid].x = vdwpotl;
    potbuf[tid].y = coulpotl;
    // sync to make sure all threads in block have finished writing share memory
    __syncthreads();
    const int nthreadblock = blockDim.x*blockDim.y;
    for (int i=1;i < nthreadblock;i *= 2) {
      int pos = tid + i;
      double vdwpot_val  = (pos < nthreadblock) ? potbuf[pos].x : 0.0;
      double coulpot_val = (pos < nthreadblock) ? potbuf[pos].y : 0.0;
      __syncthreads();
      potbuf[tid].x += vdwpot_val;
      potbuf[tid].y += coulpot_val;
      __syncthreads();
    }
    if (tid == 0) {
      //      atomicAdd((double *)&force[stride*3],   potbuf[0].x);
      //      atomicAdd((double *)&force[stride*3+1], potbuf[0].y);
      atomicAdd(&d_energy_virial.energy_vdw, potbuf[0].x);
      atomicAdd(&d_energy_virial.energy_elec, potbuf[0].y);
    }

  }

}

//
// Nonbonded force kernel
//
template <typename AT, typename CT, int tilesize, int vdw_model, int elec_model,
	  bool calc_energy>
__global__ 
void calc_force_kernel_sparse(const int ni, const ientry_t* __restrict__ ientry,
			      const int* __restrict__ tile_indj,
			      const pairs_t<tilesize>* __restrict__ pairs,
			      const int stride,
			      const float* __restrict__ vdwparam, const int nvdwparam,
			      const float4* __restrict__ xyzq, const int* __restrict__ vdwtype,
			      AT *force) {

  //
  // Shared data, common for the entire block
  //
  extern __shared__ char shmem[];

  volatile float *x_i = (float *)&shmem[0];                        // tilesize*blockDim.y
  volatile float *y_i = (float *)&x_i[tilesize*blockDim.y];        // tilesize*blockDim.y
  volatile float *z_i = (float *)&y_i[tilesize*blockDim.y];        // tilesize*blockDim.y
  volatile float *q_i = (float *)&z_i[tilesize*blockDim.y];        // tilesize*blockDim.y
  volatile int *vdwtype_i = (int *)&q_i[tilesize*blockDim.y];      // tilesize*blockDim.y
  volatile AT *sh_force = (AT *)&vdwtype_i[tilesize*blockDim.y];   // blockDim.x*blockDim.y*3
  
  // Load ijentry
  const unsigned int ientry_ind = threadIdx.y + blockDim.y*blockIdx.x;

  int indi, ish, startj, endj;
  if (ientry_ind < ni) {
    indi   = ientry[ientry_ind].indi;
    ish    = ientry[ientry_ind].ish;
    startj = ientry[ientry_ind].startj;
    endj   = ientry[ientry_ind].endj;
  } else {
    indi = 0;
    ish  = 0;
    startj = 1;
    endj = 0;
  }

  // Calculate shift for i-atom
  float shz = (ish/9 - 1)*d_setup.boxz;
  ish -= (ish/9)*9;
  float shy = (ish/3 - 1)*d_setup.boxy;
  ish -= (ish/3)*3;
  float shx = (ish - 1)*d_setup.boxx;

  const unsigned int sh_start = tilesize*threadIdx.y;

  unsigned int load_ij;
  if (tilesize == 16) {
    load_ij = threadIdx.x % tilesize;
  } else {
    load_ij = threadIdx.x;
  }

  // Load i-atom data to shared memory (and shift coordinates)
  float4 xyzq_tmp = xyzq[indi + load_ij];
  x_i[sh_start + load_ij] = xyzq_tmp.x + shx;
  y_i[sh_start + load_ij] = xyzq_tmp.y + shy;
  z_i[sh_start + load_ij] = xyzq_tmp.z + shz;
  q_i[sh_start + load_ij] = xyzq_tmp.w*ccelec;

  vdwtype_i[sh_start + load_ij] = vdwtype[indi + load_ij];

  const unsigned int shi = threadIdx.x + blockDim.x*3*threadIdx.y;
  sh_force[shi]                = (AT)0;
  sh_force[shi + blockDim.x]   = (AT)0;
  sh_force[shi + blockDim.x*2] = (AT)0;

  double vdwpotl;
  double coulpotl;
  if (calc_energy) {
    vdwpotl = 0.0;
    coulpotl = 0.0;
  }

  for (int jtile=startj;jtile <= endj;jtile++) {

    // Load j-atom data
    int indj = tile_indj[jtile];
    float4 xyzq_j = xyzq[indj + load_ij];
    int ja = vdwtype[indj + load_ij];

    // This thread calculates the interaction between i and j=load_ij
    int i = pairs[jtile].i[load_ij];

    int ii;
    ii = sh_start + i;
	
    float dx = x_i[ii] - xyzq_j.x;
    float dy = y_i[ii] - xyzq_j.y;
    float dz = z_i[ii] - xyzq_j.z;
	
    float r2 = dx*dx + dy*dy + dz*dz;

    if (r2 < d_setup.roff2) {

      float rinv = rsqrtf(r2);
      float rinv2 = rinv*rinv;
      float r = r2*rinv;
      
      float fij_elec = pair_elec_force<elec_model, calc_energy>(r2, r, rinv,
								q_i[ii], xyzq_j.w, coulpotl);

      int ia = vdwtype_i[ii];
      int aa = (ja > ia) ? ja : ia;      // aa = max(ja,ia)
      int ivdw = (aa*(aa-3) + 2*(ja + ia) - 2) >> 1;

      float c6, c12;
      float2 c6c12 = tex1Dfetch(vdwparam_texref, ivdw);
      c6  = c6c12.x;
      c12 = c6c12.y;

      float fij_vdw = pair_vdw_force<vdw_model, calc_energy>(r2, r, rinv, rinv2, c6, c12, vdwpotl);
      
      float fij = (fij_vdw - fij_elec)*rinv2;

      AT fxij;
      AT fyij;
      AT fzij;
      calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);

      // Write j forces to global memory
      write_force<AT, CT>(-fxij, -fyij, -fzij, indj+load_ij, stride, force);

      // Write i forces to shared memory
      write_force<AT, CT>(fxij, fyij, fzij, blockDim.x*3*threadIdx.y + i, blockDim.x, sh_force);

    } // if (r2 < d_setup.roff2)

  }

  // Write i forces to global memory
  write_force<AT, CT>(sh_force[shi], sh_force[shi + blockDim.x], sh_force[shi + blockDim.x*2],
		      indi+load_ij, stride, force);

  /*
  if (calc_energy) {
    // Reduce energies to (pot)
    // Reduces within thread block, uses the "xyzq_i" shared memory buffer
    __syncthreads();          // NOTE: this makes sure we can write to xyzq_i 
    double2 *potbuf = (double2 *)(x_i);
    potbuf[tid].x = vdwpotl;
    potbuf[tid].y = coulpotl;
    // sync to make sure all threads in block are finished writing share memory
    __syncthreads();
    const int nthreadblock = blockDim.x*blockDim.y;
    for (int i=1;i < nthreadblock;i *= 2) {
      int pos = tid + i;
      double vdwpot_val  = (pos < nthreadblock) ? potbuf[pos].x : 0.0;
      double coulpot_val = (pos < nthreadblock) ? potbuf[pos].y : 0.0;
      __syncthreads();
      potbuf[tid].x += vdwpot_val;
      potbuf[tid].y += coulpot_val;
      __syncthreads();
    }
    if (tid == 0) {
      atomicAdd((double *)&force[stride*3],   potbuf[0].x);
      atomicAdd((double *)&force[stride*3+1], potbuf[0].y);
    }

  }
  */

}

//
// Class creator
//
template <typename AT, typename CT>
DirectForce<AT, CT>::DirectForce() {
  vdwparam = NULL;
  nvdwparam = 0;
  vdwparam_len = 0;
  use_tex_vdwparam = true;
  vdwparam_texref_bound = false;

  vdwtype = NULL;
  vdwtype_len = 0;

  ewald_force = NULL;
  n_ewald_force = 0;

  set_calc_vdw(true);
  set_calc_elec(true);

  allocate_host<DirectEnergyVirial_t>(&h_energy_virial, 1);
  allocate_host<DirectSettings_t>(&h_setup, 1);

  clear_energy_virial();
}

//
// Class destructor
//
template <typename AT, typename CT>
DirectForce<AT, CT>::~DirectForce() {
  if (vdwparam_texref_bound) {
    cudaCheck(cudaUnbindTexture(vdwparam_texref));
    vdwparam_texref_bound = false;
  }
  if (vdwparam != NULL) deallocate<CT>(&vdwparam);
  if (vdwtype != NULL) deallocate<int>(&vdwtype);
  if (ewald_force != NULL) deallocate<CT>(&ewald_force);
  if (h_energy_virial != NULL) deallocate_host<DirectEnergyVirial_t>(&h_energy_virial);
  if (h_setup != NULL) deallocate_host<DirectSettings_t>(&h_setup);
}

//
// Copies h_setup -> d_setup
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::update_setup() {
  cudaCheck(cudaMemcpyToSymbol(d_setup, h_setup, sizeof(DirectSettings_t)));
}

//
// Sets parameters for the nonbonded computation
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::setup(CT boxx, CT boxy, CT boxz, 
				CT kappa,
				CT roff, CT ron,
				int vdw_model, int elec_model,
				bool calc_vdw, bool calc_elec) {

  CT ron2 = ron*ron;
  CT ron3 = ron*ron*ron;
  CT ron6 = ron3*ron3;

  CT roff2 = roff*roff;
  CT roff3 = roff*roff*roff;
  CT roff6 = roff3*roff3;

  h_setup->boxx = boxx;
  h_setup->boxy = boxy;
  h_setup->boxz = boxz;
  h_setup->kappa = kappa;
  h_setup->kappa2 = kappa*kappa;
  h_setup->roff2 = roff2;
  h_setup->ron2 = ron2;

  h_setup->roffinv6 = ((CT)1.0)/(roff2*roff2*roff2);
  h_setup->roffinv12 = h_setup->roffinv6*h_setup->roffinv6;
  h_setup->roffinv18 = h_setup->roffinv12*h_setup->roffinv6;

  CT roff2_min_ron2 = roff2 - ron2;
  h_setup->inv_roff2_ron2 = ((CT)1.0)/(roff2_min_ron2*roff2_min_ron2*roff2_min_ron2);

  // Constants for VFSW
  if (ron < roff) {
    h_setup->k6 = roff3/(roff3 - ron3);
    h_setup->k12 = roff6/(roff6 - ron6);
    h_setup->dv6 = -((CT)1.0)/(ron3*roff3);
    h_setup->dv12 = -((CT)1.0)/(ron6*roff6);
  } else {
    h_setup->k6 = 1.0f;
    h_setup->k12 = 1.0f;
    h_setup->dv6 = -((CT)1.0)/(roff6);
    h_setup->dv12 = -((CT)1.0)/(roff6*roff6);
  }
  h_setup->roffinv3 =  ((CT)1.0)/roff3;

  this->vdw_model = vdw_model;
  set_elec_model(elec_model);

  set_calc_vdw(calc_vdw);
  set_calc_elec(calc_elec);

  update_setup();
}

//
// Returns box sizes
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::get_box_size(CT &boxx, CT &boxy, CT &boxz) {
  boxx = h_setup->boxx;
  boxy = h_setup->boxy;
  boxz = h_setup->boxz;
}

//
// Sets box sizes
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_box_size(CT boxx, CT boxy, CT boxz) {
  h_setup->boxx = boxx;
  h_setup->boxy = boxy;
  h_setup->boxz = boxz;
  update_setup();
}

//
// Sets "calc_vdw" flag
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_calc_vdw(bool calc_vdw) {
  this->calc_vdw = calc_vdw;
}

//
// Sets "calc_elec" flag
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_calc_elec(bool calc_elec) {
  this->calc_elec = calc_elec;
}

//
// Sets VdW parameters
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_vdwparam(int nvdwparam, CT *h_vdwparam) {

  this->nvdwparam = nvdwparam;

  // "Fix" vdwparam by multiplying c6 by 6.0 and c12 by 12.0
  // NOTE: this is done in order to avoid the multiplication in the inner loop
  CT *h_vdwparam_fixed = new CT[nvdwparam];
  for(int i=0;i < nvdwparam/2;i++) {
    h_vdwparam_fixed[i*2]   = ((CT)6.0)*h_vdwparam[i*2];
    h_vdwparam_fixed[i*2+1] = ((CT)12.0)*h_vdwparam[i*2+1];
    //h_vdwparam_fixed[i*2]   = h_vdwparam[i*2];
    //h_vdwparam_fixed[i*2+1] = h_vdwparam[i*2+1];
  }

  bool vdwparam_reallocated = false;
  if (nvdwparam > vdwparam_len) {
    reallocate<CT>(&vdwparam, &vdwparam_len, nvdwparam, 1.0f);
    vdwparam_reallocated = true;
  }
  copy_HtoD<CT>(h_vdwparam_fixed, vdwparam, nvdwparam);
  delete [] h_vdwparam_fixed;

  if (use_tex_vdwparam && vdwparam_reallocated) {
    // Unbind texture
    if (vdwparam_texref_bound) {
      cudaCheck(cudaUnbindTexture(vdwparam_texref));
      vdwparam_texref_bound = false;
    }
    // Bind texture
    vdwparam_texref.normalized = 0;
    vdwparam_texref.filterMode = cudaFilterModePoint;
    vdwparam_texref.addressMode[0] = cudaAddressModeClamp;
    vdwparam_texref.channelDesc.x = 32;
    vdwparam_texref.channelDesc.y = 32;
    vdwparam_texref.channelDesc.z = 0;
    vdwparam_texref.channelDesc.w = 0;
    vdwparam_texref.channelDesc.f = cudaChannelFormatKindFloat;
    cudaCheck(cudaBindTexture(NULL, vdwparam_texref, vdwparam, 
			      nvdwparam*sizeof(float)));
    vdwparam_texref_bound = true;
  }
}

//
// Sets VdW parameters by loading them from a file
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_vdwparam(const char *filename) {
  
  int nvdwparam;
  CT *h_vdwparam;

  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(filename);

    file >> nvdwparam;

    h_vdwparam = new float[nvdwparam];

    for (int i=0;i < nvdwparam;i++) {
      file >> h_vdwparam[i];
    }

    file.close();
  }
  catch(std::ifstream::failure e) {
    std::cerr << "Error opening/reading/closing file " << filename << std::endl;
    exit(1);
  }

  set_vdwparam(nvdwparam, h_vdwparam);

  delete [] h_vdwparam;
}

//
// Sets vdwtype array
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_vdwtype(int ncoord, int *h_vdwtype) {
  reallocate<int>(&vdwtype, &vdwtype_len, ncoord, 1.5f);
  copy_HtoD<int>(h_vdwtype, vdwtype, ncoord);
}

//
// Sets vdwtype array by loading it from a file
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_vdwtype(const char *filename) {

  int ncoord;
  int *h_vdwtype;

  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(filename);

    file >> ncoord;

    h_vdwtype = new int[ncoord];

    for (int i=0;i < ncoord;i++) {
      file >> h_vdwtype[i];
    }

    file.close();
  }
  catch(std::ifstream::failure e) {
    std::cerr << "Error opening/reading/closing file " << filename << std::endl;
    exit(1);
  }

  set_vdwtype(ncoord, h_vdwtype);
  
  delete [] h_vdwtype;
}

//
// Builds Ewald lookup table
// roff  = distance cut-off
// h     = the distance between interpolation points
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::setup_ewald_force(CT h) {

  h_setup->hinv = ((CT)1.0)/h;

  n_ewald_force = (int)(sqrt(h_setup->roff2)*h_setup->hinv) + 2;

  CT *h_ewald_force = new CT[n_ewald_force];

  for (int i=1;i < n_ewald_force;i++) {
    const CT two_sqrtpi = (CT)1.12837916709551;    // 2/sqrt(pi)
    CT r = i*h;
    CT r2 = r*r;
    h_ewald_force[i] = two_sqrtpi*((CT)h_setup->kappa)*exp(-((CT)h_setup->kappa2)*r2) + 
      erfc(((CT)h_setup->kappa)*r)/r;
  }
  h_ewald_force[0] = h_ewald_force[1];

  allocate<CT>(&ewald_force, n_ewald_force);
  copy_HtoD<CT>(h_ewald_force, ewald_force, n_ewald_force);

  h_setup->ewald_force = ewald_force;

  delete [] h_ewald_force;

}

//
// Sets method for calculating electrostatic force and energy
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_elec_model(int elec_model, CT h) {
  this->elec_model = elec_model;
  
  if (elec_model == EWALD_LOOKUP) {
    setup_ewald_force(h);
  }
}

//
// Calculates direct force
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::calc_force(const int ncoord, const float4 *xyzq,
				     const NeighborList<32> *nlist,
				     const bool calc_energy,
				     const bool calc_virial,
				     const int stride, AT *force, cudaStream_t stream) {

  const int tilesize = 32;

  if (nlist->ni == 0) return;

  dim3 nthread(32, 2, 1);
  dim3 nblock_tot((nlist->ni-1)/nthread.y+1, 1, 1);

  size_t shmem_size = tilesize*nthread.y*(sizeof(float4) + sizeof(int)) + 
    warpsize*nthread.y*3*sizeof(AT);

  int vdw_model_loc = calc_vdw ? vdw_model : NONE;
  int elec_model_loc = calc_elec ? elec_model : NONE;

  int3 max_nblock3 = get_max_nblock();
  unsigned int max_nblock = max_nblock3.x;
  unsigned int base_tid = 0;

  while (nblock_tot.x != 0) {

    dim3 nblock;
    nblock.x = (nblock_tot.x > max_nblock) ? max_nblock : nblock_tot.x;
    nblock_tot.x -= nblock.x;

    if (vdw_model_loc == VDW_VSH) {
      if (elec_model_loc == EWALD) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
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
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD_LOOKUP, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD_LOOKUP, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, EWALD_LOOKUP, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
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
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, NONE, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, NONE, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSH, NONE, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else {
	std::cout<<"DirectForce<AT, CT>::calc_force, Invalid EWALD model "<<elec_model_loc<<std::endl;
	exit(1);
      }
    } else if (vdw_model_loc == VDW_VSW) {
      if (elec_model_loc == EWALD) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
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
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD_LOOKUP, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD_LOOKUP, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, EWALD_LOOKUP, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
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
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, NONE, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, NONE, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VSW, NONE, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else {
	std::cout<<"DirectForce<AT, CT>::calc_force, Invalid EWALD model "<<elec_model_loc<<std::endl;
	exit(1);
      }
    } else if (vdw_model_loc == VDW_VFSW) {
      if (elec_model_loc == EWALD) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
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
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD_LOOKUP, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD_LOOKUP, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, EWALD_LOOKUP, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
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
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, NONE, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, NONE, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_VFSW, NONE, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else {
	std::cout<<"DirectForce<AT, CT>::calc_force, Invalid EWALD model "<<elec_model_loc<<std::endl;
	exit(1);
      }
    } else if (vdw_model_loc == VDW_CUT) {
      if (elec_model_loc == EWALD) {
	if (calc_energy) {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD, true, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
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
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD_LOOKUP, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD_LOOKUP, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, EWALD_LOOKUP, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
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
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, NONE, true, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	} else {
	  if (calc_virial) {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, NONE, false, true, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  } else {
	    calc_force_kernel <AT, CT, tilesize, VDW_CUT, NONE, false, false, true>
	      <<< nblock, nthread, shmem_size, stream >>>
	      (base_tid, nlist->ni, nlist->ientry, nlist->tile_indj,
	       nlist->tile_excl,
	       stride, vdwparam, nvdwparam, xyzq, vdwtype,
	       force);
	  }
	}
      } else {
	std::cout<<"DirectForce<AT, CT>::calc_force, Invalid EWALD model "<<elec_model_loc<<std::endl;
	exit(1);
      }
    } else {
      std::cout<<"DirectForce<AT, CT>::calc_force, Invalid VDW model"<<std::endl;
      exit(1);
    }

    base_tid += nblock.x*nthread.y;

    cudaCheck(cudaGetLastError());
  }

}

//
// Sets Energies and virials to zero
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::clear_energy_virial() {
  h_energy_virial->energy_vdw = 0.0;
  h_energy_virial->energy_elec = 0.0;
  for (int i=0;i < 27;i++) {
    h_energy_virial->sforcex[i] = 0.0;
    h_energy_virial->sforcey[i] = 0.0;
    h_energy_virial->sforcez[i] = 0.0;
  }
  cudaCheck(cudaMemcpyToSymbol(d_energy_virial, h_energy_virial, sizeof(DirectEnergyVirial_t)));
}

//
// Read Energies and virials
// prev_calc_energy = true, if energy was calculated when the force kernel was last called
// prev_calc_virial = true, if virial was calculated when the force kernel was last called
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::get_energy_virial(bool prev_calc_energy, bool prev_calc_virial,
					    double *energy_vdw, double *energy_elec,
					    double *sforcex, double *sforcey, double *sforcez) {
  if (prev_calc_energy && prev_calc_virial) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, sizeof(DirectEnergyVirial_t)));
  } else if (prev_calc_energy) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, 2*sizeof(double)));
  } else if (prev_calc_virial) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, 27*3*sizeof(double),
				   2*sizeof(double)));
  }
  *energy_vdw = h_energy_virial->energy_vdw;
  *energy_elec = h_energy_virial->energy_elec;
  for (int i=0;i < 27;i++) {
    sforcex[i] = h_energy_virial->sforcex[i];
    sforcey[i] = h_energy_virial->sforcey[i];
    sforcez[i] = h_energy_virial->sforcez[i];
  }
}

//
// Explicit instances of DirectForce
//
template class DirectForce<long long int, float>;
