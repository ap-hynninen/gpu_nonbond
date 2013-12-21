#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "NeighborList.h"
#include "DirectForce.h"

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


enum {NONE, EWALD, EWALD_LOOKUP, VDW_CUT, VDW_VSH, VDW_VSW, VDW_VFSW};

//#define WARPSIZE 32                             // Number of threads per warp
//#define TILESIZE 32                             // Number of atoms per tile direction
//#define NUM_EXCL ((32*32-1)/32 + 1) // Number of exclusion mask integers
//#define NUM_THREAD_PER_EXCL (32/NUM_EXCL)       // Number of threads per exclusion mask integer

struct DirectSettings_t {
  float kappa;
  float kappa2;

  float boxx;
  float boxy;
  float boxz;

  float roff2;
  float ron2;

  float roffinv6;
  float roffinv12;
  float roffinv18;

  float inv_roff2_ron2;
};

// Settings for direct computation in host memory
static DirectSettings_t h_setup;

// Settings for direct computation in device memory
static __constant__ DirectSettings_t d_setup;

static texture<float2, 1, cudaReadModeElementType> vdwparam_texref;
static bool vdwparam_texref_bound = false;

//
// Calculates VdW pair force & energy
//
template <int vdw_model, bool calc_energy>
__forceinline__ __device__
float pair_vdw_force(float r2, float r, float rinv, float rinv2, float c6, float c12, double &vdwpotl) {

  float fij_vdw;

  if (vdw_model == VDW_VSH) {
    float r6 = r2*r2*r2;
    float rinv6 = rinv2*rinv2*rinv2;
    float rinv12 = rinv6*rinv6;
	    
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      vdwpotl += (double)(c12*one_twelve*(rinv12 + 2.0f*r6*d_setup.roffinv18 - 3.0f*d_setup.roffinv12) - 
			  c6*one_six*(rinv6 + r6*d_setup.roffinv12 - 2.0f*d_setup.roffinv6));
    }
	  
    fij_vdw = c6*(rinv6 - r6*d_setup.roffinv12) - c12*(rinv12 + r6*d_setup.roffinv18);
  } else if (vdw_model == VDW_VSW) {
    float roff2_r2_sq = d_setup.roff2 - r2;
    roff2_r2_sq *= roff2_r2_sq;
    float sw = (r2 <= d_setup.ron2) ? 1.0f : roff2_r2_sq*(d_setup.roff2 + 2.0f*r2 - 
							  3.0f*d_setup.ron2)*d_setup.inv_roff2_ron2;
    float dsw = (r2 <= d_setup.ron2) ? 0.0f : 6.0f*(d_setup.roff2-r2)*(d_setup.ron2-r2)*d_setup.inv_roff2_ron2;
    float rinv6 = rinv2*rinv2*rinv2;
    if (calc_energy) {
      vdwpotl += (double)((c12*rinv6 - c6)*rinv6*sw);
    }
    sw *= 3.0f*rinv2;
    fij_vdw = 2.0f*r*rinv6*(c12*rinv6*(-2.0f*sw + dsw)
			    + c6*(sw - dsw));
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

  } else if (vdw_model == NONE) {
    fij_vdw = 0.0f;
  }

  return fij_vdw;
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
    fij_elec = 0.0f; //qq*lookup_force(r, kappa);
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
	  bool calc_energy, bool tex_vdwparam>
__global__ void calc_force_kernel(const int ni, const ientry_t *ientry,
				  const int *tile_indj, const tile_excl_t<tilesize> *tile_excl,
				  const int stride,
				  const float *vdwparam, const int nvdwparam,
				  const float4 *xyzq, const int *vdwtype,
				  AT *force) {

  // Pre-computed constants
  const int warpsize = 32;
  const int num_excl = ((tilesize*tilesize-1)/32 + 1);
  const int num_thread_per_excl = (32/num_excl);

  //
  // Shared data, common for the entire block
  //
   extern __shared__ char shmem[];

   volatile float4 *xyzq_i = (float4 *)&shmem[0];                   // tilesize*blockDim.y
   volatile int *vdwtype_i = (int *)&xyzq_i[tilesize*blockDim.y];   // tilesize*blockDim.y
   volatile AT *fix = (AT *)&vdwtype_i[tilesize*blockDim.y];        // WARPSIZE*blockDim.y
   volatile AT *fiy = &fix[warpsize*blockDim.y];                    // WARPSIZE*blockDim.y
   volatile AT *fiz = &fiy[warpsize*blockDim.y];                    // WARPSIZE*blockDim.y
   volatile float *vdwparam_sh;

   if (tex_vdwparam) {
     vdwparam_sh = (float *)&fiz[warpsize*blockDim.y];
   }

   /*
  __shared__ float4 xyzq_i[TILESIZE*TILEX_NBLOCK];
  __shared__ int vdwtype_i[TILESIZE*TILEX_NBLOCK];
  __shared__ AT fix[WARPSIZE*TILEX_NBLOCK];
  __shared__ AT fiy[WARPSIZE*TILEX_NBLOCK];
  __shared__ AT fiz[WARPSIZE*TILEX_NBLOCK];
   */

  /*
#ifdef PREC_SPDP
  __shared__ FORCE3_T fj_tmp[WARPSIZE*TILEX_NBLOCK];
#endif
#ifndef TEX_FETCH_VDWPARAM
  __shared__ float vdwparam_sh[MAX_NVDWPARAM];
#endif
  */

  // Load ientry
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
  const unsigned int tid = threadIdx.x + blockDim.x*threadIdx.y;

  unsigned int load_ij;
  if (tilesize == 16) {
    load_ij = threadIdx.x % tilesize;
  } else {
    load_ij = threadIdx.x;
  }

  // Load i-atom data to shared memory (and shift coordinates)
  float4 xyzq_tmp = xyzq[indi + load_ij];
  xyzq_i[sh_start + load_ij].x = xyzq_tmp.x + shx;
  xyzq_i[sh_start + load_ij].y = xyzq_tmp.y + shy;
  xyzq_i[sh_start + load_ij].z = xyzq_tmp.z + shz;
  xyzq_i[sh_start + load_ij].w = xyzq_tmp.w;

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

  //  float roff2 = roff*roff;

  //  vdw.setup(ron2, roff2);

  /*
#if (VDWTYPE == VSH)
  // roffinv6  = 1/roff^6
  // roffinv12 = 1/roff^12
  // roffinv18 = 1/roff^18
  float roffinv6 = 1.0f/(roff2*roff2*roff2);
  float roffinv12 = roffinv6*roffinv6;
  float roffinv18 = roffinv12*roffinv6;
#elif (VDWTYPE == VSW)
  float ron2 = ron*ron;
  //  inv_roff2_ron2 = 1.0/(roff2 - ron2)^3
  float inv_roff2_ron2 = roff2 - ron2;
  inv_roff2_ron2 = 1.0f/(inv_roff2_ron2*inv_roff2_ron2*inv_roff2_ron2);
#endif

#if (ELECTYPE == EWALD)
  float kappa2 = kappa*kappa;
#endif

#ifdef CALC_ENERGY
  double vdwpotl = 0.0;
  double coulpotl = 0.0;
#endif
  */

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
	
	float dx = xyzq_i[ii].x - xyzq_j.x;
	float dy = xyzq_i[ii].y - xyzq_j.y;
	float dz = xyzq_i[ii].z - xyzq_j.z;
	
	float r2 = dx*dx + dy*dy + dz*dz;

	if (r2 < d_setup.roff2) {

	  int ia = vdwtype_i[ii];
	  int aa = (ja > ia) ? ja : ia;      // aa = max(ja,ia)
	  int ivdw = (aa*(aa-3) + 2*(ja + ia) - 2) >> 1;

	  float c6, c12;
	  if (tex_vdwparam) {
	    float2 c6c12 = tex1Dfetch(vdwparam_texref, ivdw);
	    c6  = c6c12.x;
	    c12 = c6c12.y;
	  } else {
	    c6 = vdwparam_sh[ivdw];
	    c12 = vdwparam_sh[ivdw+1];
	  }

	  float rinv = rsqrtf(r2);
	  float rinv2 = rinv*rinv;
	  float r = r2*rinv;

	  float fij_vdw = pair_vdw_force<vdw_model, calc_energy>(r2, r, rinv, rinv2, c6, c12, vdwpotl);

	  float fij_elec = pair_elec_force<elec_model, calc_energy>(r2, r, rinv, xyzq_i[ii].w, xyzq_j.w, coulpotl);

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
	}
      }

      // Advance exclusion mask
      excl >>= 1;
    }

    // Dump register forces (fjx, fjy, fjz)
    write_force<AT, CT>(fjx, fjy, fjz, indj+load_ij, stride, force);
  }

  // Dump shared memory force (fi)
  __syncthreads();
  write_force<AT, CT>(fix[tid], fiy[tid], fiz[tid], indi+load_ij, stride, force);

  if (calc_energy) {
    // Reduce energies to (pot)
    // Reduces within thread block, uses the "xyzq_i" shared memory buffer
    __syncthreads();          // NOTE: this makes sure we can write to xyzq_i 
    double2 *potbuf = (double2 *)(xyzq_i);
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

  set_calc_vdw(true);
  set_calc_elec(true);
}

//
// Class destructor
//
template <typename AT, typename CT>
DirectForce<AT, CT>::~DirectForce() {
  if (vdwparam != NULL) deallocate<CT>(&vdwparam);
  if (vdwtype != NULL) deallocate<int>(&vdwtype);
}

//
// Copies h_setup -> d_setup
//
void update_setup() {
  cudaCheck(cudaMemcpyToSymbol(&d_setup, &h_setup, sizeof(DirectSettings_t)));
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
  h_setup.boxx = boxx;
  h_setup.boxy = boxy;
  h_setup.boxz = boxz;
  h_setup.kappa = kappa;
  h_setup.kappa2 = kappa*kappa;
  h_setup.roff2 = roff*roff;
  h_setup.ron2 = ron*ron;

  h_setup.roffinv6 = ((CT)1.0)/(h_setup.roff2*h_setup.roff2*h_setup.roff2);
  h_setup.roffinv12 = h_setup.roffinv6*h_setup.roffinv6;
  h_setup.roffinv18 = h_setup.roffinv12*h_setup.roffinv6;

  h_setup.inv_roff2_ron2 = h_setup.roff2 - h_setup.ron2;
  h_setup.inv_roff2_ron2 = ((CT)1.0)/(h_setup.inv_roff2_ron2*h_setup.inv_roff2_ron2*h_setup.inv_roff2_ron2);

  this->vdw_model = vdw_model;
  this->elec_model = elec_model;

  set_calc_vdw(calc_vdw);
  set_calc_elec(calc_elec);

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
// Sets box size
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_box_size(CT boxx, CT boxy, CT boxz) {
  h_setup.boxx = boxx;
  h_setup.boxy = boxy;
  h_setup.boxz = boxz;
  update_setup();
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
      vdwparam_texref_bound = 0;
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
  reallocate<int>(&vdwtype, &vdwtype_len, ncoord, 1.2f);
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
// Calculates direct force
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::calc_force(const int ncoord, const float4 *xyzq,				     
				     const NeighborList<32> *nlist, const bool calc_energy,
				     AT *force) {

  int stride = 0;

  dim3 nthread(32, 6, 1);
  dim3 nblock((nlist->ni-1)/nthread.y+1, 1, 1);

  int vdw_model_loc = calc_vdw ? vdw_model : NONE;
  int elec_model_loc = calc_elec ? elec_model : NONE;

  if (vdw_model_loc == VDW_VSH) {
    if (elec_model_loc == EWALD) {
      if (calc_energy) {
	calc_force_kernel <AT, CT, 32, VDW_VSH, EWALD, true, true>
	  <<< nblock, nthread >>>(nlist->ni, nlist->ientry, nlist->tile_indj, nlist->tile_excl,
				  stride, vdwparam, nvdwparam, xyzq, vdwtype,
				  force);
      } else {
	calc_force_kernel <AT, CT, 32, VDW_VSH, EWALD, false, true>
	  <<< nblock, nthread >>>(nlist->ni, nlist->ientry, nlist->tile_indj, nlist->tile_excl,
				  stride, vdwparam, nvdwparam, xyzq, vdwtype,
				  force);
      }
    } else {
      std::cout<<"DirectForce<AT, CT>::calc_force, Invalid EWALD model"<<std::endl;
      exit(1);
    }
  } else {
    std::cout<<"DirectForce<AT, CT>::calc_force, Invalid VDW model"<<std::endl;
    exit(1);
  }
}


//
// Explicit instances of DirectForce
//
template class DirectForce<long long int, float>;
