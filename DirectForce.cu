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
static texture<float2, 1, cudaReadModeElementType> vdwparam14_texref;
static bool vdwparam14_texref_bound = false;

//
// Calculates VdW pair force & energy
//
template <int vdw_model, bool calc_energy>
__forceinline__ __device__
float pair_vdw_force(const float r2, const float r, const float rinv, const float rinv2,
		     const float c6, const float c12,
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
__forceinline__ __device__ float lookup_force(const float r, const float hinv) {
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
float pair_elec_force(const float r2, const float r, const float rinv, 
		      const float qq, double &coulpotl) {

  float fij_elec;

  if (elec_model == EWALD_LOOKUP) {
    fij_elec = qq*lookup_force(r, d_setup.hinv);
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
// Calculates electrostatic force & energy for 1-4 interactions and exclusions
//
template <int elec_model, bool calc_energy>
__forceinline__ __device__
float pair_elec_force_14(const float r2, const float r, const float rinv,
			 const float qq, const float e14fac, double &coulpotl) {

  float fij_elec;

  if (elec_model == EWALD) {
    float erfc_val = fasterfc(d_setup.kappa*r);
    float exp_val = expf(-d_setup.kappa2*r2);
    float qq_efac_rinv = qq*(erfc_val + e14fac - 1.0f)*rinv;
    if (calc_energy) {
      coulpotl += (double)qq_efac_rinv;
    }
    const float two_sqrtpi = 1.12837916709551f;    // 2/sqrt(pi)
    fij_elec = -qq*two_sqrtpi*d_setup.kappa*exp_val - qq_efac_rinv;
  } else if (elec_model == NONE) {
    fij_elec = 0.0f;
  }

  return fij_elec;
}

//
// 1-4 interaction force
//
template <typename AT, typename CT, int vdw_model, int elec_model, 
	  bool calc_energy, bool calc_virial, bool tex_vdwparam>
__device__ void calc_in14_force_device(const int pos, const list14_t* in14list,
				       const int* vdwtype, const float* vdwparam14,
				       const float4* xyzq, const int stride, AT *force,
				       double &vdw_pot, double &elec_pot) {

  int i = in14list[pos].i - 1;
  int j = in14list[pos].j - 1;
  int ish = in14list[pos].ishift;
  float3 sh_xyz = calc_box_shift(ish, d_setup.boxx, d_setup.boxy, d_setup.boxz);
  // Load atom coordinates
  float4 xyzqi = xyzq[i];
  float4 xyzqj = xyzq[j];
  // Calculate distance
  CT dx = xyzqi.x - xyzqj.x + sh_xyz.x;
  CT dy = xyzqi.y - xyzqj.y + sh_xyz.y;
  CT dz = xyzqi.z - xyzqj.z + sh_xyz.z;
  CT r2 = dx*dx + dy*dy + dz*dz;
  CT qq = ccelec*xyzqi.w*xyzqj.w;
  // Calculate the interaction
  CT r = sqrtf(r2);
  CT rinv = ((CT)1)/r;

  int ia = vdwtype[i];
  int ja = vdwtype[j];
  int aa = max(ja, ia);
  int ivdw = (aa*(aa-3) + 2*(ja + ia) - 2) >> 1;

  CT c6, c12;
  if (tex_vdwparam) {
    //c6 = __ldg(&vdwparam[ivdw]);
    //c12 = __ldg(&vdwparam[ivdw+1]);
    float2 c6c12 = tex1Dfetch(vdwparam14_texref, ivdw);
    c6  = c6c12.x;
    c12 = c6c12.y;
  } else {
    c6 = vdwparam14[ivdw];
    c12 = vdwparam14[ivdw+1];
  }

  CT rinv2 = rinv*rinv;

  CT fij_vdw = pair_vdw_force<vdw_model, calc_energy>(r2, r, rinv, rinv2, c6, c12, vdw_pot);

  CT fij_elec = pair_elec_force_14<elec_model, calc_energy>(r2, r, rinv, qq,
							    d_setup.e14fac, elec_pot);

  CT fij = (fij_vdw + fij_elec)*rinv2;

  // Calculate force components
  AT fxij, fyij, fzij;
  calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);

  // Store forces
  write_force<AT>(fxij, fyij, fzij,    i, stride, force);
  write_force<AT>(-fxij, -fyij, -fzij, j, stride, force);
  
  // Store shifted forces
  if (calc_virial) {
    //sforce(is)   = sforce(is)   + fijx
    //sforce(is+1) = sforce(is+1) + fijy
    //sforce(is+2) = sforce(is+2) + fijz
  }

}

//
// 1-4 exclusion force
//
template <typename AT, typename CT, int elec_model, bool calc_energy, bool calc_virial>
__device__ void calc_ex14_force_device(const int pos, const list14_t* ex14list,
				       const float4* xyzq, const int stride, AT *force,
				       double &elec_pot) {

  int i = ex14list[pos].i - 1;
  int j = ex14list[pos].j - 1;
  int ish = ex14list[pos].ishift;
  float3 sh_xyz = calc_box_shift(ish, d_setup.boxx, d_setup.boxy, d_setup.boxz);
  // Load atom coordinates
  float4 xyzqi = xyzq[i];
  float4 xyzqj = xyzq[j];
  // Calculate distance
  CT dx = xyzqi.x - xyzqj.x + sh_xyz.x;
  CT dy = xyzqi.y - xyzqj.y + sh_xyz.y;
  CT dz = xyzqi.z - xyzqj.z + sh_xyz.z;
  CT r2 = dx*dx + dy*dy + dz*dz;
  CT qq = ccelec*xyzqi.w*xyzqj.w;
  // Calculate the interaction
  CT r = sqrtf(r2);
  CT rinv = ((CT)1)/r;
  CT rinv2 = rinv*rinv;
  CT fij_elec = pair_elec_force_14<elec_model, calc_energy>(r2, r, rinv, qq,
							    0.0f, elec_pot);
  CT fij = fij_elec*rinv2;
  // Calculate force components
  AT fxij, fyij, fzij;
  calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);

  // Store forces
  write_force<AT>(fxij, fyij, fzij,    i, stride, force);
  write_force<AT>(-fxij, -fyij, -fzij, j, stride, force);
  // Store shifted forces
  if (calc_virial) {
    //sforce(is)   = sforce(is)   + fijx
    //sforce(is+1) = sforce(is+1) + fijy
    //sforce(is+2) = sforce(is+2) + fijz
  }

}

//
// 1-4 exclusion and interaction calculation kernel
//
template <typename AT, typename CT, int vdw_model, int elec_model, 
	  bool calc_energy, bool calc_virial, bool tex_vdwparam>
__global__ void calc_14_force_kernel(const int nin14list, const int nex14list,
				     const int nin14block,
				     const list14_t* in14list, const list14_t* ex14list,
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
// Nonbonded virial
//
template <typename AT, typename CT>
__global__ void calc_virial_kernel(const int ncoord, const float4* __restrict__ xyzq,
				   const int stride, const AT* __restrict__ force) {
  // Shared memory:
  // Required memory
  // blockDim.x*9*sizeof(double) for __CUDA_ARCH__ < 300
  // blockDim.x*9*sizeof(double)/warpsize for __CUDA_ARCH__ >= 300
  extern __shared__ volatile double sh_vir[];

  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  const int ish = i - ncoord;

  double vir[9];
  if (i < ncoord) {
    float4 xyzqi = xyzq[i];
    double x = (double)xyzqi.x;
    double y = (double)xyzqi.y;
    double z = (double)xyzqi.z;
    double fx = ((double)force[i])*INV_FORCE_SCALE;
    double fy = ((double)force[i+stride])*INV_FORCE_SCALE;
    double fz = ((double)force[i+stride*2])*INV_FORCE_SCALE;
    vir[0] = x*fx;
    vir[1] = x*fy;
    vir[2] = x*fz;
    vir[3] = y*fx;
    vir[4] = y*fy;
    vir[5] = y*fz;
    vir[6] = z*fx;
    vir[7] = z*fy;
    vir[8] = z*fz;
  } else if (ish >= 0 && ish <= 26) {
    double sforcex = d_energy_virial.sforcex[ish];
    double sforcey = d_energy_virial.sforcey[ish];
    double sforcez = d_energy_virial.sforcez[ish];
    int ish_tmp = ish;
    double shz = (double)((ish_tmp/9 - 1)*d_setup.boxz);
    ish_tmp -= (ish_tmp/9)*9;
    double shy = (double)((ish_tmp/3 - 1)*d_setup.boxy);
    ish_tmp -= (ish_tmp/3)*3;
    double shx = (double)((ish_tmp - 1)*d_setup.boxx);
    vir[0] = shx*sforcex;
    vir[1] = shx*sforcey;
    vir[2] = shx*sforcez;
    vir[3] = shy*sforcex;
    vir[4] = shy*sforcey;
    vir[5] = shy*sforcez;
    vir[6] = shz*sforcex;
    vir[7] = shz*sforcey;
    vir[8] = shz*sforcez;
  } else {
#pragma unroll
    for (int k=0;k < 9;k++)
      vir[k] = 0.0;
  }

  // Reduce
  //#if __CUDA_ARCH__ < 300
  // 0-2
#pragma unroll
  for (int k=0;k < 3;k++)
    sh_vir[threadIdx.x + k*blockDim.x] = vir[k];
  __syncthreads();
  for (int i=1;i < blockDim.x;i *= 2) {
    int pos = threadIdx.x + i;
    double vir_val[3];
#pragma unroll
    for (int k=0;k < 3;k++)
      vir_val[k] = (pos < blockDim.x) ? sh_vir[pos + k*blockDim.x] : 0.0;
    __syncthreads();
#pragma unroll
    for (int k=0;k < 3;k++)
      sh_vir[threadIdx.x + k*blockDim.x] += vir_val[k];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k=0;k < 3;k++)
      atomicAdd(&d_energy_virial.vir[k], -sh_vir[k*blockDim.x]);
  }

  // 3-5
#pragma unroll
  for (int k=0;k < 3;k++)
    sh_vir[threadIdx.x + k*blockDim.x] = vir[k+3];
  __syncthreads();
  for (int i=1;i < blockDim.x;i *= 2) {
    int pos = threadIdx.x + i;
    double vir_val[3];
#pragma unroll
    for (int k=0;k < 3;k++)
      vir_val[k] = (pos < blockDim.x) ? sh_vir[pos + k*blockDim.x] : 0.0;
    __syncthreads();
#pragma unroll
    for (int k=0;k < 3;k++)
      sh_vir[threadIdx.x + k*blockDim.x] += vir_val[k];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k=0;k < 3;k++)
      atomicAdd(&d_energy_virial.vir[k+3], -sh_vir[k*blockDim.x]);
  }

  // 6-8
#pragma unroll
  for (int k=0;k < 3;k++)
    sh_vir[threadIdx.x + k*blockDim.x] = vir[k+6];
  __syncthreads();
  for (int i=1;i < blockDim.x;i *= 2) {
    int pos = threadIdx.x + i;
    double vir_val[3];
#pragma unroll
    for (int k=0;k < 3;k++)
      vir_val[k] = (pos < blockDim.x) ? sh_vir[pos + k*blockDim.x] : 0.0;
    __syncthreads();
#pragma unroll
    for (int k=0;k < 3;k++)
      sh_vir[threadIdx.x + k*blockDim.x] += vir_val[k];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k=0;k < 3;k++)
      atomicAdd(&d_energy_virial.vir[k+6], -sh_vir[k*blockDim.x]);
  }

  /*
#else

  // Warp index
  const int wid = threadIdx.x / warpsize;
  // Thread index within warp
  const int tid = threadIdx.x % warpsize;

  int blockdim = blockDim.x;
  while (blockdim > 0) {
    // Reduce within warp
    for (int i=16;i >= 1;i /= 2) {
#pragma unroll
      for (int k=0;k < 9;k++)
	vir[k] += __hiloint2double(__shfl_xor(__double2hiint(vir[k]), i),
				   __shfl_xor(__double2loint(vir[k]), i));
    }

    // After reducing withing warps, block size is reduced by a factor warpsize
    blockdim /= warpsize;

    // Root thread of the warp stores result into shared memory
    if (tid == 0) {
#pragma unroll
      for (int k=0;k < 9;k++)
	sh_vir[wid + k*blockdim] = vir[k];
    }
    __syncthreads();

    if (threadIdx.x < blockdim) {
#pragma unroll
      for (int k=0;k < 9;k++)
	vir[k] = sh_vir[threadIdx.x + k*blockdim];
    }

  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int k=0;k < 9;k++)
      atomicAdd(&d_energy_virial.vir[k], -vir[k]);
  }

#endif
  */

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

//
// Sets vdwtype from a global list
//
__global__ void set_vdwtype_kernel(const int ncoord, const int* __restrict__ glo_vdwtype,
				   const int* __restrict__ loc2glo, int* __restrict__ vdwtype) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < ncoord) {
    int j = loc2glo[i];
    vdwtype[i] = glo_vdwtype[j];
  }
}


//########################################################################################
//########################################################################################
//########################################################################################

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

  vdwparam14 = NULL;
  nvdwparam14 = 0;
  vdwparam14_len = 0;
  use_tex_vdwparam14 = true;
  vdwparam14_texref_bound = false;

  nin14list = 0;
  in14list_len = 0;
  in14list = NULL;

  nex14list = 0;
  ex14list_len = 0;
  ex14list = NULL;

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
  if (vdwparam14_texref_bound) {
    cudaCheck(cudaUnbindTexture(vdwparam14_texref));
    vdwparam14_texref_bound = false;
  }
  if (vdwparam14 != NULL) deallocate<CT>(&vdwparam14);
  if (in14list != NULL) deallocate<list14_t>(&in14list);
  if (ex14list != NULL) deallocate<list14_t>(&ex14list);
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
				CT e14fac,
				int vdw_model, int elec_model) {

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

  h_setup->e14fac = e14fac;

  this->vdw_model = vdw_model;
  set_elec_model(elec_model);

  set_calc_vdw(true);
  set_calc_elec(true);

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
void DirectForce<AT, CT>::set_box_size(const CT boxx, const CT boxy, const CT boxz) {
  h_setup->boxx = boxx;
  h_setup->boxy = boxy;
  h_setup->boxz = boxz;
  update_setup();
}

//
// Sets "calc_vdw" flag
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_calc_vdw(const bool calc_vdw) {
  this->calc_vdw = calc_vdw;
}

//
// Sets "calc_elec" flag
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_calc_elec(const bool calc_elec) {
  this->calc_elec = calc_elec;
}

//
// Sets VdW parameters by copying them from CPU
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::setup_vdwparam(const int type, const int h_nvdwparam, const CT *h_vdwparam) {
  assert(type == VDW_MAIN || type == VDW_IN14);

  int *nvdwparam_loc;
  int *vdwparam_len_loc;
  CT **vdwparam_loc;

  if (type == VDW_MAIN) {
    nvdwparam_loc = &this->nvdwparam;
    vdwparam_len_loc = &this->vdwparam_len;
    vdwparam_loc = &this->vdwparam;
  } else {
    nvdwparam_loc = &this->nvdwparam14;
    vdwparam_len_loc = &this->vdwparam14_len;
    vdwparam_loc = &this->vdwparam14;
  }

  *nvdwparam_loc = h_nvdwparam;

  // "Fix" vdwparam by multiplying c6 by 6.0 and c12 by 12.0
  // NOTE: this is done in order to avoid the multiplication in the inner loop
  CT *h_vdwparam_fixed = new CT[*nvdwparam_loc];
  for(int i=0;i < *nvdwparam_loc/2;i++) {
    h_vdwparam_fixed[i*2]   = ((CT)6.0)*h_vdwparam[i*2];
    h_vdwparam_fixed[i*2+1] = ((CT)12.0)*h_vdwparam[i*2+1];
  }

  bool vdwparam_reallocated = false;
  if (*nvdwparam_loc > *vdwparam_len_loc) {
    reallocate<CT>(vdwparam_loc, vdwparam_len_loc, *nvdwparam_loc, 1.0f);
    vdwparam_reallocated = true;
  }
  copy_HtoD<CT>(h_vdwparam_fixed, *vdwparam_loc, *nvdwparam_loc);
  delete [] h_vdwparam_fixed;

  bool *use_tex_vdwparam_loc;
  bool *vdwparam_texref_bound_loc;
  texture<float2, 1, cudaReadModeElementType> *vdwparam_texref_loc;
  if (type == VDW_MAIN) {
    use_tex_vdwparam_loc = &this->use_tex_vdwparam;
    vdwparam_texref_bound_loc = &vdwparam_texref_bound;
    vdwparam_texref_loc = &vdwparam_texref;
  } else {
    use_tex_vdwparam_loc = &this->use_tex_vdwparam14;
    vdwparam_texref_bound_loc = &vdwparam14_texref_bound;
    vdwparam_texref_loc = &vdwparam14_texref;
  }

  if (*use_tex_vdwparam_loc && vdwparam_reallocated) {
    // Unbind texture
    if (*vdwparam_texref_bound_loc) {
      cudaCheck(cudaUnbindTexture(*vdwparam_texref_loc));
      *vdwparam_texref_bound_loc = false;
    }
    // Bind texture
    vdwparam_texref_loc->normalized = 0;
    vdwparam_texref_loc->filterMode = cudaFilterModePoint;
    vdwparam_texref_loc->addressMode[0] = cudaAddressModeClamp;
    vdwparam_texref_loc->channelDesc.x = 32;
    vdwparam_texref_loc->channelDesc.y = 32;
    vdwparam_texref_loc->channelDesc.z = 0;
    vdwparam_texref_loc->channelDesc.w = 0;
    vdwparam_texref_loc->channelDesc.f = cudaChannelFormatKindFloat;
    cudaCheck(cudaBindTexture(NULL, *vdwparam_texref_loc, *vdwparam_loc, 
			      (*nvdwparam_loc)*sizeof(float)));
    *vdwparam_texref_bound_loc = true;
  }

}

//
// Loads vdwparam from a file
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::load_vdwparam(const char *filename, const int nvdwparam, CT **h_vdwparam) {
  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(filename);
    *h_vdwparam = new float[nvdwparam];
    for (int i=0;i < nvdwparam;i++) {
      file >> (*h_vdwparam)[i];
    }
    file.close();
  }
  catch(std::ifstream::failure e) {
    std::cerr << "Error opening/reading/closing file " << filename << std::endl;
    exit(1);
  }
}

//
// Sets VdW parameters by copying them from CPU
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_vdwparam(const int nvdwparam, const CT *h_vdwparam) {
  setup_vdwparam(VDW_MAIN, nvdwparam, h_vdwparam);
}

//
// Sets VdW parameters by loading them from a file
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_vdwparam(const int nvdwparam, const char *filename) {  
  CT *h_vdwparam;
  load_vdwparam(filename, nvdwparam, &h_vdwparam);
  setup_vdwparam(VDW_MAIN, nvdwparam, h_vdwparam);
  delete [] h_vdwparam;
}

//
// Sets VdW parameters by copying them from CPU
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_vdwparam14(const int nvdwparam, const CT *h_vdwparam) {
  setup_vdwparam(VDW_IN14, nvdwparam, h_vdwparam);
}

//
// Sets VdW parameters by loading them from a file
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_vdwparam14(const int nvdwparam, const char *filename) {  
  CT *h_vdwparam;
  load_vdwparam(filename, nvdwparam, &h_vdwparam);
  setup_vdwparam(VDW_IN14, nvdwparam, h_vdwparam);
  delete [] h_vdwparam;
}

//
// Sets vdwtype array from global list in device memory memory
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_vdwtype(const int ncoord, const int *glo_vdwtype,
				      const int *loc2glo, cudaStream_t stream) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord-1)/warpsize+1)*warpsize;
  reallocate<int>(&vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);

  int nthread = 512;
  int nblock = (ncoord - 1)/nthread + 1;
  set_vdwtype_kernel<<< nblock, nthread, 0, stream >>>
    (ncoord, glo_vdwtype, loc2glo, vdwtype);
  cudaCheck(cudaGetLastError());
}

//
// Sets vdwtype array from host memory
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_vdwtype(const int ncoord, const int *h_vdwtype) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord-1)/warpsize+1)*warpsize;
  reallocate<int>(&vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);
  copy_HtoD<int>(h_vdwtype, vdwtype, ncoord);
}

//
// Sets vdwtype array by loading it from a file
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_vdwtype(const int ncoord, const char *filename) {

  int *h_vdwtype;

  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(filename);

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
// Sets 1-4 interaction and exclusion lists
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::set_14_list(int nin14list, int nex14list,
				      list14_t* h_in14list, list14_t* h_ex14list) {

  this->nin14list = nin14list;
  this->nex14list = nex14list;

  if (nin14list > 0) {
    reallocate<list14_t>(&in14list, &in14list_len, nin14list);
    copy_HtoD<list14_t>(h_in14list, in14list, nin14list);
  }

  if (nex14list > 0) {
    reallocate<list14_t>(&ex14list, &ex14list_len, nex14list);
    copy_HtoD<list14_t>(h_ex14list, ex14list, nex14list);
  }

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
// Calculates 1-4 exclusions and interactions
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::calc_14_force(const float4 *xyzq,
					const bool calc_energy, const bool calc_virial,
					const int stride, AT *force, cudaStream_t stream) {

  if (!vdwparam_texref_bound) {
    std::cerr << "DirectForce<AT, CT>::calc_14_force, vdwparam14_texref must be bound" << std::endl;
    exit(1);
  }

  int nthread = 512;
  //int nblock = (nin14list + nex14list - 1)/nthread + 1;
  int nin14block = (nin14list - 1)/nthread + 1;
  int nex14block = (nex14list - 1)/nthread + 1;
  int nblock = nin14block + nex14block;
  int shmem_size = 0;
  if (calc_energy) {
    shmem_size = nthread*sizeof(double2);
  }

  int vdw_model_loc = calc_vdw ? vdw_model : NONE;
  int elec_model_loc = calc_elec ? elec_model : NONE;
  if (elec_model_loc == NONE && vdw_model_loc == NONE) return;

  if (vdw_model_loc == VDW_VSH) {
    if (elec_model_loc == EWALD) {
      if (calc_energy) {
	if (calc_virial) {
	  calc_14_force_kernel <AT, CT, VDW_VSH, EWALD, true, true, true>
	    <<< nblock, nthread, shmem_size, stream >>>
	    (nin14list, nex14list, nin14block, in14list, ex14list, vdwtype, vdwparam14, xyzq,
	     stride, force);
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
      std::cout<<"DirectForce<AT, CT>::calc_14_force, Invalid EWALD model "
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
      std::cout<<"DirectForce<AT, CT>::calc_14_force, Invalid EWALD model "
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
      std::cout<<"DirectForce<AT, CT>::calc_14_force, Invalid EWALD model "
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
      std::cout<<"DirectForce<AT, CT>::calc_14_force, Invalid EWALD model "
	       <<elec_model_loc<<std::endl;
      exit(1);
    }
  } else {
    std::cout<<"DirectForce<AT, CT>::calc_14_force, Invalid VDW model"<<std::endl;
    exit(1);
  }

  cudaCheck(cudaGetLastError());
}

//
// Calculates direct force
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::calc_force(const float4 *xyzq,
				     const NeighborList<32> *nlist,
				     const bool calc_energy,
				     const bool calc_virial,
				     const int stride, AT *force, cudaStream_t stream) {

  const int tilesize = 32;

  if (!vdwparam_texref_bound) {
    std::cerr << "DirectForce<AT, CT>::calc_force, vdwparam_texref must be bound" << std::endl;
    exit(1);
  }

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
	std::cout<<"DirectForce<AT, CT>::calc_force, Invalid EWALD model "<<elec_model_loc<<std::endl;
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
	std::cout<<"DirectForce<AT, CT>::calc_force, Invalid EWALD model "<<elec_model_loc<<std::endl;
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
	std::cout<<"DirectForce<AT, CT>::calc_force, Invalid EWALD model "<<elec_model_loc<<std::endl;
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
	std::cout<<"DirectForce<AT, CT>::calc_force, Invalid EWALD model "
		 <<elec_model_loc<<std::endl;
	exit(1);
      }
    } else {
      std::cout<<"DirectForce<AT, CT>::calc_force, Invalid VDW model"<<std::endl;
      exit(1);
    }

    base += (nthread/warpsize)*nblock;

    cudaCheck(cudaGetLastError());
  }

}

//
// Calculates virial
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::calc_virial(const int ncoord, const float4 *xyzq,
				      const int stride, AT *force,
				      cudaStream_t stream) {

  int nthread, nblock, shmem_size;
  nthread = 256;
  nblock = (ncoord+27-1)/nthread + 1;
  shmem_size = nthread*3*sizeof(double);

  calc_virial_kernel <AT, CT>
    <<< nblock, nthread, shmem_size, stream>>>
    (ncoord, xyzq, stride, force);

  cudaCheck(cudaGetLastError());
}

//
// Sets Energies and virials to zero
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::clear_energy_virial(cudaStream_t stream) {
  //clear_gpu_array<DirectEnergyVirial_t>(&d_energy_virial, 1, stream);
  h_energy_virial->energy_vdw = 0.0;
  h_energy_virial->energy_elec = 0.0;
  h_energy_virial->energy_excl = 0.0;
  for (int i=0;i < 9;i++)
    h_energy_virial->vir[i] = 0.0;
  for (int i=0;i < 27;i++) {
    h_energy_virial->sforcex[i] = 0.0;
    h_energy_virial->sforcey[i] = 0.0;
    h_energy_virial->sforcez[i] = 0.0;
  }
  cudaCheck(cudaMemcpyToSymbolAsync(d_energy_virial, h_energy_virial, sizeof(DirectEnergyVirial_t),
				    0, cudaMemcpyHostToDevice, stream));
}

//
// Read Energies and virials
// prev_calc_energy = true, if energy was calculated when the force kernel was last called
// prev_calc_virial = true, if virial was calculated when the force kernel was last called
//
template <typename AT, typename CT>
void DirectForce<AT, CT>::get_energy_virial(bool prev_calc_energy, bool prev_calc_virial,
					    double *energy_vdw, double *energy_elec,
					    double *energy_excl,
					    double *vir) {
  if (prev_calc_energy && prev_calc_virial) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, (3+9)*sizeof(double) ));
  } else if (prev_calc_energy) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, 3*sizeof(double)));
  } else if (prev_calc_virial) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, 9*sizeof(double),
				   3*sizeof(double)));
  }
  *energy_vdw = h_energy_virial->energy_vdw;
  *energy_elec = h_energy_virial->energy_elec;
  *energy_excl = h_energy_virial->energy_excl;
  for (int i=0;i < 9;i++) {
    vir[i] = h_energy_virial->vir[i];
  }

}

//
// Explicit instances of DirectForce
//
template class DirectForce<long long int, float>;
