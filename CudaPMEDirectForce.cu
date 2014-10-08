#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "NeighborList.h"
#include "CudaPMEDirectForce.h"

// Settings for direct computation in device memory
static __constant__ DirectSettings_t d_setup;

// Energy and virial in device memory
static __device__ DirectEnergyVirial_t d_energy_virial;

#ifndef USE_TEXTURE_OBJECTS
// VdW parameter texture reference
static texture<float2, 1, cudaReadModeElementType> vdwparam_texref;
static bool vdwparam_texref_bound = false;
static texture<float2, 1, cudaReadModeElementType> vdwparam14_texref;
static bool vdwparam14_texref_bound = false;
#endif

#ifndef USE_TEXTURE_OBJECTS
#define VDWPARAM_TEXREF vdwparam_texref
#define VDWPARAM14_TEXREF vdwparam14_texref
#endif

#include "CudaDirectForce_util.h"

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

__global__ void set_14_list_kernel(const int nin14_tbl, const int* __restrict__ in14_tbl,
				   const xx14_t* __restrict__ in14, xx14list_t* __restrict__ in14list,
				   const int nex14_tbl, const int* __restrict__ ex14_tbl,
				   const xx14_t* __restrict__ ex14, xx14list_t* __restrict__ ex14list,
				   const float4* __restrict__ xyzq,
				   const float3 half_box, const int*__restrict__ glo2loc_ind) {
  int pos = threadIdx.x + blockIdx.x*blockDim.x;
  if (pos < nin14_tbl) {
    int j = in14_tbl[pos];
    xx14_t in14v = in14[j];
    xx14list_t in14listv;
    in14listv.i = glo2loc_ind[in14v.i];
    in14listv.j = glo2loc_ind[in14v.j];
    float4 xyzq_i = xyzq[in14listv.i];
    float4 xyzq_j = xyzq[in14listv.j];
    in14listv.ishift = calc_ishift(xyzq_i, xyzq_j, half_box);
    in14list[pos] = in14listv;
  } else if (pos < nin14_tbl + nex14_tbl) {
    pos -= nin14_tbl;
    int j = ex14_tbl[pos];
    xx14_t ex14v = ex14[j];
    xx14list_t ex14listv;
    ex14listv.i = glo2loc_ind[ex14v.i];
    ex14listv.j = glo2loc_ind[ex14v.j];
    float4 xyzq_i = xyzq[ex14listv.i];
    float4 xyzq_j = xyzq[ex14listv.j];
    ex14listv.ishift = calc_ishift(xyzq_i, xyzq_j, half_box);
    ex14list[pos] = ex14listv;
  }
}

//########################################################################################
//########################################################################################
//########################################################################################

//
// Class creator
//
template <typename AT, typename CT>
CudaPMEDirectForce<AT, CT>::CudaPMEDirectForce() {
  vdwparam = NULL;
  nvdwparam = 0;
  vdwparam_len = 0;
  use_tex_vdwparam = true;
#ifdef USE_TEXTURE_OBJECTS
  vdwparam_tex = 0;
#endif
  vdwparam_texref_bound = false;

  vdwparam14 = NULL;
  nvdwparam14 = 0;
  vdwparam14_len = 0;
  use_tex_vdwparam14 = true;
#ifdef USE_TEXTURE_OBJECTS
  vdwparam14_tex = 0;
#endif
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
CudaPMEDirectForce<AT, CT>::~CudaPMEDirectForce() {
#ifdef USE_TEXTURE_OBJECTS
  if (vdwparam_tex != 0) {
    cudaCheck(cudaDestroyTextureObject(vdwparam_tex));
    vdwparam_tex = 0;
  }
  if (vdwparam14_tex != 0) {
    cudaCheck(cudaDestroyTextureObject(vdwparam14_tex));
    vdwparam14_tex = 0;
  }
#else
  if (vdwparam_texref_bound) {
    cudaCheck(cudaUnbindTexture(VDWPARAM_TEXREF));
    vdwparam_texref_bound = false;
  }
  if (vdwparam14_texref_bound) {
    cudaCheck(cudaUnbindTexture(VDWPARAM14_TEXREF));
    vdwparam14_texref_bound = false;
  }
#endif
  if (vdwparam != NULL) deallocate<CT>(&vdwparam);
  if (vdwparam14 != NULL) deallocate<CT>(&vdwparam14);
  if (in14list != NULL) deallocate<xx14list_t>(&in14list);
  if (ex14list != NULL) deallocate<xx14list_t>(&ex14list);
  if (vdwtype != NULL) deallocate<int>(&vdwtype);
  if (ewald_force != NULL) deallocate<CT>(&ewald_force);
  if (h_energy_virial != NULL) deallocate_host<DirectEnergyVirial_t>(&h_energy_virial);
  if (h_setup != NULL) deallocate_host<DirectSettings_t>(&h_setup);
}

//
// Copies h_setup -> d_setup
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::update_setup() {
  cudaCheck(cudaMemcpyToSymbol(d_setup, h_setup, sizeof(DirectSettings_t)));
}

//
// Sets parameters for the nonbonded computation
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setup(CT boxx, CT boxy, CT boxz, 
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
void CudaPMEDirectForce<AT, CT>::get_box_size(CT &boxx, CT &boxy, CT &boxz) {
  boxx = h_setup->boxx;
  boxy = h_setup->boxy;
  boxz = h_setup->boxz;
}

//
// Sets box sizes
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_box_size(const CT boxx, const CT boxy, const CT boxz) {
  h_setup->boxx = boxx;
  h_setup->boxy = boxy;
  h_setup->boxz = boxz;
  update_setup();
}

//
// Sets "calc_vdw" flag
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_calc_vdw(const bool calc_vdw) {
  this->calc_vdw = calc_vdw;
}

//
// Sets "calc_elec" flag
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_calc_elec(const bool calc_elec) {
  this->calc_elec = calc_elec;
}

//
// Sets VdW parameters by copying them from CPU
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setup_vdwparam(const int type, const int h_nvdwparam, const CT *h_vdwparam) {
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
  copy_HtoD_sync<CT>(h_vdwparam_fixed, *vdwparam_loc, *nvdwparam_loc);
  delete [] h_vdwparam_fixed;

  bool *use_tex_vdwparam_loc = (type == VDW_MAIN) ? &this->use_tex_vdwparam : &this->use_tex_vdwparam14;
#ifdef USE_TEXTURE_OBJECTS
  cudaTextureObject_t *tex = (type == VDW_MAIN) ? &vdwparam_tex : &vdwparam14_tex;
#else
  bool *vdwparam_texref_bound_loc = (type == VDW_MAIN) ? &vdwparam_texref_bound : &vdwparam14_texref_bound;
  texture<float2, 1, cudaReadModeElementType> *vdwparam_texref_loc = 
    (type == VDW_MAIN) ? &VDWPARAM_TEXREF : &VDWPARAM14_TEXREF;
#endif

  if (*use_tex_vdwparam_loc && vdwparam_reallocated) {
#ifdef USE_TEXTURE_OBJECTS
    if (*tex != 0) {
      cudaDestroyTextureObject(*tex);
      *tex = 0;
    }
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = *vdwparam_loc;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = sizeof(CT)*8;
    resDesc.res.linear.desc.y = sizeof(CT)*8;
    resDesc.res.linear.sizeInBytes = (*nvdwparam_loc)*sizeof(CT);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    cudaCheck(cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL));
#else
    // Unbind texture
    if (*vdwparam_texref_bound_loc) {
      cudaCheck(cudaUnbindTexture(*vdwparam_texref_loc));
      *vdwparam_texref_bound_loc = false;
    }
    // Bind texture
    memset(vdwparam_texref_loc, 0, sizeof(*vdwparam_texref_loc));
    vdwparam_texref_loc->normalized = 0;
    vdwparam_texref_loc->filterMode = cudaFilterModePoint;
    vdwparam_texref_loc->addressMode[0] = cudaAddressModeClamp;
    vdwparam_texref_loc->channelDesc.x = sizeof(CT)*8;
    vdwparam_texref_loc->channelDesc.y = sizeof(CT)*8;
    vdwparam_texref_loc->channelDesc.z = 0;
    vdwparam_texref_loc->channelDesc.w = 0;
    vdwparam_texref_loc->channelDesc.f = cudaChannelFormatKindFloat;
    cudaCheck(cudaBindTexture(NULL, *vdwparam_texref_loc, *vdwparam_loc, 
			      (*nvdwparam_loc)*sizeof(CT)));
    *vdwparam_texref_bound_loc = true;
#endif
  }
}

//
// Loads vdwparam from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::load_vdwparam(const char *filename, const int nvdwparam, CT **h_vdwparam) {
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
void CudaPMEDirectForce<AT, CT>::set_vdwparam(const int nvdwparam, const CT *h_vdwparam) {
  setup_vdwparam(VDW_MAIN, nvdwparam, h_vdwparam);
}

//
// Sets VdW parameters by loading them from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam(const int nvdwparam, const char *filename) {  
  CT *h_vdwparam;
  load_vdwparam(filename, nvdwparam, &h_vdwparam);
  setup_vdwparam(VDW_MAIN, nvdwparam, h_vdwparam);
  delete [] h_vdwparam;
}

//
// Sets VdW parameters by copying them from CPU
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam14(const int nvdwparam, const CT *h_vdwparam) {
  setup_vdwparam(VDW_IN14, nvdwparam, h_vdwparam);
}

//
// Sets VdW parameters by loading them from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam14(const int nvdwparam, const char *filename) {  
  CT *h_vdwparam;
  load_vdwparam(filename, nvdwparam, &h_vdwparam);
  setup_vdwparam(VDW_IN14, nvdwparam, h_vdwparam);
  delete [] h_vdwparam;
}

//
// Sets vdwtype array from global list in device memory memory
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype(const int ncoord, const int *glo_vdwtype,
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
void CudaPMEDirectForce<AT, CT>::set_vdwtype(const int ncoord, const int *h_vdwtype) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord-1)/warpsize+1)*warpsize;
  reallocate<int>(&vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);
  copy_HtoD_sync<int>(h_vdwtype, vdwtype, ncoord);
}

//
// Sets vdwtype array by loading it from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype(const int ncoord, const char *filename) {

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
void CudaPMEDirectForce<AT, CT>::set_14_list(int nin14list, int nex14list,
					     xx14list_t* h_in14list, xx14list_t* h_ex14list) {

  this->nin14list = nin14list;
  this->nex14list = nex14list;

  if (nin14list > 0) {
    reallocate<xx14list_t>(&in14list, &in14list_len, nin14list);
    copy_HtoD_sync<xx14list_t>(h_in14list, in14list, nin14list);
  }

  if (nex14list > 0) {
    reallocate<xx14list_t>(&ex14list, &ex14list_len, nex14list);
    copy_HtoD_sync<xx14list_t>(h_ex14list, ex14list, nex14list);
  }

}

//
// Setup 1-4 interaction and exclusion lists from device memory using global data:
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_14_list(const float4 *xyzq,
				      const float boxx, const float boxy, const float boxz,
				      const int *glo2loc_ind,
				      const int nin14_tbl, const int *in14_tbl, const xx14_t *in14,
				      const int nex14_tbl, const int *ex14_tbl, const xx14_t *ex14,
				      cudaStream_t stream) {

  this->nin14list = nin14_tbl;
  if (nin14list > 0) reallocate<xx14list_t>(&in14list, &in14list_len, nin14list, 1.2f);

  this->nex14list = nex14_tbl;
  if (nex14list > 0) reallocate<xx14list_t>(&ex14list, &ex14list_len, nex14list, 1.2f);

  float3 half_box;
  half_box.x = boxx*0.5f;
  half_box.y = boxy*0.5f;
  half_box.z = boxz*0.5f;

  int nthread = 512;
  int nblock = (nin14_tbl + nex14_tbl - 1)/nthread + 1;
  set_14_list_kernel<<< nblock, nthread, 0, stream >>>
    (nin14_tbl, in14_tbl, in14, in14list,
     nex14_tbl, ex14_tbl, ex14, ex14list,
     xyzq, half_box, glo2loc_ind);
  cudaCheck(cudaGetLastError());
}

//
// Builds Ewald lookup table
// roff  = distance cut-off
// h     = the distance between interpolation points
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setup_ewald_force(CT h) {
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
  copy_HtoD_sync<CT>(h_ewald_force, ewald_force, n_ewald_force);

  h_setup->ewald_force = ewald_force;

  delete [] h_ewald_force;
}

//
// Sets method for calculating electrostatic force and energy
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_elec_model(int elec_model, CT h) {
  this->elec_model = elec_model;
  
  if (elec_model == EWALD_LOOKUP) {
    setup_ewald_force(h);
  }
}

//
// Calculates 1-4 exclusions and interactions
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::calc_14_force(const float4 *xyzq,
					       const bool calc_energy, const bool calc_virial,
					       const int stride, AT *force, cudaStream_t stream) {
#ifdef USE_TEXTURE_OBJECTS
  if (vdwparam14_tex == 0) {
    std::cerr << "CudaPMEDirectForce<AT, CT>::calc_14_force, vdwparam14_tex must be created" << std::endl;
    exit(1);
  }
#else
  if (!vdwparam14_texref_bound) {
    std::cerr << "CudaPMEDirectForce<AT, CT>::calc_14_force, vdwparam14_texref must be bound" << std::endl;
    exit(1);
  }
#endif

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

#ifdef USE_TEXTURE_OBJECTS
  CREATE_KERNELS(CREATE_KERNEL14, calc_14_force_kernel, vdwparam14_tex,
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
// Calculates direct force
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::calc_force(const float4 *xyzq,
					    const NeighborList<32>& nlist,
					    const bool calc_energy,
					    const bool calc_virial,
					    const int stride, AT *force, cudaStream_t stream) {

  const int tilesize = 32;

#ifdef USE_TEXTURE_OBJECTS
  if (vdwparam_tex == 0) {
    std::cerr << "CudaPMEDirectForce<AT, CT>::calc_14_force, vdwparam_tex must be created" << std::endl;
    exit(1);
  }
#else
  if (!vdwparam_texref_bound) {
    std::cerr << "CudaPMEDirectForce<AT, CT>::calc_14_force, vdwparam_texref must be bound" << std::endl;
    exit(1);
  }
#endif

  if (nlist.n_ientry == 0) return;
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
  int nblock_tot = (nlist.n_ientry-1)/(nthread/warpsize)+1;

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

#ifdef USE_TEXTURE_OBJECTS
    CREATE_KERNELS(CREATE_KERNEL, calc_force_kernel, vdwparam_tex,
		   base, nlist.n_ientry, nlist.ientry, nlist.tile_indj,
		   nlist.tile_excl, stride, this->vdwparam, this->nvdwparam, xyzq, this->vdwtype,
		   force);
#else
    CREATE_KERNELS(CREATE_KERNEL, calc_force_kernel,
		   base, nlist.n_ientry, nlist.ientry, nlist.tile_indj,
		   nlist.tile_excl, stride, this->vdwparam, this->nvdwparam, xyzq, this->vdwtype,
		   force);
#endif

    base += (nthread/warpsize)*nblock;

    cudaCheck(cudaGetLastError());
  }

}

//
// Calculates virial
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::calc_virial(const int ncoord, const float4 *xyzq,
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
void CudaPMEDirectForce<AT, CT>::clear_energy_virial(cudaStream_t stream) {
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
void CudaPMEDirectForce<AT, CT>::get_energy_virial(bool prev_calc_energy, bool prev_calc_virial,
						   double *energy_vdw, double *energy_elec,
						   double *energy_excl,
						   double *vir) {

  if (prev_calc_energy || prev_calc_virial) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, sizeof(DirectEnergyVirial_t),
				   0, cudaMemcpyDeviceToHost));
  }

  if (prev_calc_virial) {
    for (int i=0;i < 9;i++) {
      h_energy_virial_prev.vir[i] = h_energy_virial->vir[i];
    }
  }

  if (prev_calc_energy) {
    h_energy_virial_prev.energy_vdw  = h_energy_virial->energy_vdw;
    h_energy_virial_prev.energy_elec = h_energy_virial->energy_elec;
    h_energy_virial_prev.energy_excl = h_energy_virial->energy_excl;
  }

  *energy_vdw = h_energy_virial_prev.energy_vdw;
  *energy_elec = h_energy_virial_prev.energy_elec;
  *energy_excl = h_energy_virial_prev.energy_excl;
  for (int i=0;i < 9;i++) {
    vir[i] = h_energy_virial_prev.vir[i];
  }

}

//
// Explicit instances of CudaPMEDirectForce
//
template class CudaPMEDirectForce<long long int, float>;

#ifndef USE_TEXTURE_OBJECTS
#undef VDWPARAM_TEXREF
#undef VDWPARAM14_TEXREF
#endif
