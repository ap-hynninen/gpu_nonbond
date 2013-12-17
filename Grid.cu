#include <iostream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "Matrix3d.h"
#include "MultiNodeMatrix3d.h"
#include "Grid.h"

//
// Grid class
//
// AT  = Accumulation Type
// CT  = Calculation Type (real)
// CT2 = Calculation Type (complex)
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//
// In real space:
// Each instance of grid is responsible for grid region (x0..x1) x (y0..y1) x (z0..z1)
// Note that usually x0=0, x1=nfftx-1
//

template <typename T>
__forceinline__ __device__ void write_grid(const float val, const int ind,
					   T* data) {
  // The generic version can not be used
}

// Template specialization for 64bit integer = "long long int"
template <>
__forceinline__ __device__ void write_grid <long long int> (const float val,
							    const int ind,
							    long long int* data) {
  unsigned long long int qintp = llitoulli(lliroundf(FORCE_SCALE*val));
  atomicAdd((unsigned long long int *)&data[ind], qintp);
}

template <typename AT, typename CT>
__global__ void reduce_data(const int nfft_tot,
			    const AT *data_in,
			    CT *data_out) {
  // The generic version can not be used
}

// Convert "long long int" -> "float"
template <>
__global__ void reduce_data<long long int, float>(const int nfft_tot,
						  const long long int *data_in,
						  float *data_out) {
  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (pos < nfft_tot) {
    long long int val = data_in[pos];
    data_out[pos] = ((float)val)*INV_FORCE_SCALE;
    pos += blockDim.x*gridDim.x;
  }

}

// Convert "long long int" -> "double"
template <>
__global__ void reduce_data<long long int, double>(const int nfft_tot,
						   const long long int *data_in,
						   double *data_out) {
  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (pos < nfft_tot) {
    long long int val = data_in[pos];
    data_out[pos] = ((double)val)*INV_FORCE_SCALE;
    pos += blockDim.x*gridDim.x;
  }

}

/*
//
// Temporary kernels that change the data layout
//
__global__ void change_gridp(const int ncoord, const gridp_t *gridp,
			     int *ixtbl, int *iytbl, int *iztbl, float *charge) {

  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  if (pos < ncoord) {
    gridp_t gridpval = gridp[pos];
    int x = gridpval.x;
    int y = gridpval.y;
    int z = gridpval.z;
    float q = gridpval.q;
    
    ixtbl[pos] = x;
    iytbl[pos] = y;
    iztbl[pos] = z;
    charge[pos] = q;
  }

}
*/

/*
__global__ void change_theta(const int ncoord, const float3 *theta,
			     float4 *thetax, float4 *thetay, float4 *thetaz) {

  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  if (pos < ncoord) {
  thetax[pos].x = theta[pos*4].x;
    thetax[pos].y = theta[pos*4+1].x;
    thetax[pos].z = theta[pos*4+2].x;
    thetax[pos].w = theta[pos*4+3].x;

    thetay[pos].x = theta[pos*4].y;
    thetay[pos].y = theta[pos*4+1].y;
    thetay[pos].z = theta[pos*4+2].y;
    thetay[pos].w = theta[pos*4+3].y;

    thetaz[pos].x = theta[pos*4].z;
    thetaz[pos].y = theta[pos*4+1].z;
    thetaz[pos].z = theta[pos*4+2].z;
    thetaz[pos].w = theta[pos*4+3].z;    
  }

}
*/

//
// Data structure for spread_charge -kernels
//
struct spread_t {
  int ix;
  int iy;
  int iz;
  float thetax[4];
  float thetay[4];
  float thetaz[4];
};

//
// Spreads the charge on the grid
// blockDim.x               = Number of atoms each block loads
// blockDim.y*blockDim.x/64 = Number of atoms we spread at once
//
template <typename AT>
__global__ void
spread_charge_4(const int ncoord,
		const int *ixtbl, const int *iytbl, const int *iztbl, const float *charge,
		const float4 *thetax, const float4 *thetay, const float4 *thetaz,
		const int nfftx, const int nffty, const int nfftz,
		AT* data) {

  // Shared memory
  extern __shared__ spread_t shmem[];

  // Process atoms pos to pos_end-1
  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int pos_end = min((blockIdx.x+1)*blockDim.x, ncoord);

  // Load atom data into shared memory
  if (pos < pos_end) {
    int ix = ixtbl[pos];
    int iy = iytbl[pos];
    int iz = iztbl[pos];
    float q = charge[pos];
    // For each atom we write 4x4x4=64 grid points
    // 3*4 = 12 values per atom stored:
    // theta_sh[i].x = theta_x for grid point i=0...3
    // theta_sh[i].y = theta_y for grid point i=0...3
    // theta_sh[i].z = theta_z for grid point i=0...3
    float4 thetax_tmp = thetax[pos];
    float4 thetay_tmp = thetay[pos];
    float4 thetaz_tmp = thetaz[pos];

    thetax_tmp.x *= q;
    thetax_tmp.y *= q;
    thetax_tmp.z *= q;
    thetax_tmp.w *= q;

    shmem[threadIdx.x].ix = ix;
    shmem[threadIdx.x].iy = iy;
    shmem[threadIdx.x].iz = iz;
    
    shmem[threadIdx.x].thetax[0] = thetax_tmp.x;
    shmem[threadIdx.x].thetax[1] = thetax_tmp.y;
    shmem[threadIdx.x].thetax[2] = thetax_tmp.z;
    shmem[threadIdx.x].thetax[3] = thetax_tmp.w;
    
    shmem[threadIdx.x].thetay[0] = thetay_tmp.x;
    shmem[threadIdx.x].thetay[1] = thetay_tmp.y;
    shmem[threadIdx.x].thetay[2] = thetay_tmp.z;
    shmem[threadIdx.x].thetay[3] = thetay_tmp.w;

    shmem[threadIdx.x].thetaz[0] = thetaz_tmp.x;
    shmem[threadIdx.x].thetaz[1] = thetaz_tmp.y;
    shmem[threadIdx.x].thetaz[2] = thetaz_tmp.z;
    shmem[threadIdx.x].thetaz[3] = thetaz_tmp.w;

  }
  __syncthreads();

  // Grid point location, values of (ix0, iy0, iz0) are in range 0..3
  const int tid = (threadIdx.x + threadIdx.y*blockDim.x) % 64;
  const int x0 = tid & 3;
  const int y0 = (tid >> 2) & 3;
  const int z0 = tid >> 4;

  // Loop over atoms pos..pos_end-1
  int iadd = blockDim.x*blockDim.y/64;
  int i = (threadIdx.x + threadIdx.y*blockDim.x)/64;
  int iend = pos_end - blockIdx.x*blockDim.x;
  for (;i < iend;i += iadd) {
    int x = shmem[i].ix + x0;
    int y = shmem[i].iy + y0;
    int z = shmem[i].iz + z0;
      
    if (x >= nfftx) x -= nfftx;
    if (y >= nffty) y -= nffty;
    if (z >= nfftz) z -= nfftz;
      
    // Get position on the grid
    int ind = x + nfftx*(y + nffty*z);
      
    // Here we unroll the 4x4x4 loop with 64 threads
    // Calculate interpolated charge value and store it to global memory
    write_grid<AT>(shmem[i].thetax[x0]*shmem[i].thetay[y0]*shmem[i].thetaz[z0], ind, data);

  }

}

//
// Performs scalar sum on data(nfft1, nfft2, nfft3)
// T = {float, double}
// T2 = {float2, double2}
//
template <typename T, typename T2>
__global__ void scalar_sum_ortho_kernel(const int nfft1, const int nfft2, const int nfft3,
					const int size1, const int size2, const int size3,
					const int nf1, const int nf2, const int nf3,
					const T recip11, const T recip22, const T recip33,
					const T* prefac1, const T* prefac2, const T* prefac3,
					const T fac, const T piv_inv,
					const bool global_base, T2* data) {
  extern __shared__ T sh_prefac[];

  // Create pointers to shared memory
  T* sh_prefac1 = (T *)&sh_prefac[0];
  T* sh_prefac2 = (T *)&sh_prefac[size1];
  T* sh_prefac3 = (T *)&sh_prefac[size1+size2];

  // Calculate start position (k1, k2, k3) for each thread
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int k3 = tid/(size1*size2);
  tid -= k3*size1*size2;
  int k2 = tid/size1;
  int k1 = tid - k2*size1;

  // Calculate increments (k1_inc, k2_inc, k3_inc)
  int tot_inc = blockDim.x*gridDim.x;
  int k3_inc = tot_inc/(size1*size2);
  tot_inc -= k3_inc*size1*size2;
  int k2_inc = tot_inc/size1;
  int k1_inc = tot_inc - k2_inc*size1;

  // Load prefac data into shared memory
  int pos = threadIdx.x;
  while (pos < size1) {
    sh_prefac1[pos] = prefac1[pos];
    pos += blockDim.x;
  }
  pos = threadIdx.x;
  while (pos < size2) {
    sh_prefac2[pos] = prefac2[pos];
    pos += blockDim.x;
  }
  pos = threadIdx.x;
  while (pos < size3) {
    sh_prefac3[pos] = prefac3[pos];
    pos += blockDim.x;
  }
  __syncthreads();

  while (k3 < size3) {

    int pos = k1 + (k2 + k3*size2)*size1;
    T2 q = data[pos];

    int m1 = k1;
    int m2 = k2;
    int m3 = k3;
    if (k1 >= nf1) m1 -= nfft1;
    if (k2 >= nf2) m2 -= nfft2;
    if (k3 >= nf3) m3 -= nfft3;

    T mhat1 = recip11*m1;
    T mhat2 = recip22*m2;
    T mhat3 = recip33*m3;

    T msq = mhat1*mhat1 + mhat2*mhat2 + mhat3*mhat3;
    T msq_inv = (T)1.0/msq;

    // NOTE: check if it's faster to pre-calculate exp()
    T eterm = exp(-fac*msq)*piv_inv*sh_prefac1[k1]*sh_prefac2[k2]*sh_prefac3[k3]*msq_inv;

    q.x *= eterm;
    q.y *= eterm;
    data[pos] = q;
    
    // Increment position
    k1 += k1_inc;
    if (k1 >= size1) {
      k1 -= size1;
      k2++;
    }
    k2 += k2_inc;
    if (k2 >= size2) {
      k2 -= size2;
      k3++;
    }
    k3 += k3_inc;
  }

  // Set data[0] = 0 for the global (0,0,0)
  if (global_base && (blockIdx.x + threadIdx.x == 0)) {
    T2 zero;
    zero.x = (T)0;
    zero.y = (T)0;
    data[0] = zero;
  }
}

texture<float, 1, cudaReadModeElementType> grid_texref;

// Per atom data structure for the gather_force -kernels
template <typename T>
struct gather_t {
  int ix;
  int iy;
  int iz;
  T charge;
  T thetax[4];
  T thetay[4];
  T thetaz[4];
  T dthetax[4];
  T dthetay[4];
  T dthetaz[4];
  float f1;
  float f2;
  float f3;
};

template <typename T>
__forceinline__ __device__ void write_force_atomic(const float fx,
						   const float fy,
						   const float fz,
						   const int ind,
						   const int stride,
						   const int stride2,
						   T* force) {
  // The generic version can not be used for anything
}

template <typename T>
__forceinline__ __device__ void write_force(const float fx,
					    const float fy,
					    const float fz,
					    const int ind,
					    const int stride,
					    const int stride2,
					    T* force) {
  // The generic version can not be used for anything
}

// Template specialization for 64bit integer = "long long int"
template <>
__forceinline__ __device__ void write_force_atomic <long long int> (const float fx,
								    const float fy,
								    const float fz,
								    const int ind,
								    const int stride,
								    const int stride2,
								    long long int* force) {
  unsigned long long int fx_ulli = llitoulli(lliroundf(FORCE_SCALE*fx));
  unsigned long long int fy_ulli = llitoulli(lliroundf(FORCE_SCALE*fy));
  unsigned long long int fz_ulli = llitoulli(lliroundf(FORCE_SCALE*fz));
  atomicAdd((unsigned long long int *)&force[ind          ], fx_ulli);
  atomicAdd((unsigned long long int *)&force[ind + stride ], fy_ulli);
  atomicAdd((unsigned long long int *)&force[ind + stride2], fz_ulli);
}

// Template specialization for 64bit integer = "long long int"
template <>
__forceinline__ __device__ void write_force <long long int> (const float fx,
							     const float fy,
							     const float fz,
							     const int ind,
							     const int stride,
							     const int stride2,
							     long long int* force) {
  unsigned long long int fx_ulli = llitoulli(lliroundf(FORCE_SCALE*fx));
  unsigned long long int fy_ulli = llitoulli(lliroundf(FORCE_SCALE*fy));
  unsigned long long int fz_ulli = llitoulli(lliroundf(FORCE_SCALE*fz));
  unsigned long long int *force_ulli = (unsigned long long int *)force;
  force_ulli[ind          ] += fx_ulli;
  force_ulli[ind + stride ] += fy_ulli;
  force_ulli[ind + stride2] += fz_ulli;
}

//
// Gathers forces from the grid
// blockDim.x            = Number of atoms each block loads
// blockDim.x*blockDim.y = Total number of threads per block
//
template <typename AT, typename CT>
__global__ void gather_force_4_ortho_kernel(const int ncoord,
					    const int nfftx, const int nffty, const int nfftz,
					    const int xsize, const int ysize, const int zsize,
					    const float recip1, const float recip2, const float recip3,
					    const int *gix, const int *giy, const int *giz,
					    const float *charge,
					    const float4 *thetax, const float4 *thetay,
					    const float4 *thetaz,
					    const float4 *dthetax, const float4 *dthetay,
					    const float4 *dthetaz,
					    const int stride,
					    CT *force) {
  // Shared memory
  extern __shared__ gather_t<CT> shbuf[];

  const int tid = threadIdx.x + threadIdx.y*blockDim.x;
  volatile gather_t<CT> *shmem = shbuf;
  volatile float3 *shred = &((float3 *)&shbuf[blockDim.x])[(tid/8)*8];

  const int pos = blockIdx.x*blockDim.x + threadIdx.x;
  const int pos_end = min((blockIdx.x+1)*blockDim.x, ncoord);

  // Load atom data into shared memory
  if (pos < pos_end) {
    shmem[threadIdx.x].ix = gix[pos];
    shmem[threadIdx.x].iy = giy[pos];
    shmem[threadIdx.x].iz = giz[pos];
    shmem[threadIdx.x].charge = charge[pos];

    float4 tmpx = thetax[pos];
    float4 tmpy = thetay[pos];
    float4 tmpz = thetaz[pos];

    shmem[threadIdx.x].thetax[0] = tmpx.x;
    shmem[threadIdx.x].thetax[1] = tmpx.y;
    shmem[threadIdx.x].thetax[2] = tmpx.z;
    shmem[threadIdx.x].thetax[3] = tmpx.w;

    shmem[threadIdx.x].thetay[0] = tmpy.x;
    shmem[threadIdx.x].thetay[1] = tmpy.y;
    shmem[threadIdx.x].thetay[2] = tmpy.z;
    shmem[threadIdx.x].thetay[3] = tmpy.w;

    shmem[threadIdx.x].thetaz[0] = tmpz.x;
    shmem[threadIdx.x].thetaz[1] = tmpz.y;
    shmem[threadIdx.x].thetaz[2] = tmpz.z;
    shmem[threadIdx.x].thetaz[3] = tmpz.w;

    tmpx = dthetax[pos];
    tmpy = dthetay[pos];
    tmpz = dthetaz[pos];

    shmem[threadIdx.x].dthetax[0] = tmpx.x;
    shmem[threadIdx.x].dthetax[1] = tmpx.y;
    shmem[threadIdx.x].dthetax[2] = tmpx.z;
    shmem[threadIdx.x].dthetax[3] = tmpx.w;

    shmem[threadIdx.x].dthetay[0] = tmpy.x;
    shmem[threadIdx.x].dthetay[1] = tmpy.y;
    shmem[threadIdx.x].dthetay[2] = tmpy.z;
    shmem[threadIdx.x].dthetay[3] = tmpy.w;

    shmem[threadIdx.x].dthetaz[0] = tmpz.x;
    shmem[threadIdx.x].dthetaz[1] = tmpz.y;
    shmem[threadIdx.x].dthetaz[2] = tmpz.z;
    shmem[threadIdx.x].dthetaz[3] = tmpz.w;

  }
  __syncthreads();

  
  // Calculate the index this thread is calculating
  const int tx = 0;             // 0
  const int ty = (tid & 1);     // 0, 1
  const int tz = (tid/2) & 3;   // 0, 1, 2, 3

  // Calculate force by looping 64/8=8 times
  int base = tid/8;
  const int base_end = pos_end - blockIdx.x*blockDim.x;
  while (base < base_end) {
    int ix0 = shmem[base].ix + tx;
    int iy0 = shmem[base].iy + ty;
    int iz0 = shmem[base].iz + tz;

    int ix1 = ix0 + 1;
    int ix2 = ix0 + 2;
    int ix3 = ix0 + 3;

    int iy1 = iy0 + 2;

    if (ix0 >= nfftx) ix0 -= nfftx;
    if (iy0 >= nffty) iy0 -= nffty;
    if (iz0 >= nfftz) iz0 -= nfftz;

    if (ix1 >= nfftx) ix1 -= nfftx;
    if (ix2 >= nfftx) ix2 -= nfftx;
    if (ix3 >= nfftx) ix3 -= nfftx;

    if (iy1 >= nffty) iy1 -= nffty;

    float q0 = tex1Dfetch(grid_texref, ix0 + (iy0 + iz0*ysize)*xsize);
    float q1 = tex1Dfetch(grid_texref, ix1 + (iy0 + iz0*ysize)*xsize);
    float q2 = tex1Dfetch(grid_texref, ix2 + (iy0 + iz0*ysize)*xsize);
    float q3 = tex1Dfetch(grid_texref, ix3 + (iy0 + iz0*ysize)*xsize);
    float q4 = tex1Dfetch(grid_texref, ix0 + (iy1 + iz0*ysize)*xsize);
    float q5 = tex1Dfetch(grid_texref, ix1 + (iy1 + iz0*ysize)*xsize);
    float q6 = tex1Dfetch(grid_texref, ix2 + (iy1 + iz0*ysize)*xsize);
    float q7 = tex1Dfetch(grid_texref, ix3 + (iy1 + iz0*ysize)*xsize);

    float thx0 = shmem[base].thetax[tx+0];
    float thx1 = shmem[base].thetax[tx+1];
    float thx2 = shmem[base].thetax[tx+2];
    float thx3 = shmem[base].thetax[tx+3];
    float thy0 = shmem[base].thetay[ty];
    float thy1 = shmem[base].thetay[ty+2];
    float thz0 = shmem[base].thetaz[tz];

    float dthx0 = shmem[base].dthetax[tx+0];
    float dthx1 = shmem[base].dthetax[tx+1];
    float dthx2 = shmem[base].dthetax[tx+2];
    float dthx3 = shmem[base].dthetax[tx+3];
    float dthy0 = shmem[base].dthetay[ty];
    float dthy1 = shmem[base].dthetay[ty+2];
    float dthz0 = shmem[base].dthetaz[tz];

    float thy0_thz0  = thy0 * thz0;
    float dthy0_thz0 = dthy0 * thz0;
    float thy0_dthz0 = thy0 * dthz0;

    float thy1_thz0  = thy1 * thz0;
    float dthy1_thz0 = dthy1 * thz0;
    float thy1_dthz0 = thy1 * dthz0;

    float f1 = dthx0 * thy0_thz0 * q0;
    float f2 = thx0 * dthy0_thz0 * q0;
    float f3 = thx0 * thy0_dthz0 * q0;

    f1 += dthx1 * thy0_thz0 * q1;
    f2 += thx1 * dthy0_thz0 * q1;
    f3 += thx1 * thy0_dthz0 * q1;

    f1 += dthx2 * thy0_thz0 * q2;
    f2 += thx2 * dthy0_thz0 * q2;
    f3 += thx2 * thy0_dthz0 * q2;

    f1 += dthx3 * thy0_thz0 * q3;
    f2 += thx3 * dthy0_thz0 * q3;
    f3 += thx3 * thy0_dthz0 * q3;

    f1 += dthx0 * thy1_thz0 * q4;
    f2 += thx0 * dthy1_thz0 * q4;
    f3 += thx0 * thy1_dthz0 * q4;

    f1 += dthx1 * thy1_thz0 * q5;
    f2 += thx1 * dthy1_thz0 * q5;
    f3 += thx1 * thy1_dthz0 * q5;

    f1 += dthx2 * thy1_thz0 * q6;
    f2 += thx2 * dthy1_thz0 * q6;
    f3 += thx2 * thy1_dthz0 * q6;

    f1 += dthx3 * thy1_thz0 * q7;
    f2 += thx3 * dthy1_thz0 * q7;
    f3 += thx3 * thy1_dthz0 * q7;

    // Reduce
    const int i = threadIdx.x & 7;
    shred[i].x = f1;
    shred[i].y = f2;
    shred[i].z = f3;

    if (i < 4) {
      shred[i].x += shred[i+4].x;
      shred[i].y += shred[i+4].y;
      shred[i].z += shred[i+4].z;
    }

    if (i < 2) {
      shred[i].x += shred[i+2].x;
      shred[i].y += shred[i+2].y;
      shred[i].z += shred[i+2].z;
    }

    if (i == 0) {
      shmem[base].f1 = shred[0].x + shred[1].x;
      shmem[base].f2 = shred[0].y + shred[1].y;
      shmem[base].f3 = shred[0].z + shred[1].z;
    }

    base += 8;
  }

  // Write forces
  const int stride2 = 2*stride;
  __syncthreads();
  if (pos < pos_end) {
    float f1 = shmem[threadIdx.x].f1;
    float f2 = shmem[threadIdx.x].f2;
    float f3 = shmem[threadIdx.x].f3;
    float q = shmem[threadIdx.x].charge;
    float fx = q*recip1*f1;
    float fy = q*recip2*f2;
    float fz = q*recip3*f3;
    force[pos]         = fx;
    force[pos+stride]  = fy;
    force[pos+stride2] = fz;
  }

}

template<typename T>
void bind_grid_texture(const T *data, const int data_len) {
  std::cerr << "Fatal error: cannot bind generic data type" << std::endl;
  exit(1);
}

template<>
void bind_grid_texture<float>(const float *data, const int data_len) {
  grid_texref.normalized = 0;
  grid_texref.filterMode = cudaFilterModePoint;
  grid_texref.addressMode[0] = cudaAddressModeClamp;
  grid_texref.channelDesc.x = 32;
  grid_texref.channelDesc.y = 0;
  grid_texref.channelDesc.z = 0;
  grid_texref.channelDesc.w = 0;
  grid_texref.channelDesc.f = cudaChannelFormatKindFloat;
  cudaCheck(cudaBindTexture(NULL, grid_texref, data, data_len*sizeof(float)));  
}

//
// Initializer
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::init(int x0, int x1, int y0, int y1, int z0, int z1, int order, 
			     bool y_land_locked, bool z_land_locked) {
  
  this->x0 = x0;
  this->x1 = x1;
  
  this->y0 = y0;
  this->y1 = y1;
  
  this->z0 = z0;
  this->z1 = z1;
  
  this->order = order;
  
  xlo = x0;
  xhi = x1;

  ylo = y0;
  yhi = y1;

  zlo = z0;
  zhi = z1;

  /*
  xhi += (order-1);

  if (y_land_locked) ylo -= (order-1);
  yhi += (order-1);
  
  if (z_land_locked) zlo -= (order-1);
  zhi += (order-1);
  */

  xsize = xhi - xlo + 1;
  ysize = yhi - ylo + 1;
  zsize = zhi - zlo + 1;

  data_size = (2*(xsize/2+1))*ysize*zsize;

  make_fft_plans();

  // data1 is used for accumulation, make sure it has enough space
  allocate<CT>(&data1, data_size*sizeof(AT)/sizeof(CT));
  allocate<CT>(&data2, data_size);

  if (multi_gpu) {
#if CUDA_VERSION >= 6000
    cufftCheck(cufftXtMalloc(r2c_plan, &multi_data, CUFFT_XT_FORMAT_INPLACE));
    host_data = new CT2[xsize*ysize*zsize];
    host_tmp = new CT[2*(xsize/2+1)*ysize*zsize];
#else
    std::cerr << "No Multi-gpu FFT support in CUDA versions below 6.0" << std::endl;
    exit(1);
#endif
  }

  data1_len = data_size*sizeof(AT)/sizeof(CT);
  data2_len = data_size;

  accum_grid  = new Matrix3d<AT>(xsize, ysize, zsize, xsize, ysize, zsize, (AT *)data1);
  charge_grid = new Matrix3d<CT>(xsize, ysize, zsize, xsize, ysize, zsize, (CT *)data2);

  if (fft_type == COLUMN) {
    xfft_grid   = new Matrix3d<CT2>(xsize/2+1, ysize, zsize, xsize/2+1, ysize, zsize, (CT2 *)data2);
    yfft_grid   = new Matrix3d<CT2>(ysize, zsize, xsize/2+1, ysize, zsize, xsize/2+1, (CT2 *)data1);
    zfft_grid   = new Matrix3d<CT2>(zsize, xsize/2+1, ysize, zsize, xsize/2+1, ysize, (CT2 *)data2);
    solved_grid = new Matrix3d<CT>(xsize, ysize, zsize, xsize, ysize, zsize, (CT *)data2);
  } else if (fft_type == SLAB) {
    xyfft_grid = new Matrix3d<CT2>(xsize/2+1, ysize, zsize, xsize/2+1, ysize, zsize, (CT2 *)data2);
    zfft_grid   = new Matrix3d<CT2>(zsize, xsize/2+1, ysize, zsize, xsize/2+1, ysize, (CT2 *)data1);
    solved_grid = new Matrix3d<CT>(xsize, ysize, zsize, xsize, ysize, zsize, (CT *)data2);
  } else if (fft_type == BOX) {
    fft_grid = new Matrix3d<CT2>(xsize/2+1, ysize, zsize, xsize/2+1, ysize, zsize, (CT2 *)data2);
    solved_grid = new Matrix3d<CT>(xsize, ysize, zsize, xsize, ysize, zsize, (CT *)data2);
  }

  // Bind grid_texture to solved_grid->data (data2)
  bind_grid_texture<CT>(solved_grid->data, xsize*ysize*zsize);

}

//
// Class creator 
//
template <typename AT, typename CT, typename CT2>
Grid<AT, CT, CT2>::Grid(int nfftx, int nffty, int nfftz, int order,
			FFTtype fft_type=COLUMN,
			int nnode=1,
			int mynode=0) : nfftx(nfftx), nffty(nffty), nfftz(nfftz), fft_type(fft_type) {

    assert(nnode >= 1);
    assert(mynode >= 0 && mynode < nnode);
    assert(sizeof(AT) >= sizeof(CT));

    int nnode_y, nnode_z;

    if (fft_type == COLUMN) {
      nnode_y = max(1,(int)ceil( sqrt( (double)(nnode*nffty) / (double)(nfftz) )));
      nnode_z = nnode/nnode_y;
      while (nnode_y*nnode_z != nnode) {
	nnode_y = nnode_y - 1;
	nnode_z = nnode/nnode_y;
      }
    } else if (fft_type == SLAB) {
      nnode_y = 1;
      nnode_z = nnode;
      assert(nfftz/nnode_z >= 1);
    } else if (fft_type == BOX) {
      assert(nnode == 1);
      nnode_y = 1;
      nnode_z = 1;
    } else {
      std::cerr<<"Grid::fft_type invalid"<<std::endl;
      exit(1);
    }

    // We have nodes nnode_y * nnode_z. Get y and z index of this node:
    int inode_y = mynode % nnode_y;
    int inode_z = mynode/nnode_y;

    assert(nnode_y != 0);
    assert(nnode_z != 0);

    int x0 = 0;
    int x1 = nfftx-1;
      
    int y0 = inode_y*nffty/nnode_y;
    int y1 = (inode_y+1)*nffty/nnode_y - 1;

    int z0 = inode_z*nfftz/nnode_z;
    int z1 = (inode_z+1)*nfftz/nnode_z - 1;

    bool y_land_locked = (inode_y-1 >= 0) && (inode_y+1 < nnode_y);
    bool z_land_locked = (inode_z-1 >= 0) && (inode_z+1 < nnode_z);

    multi_gpu = false;

    assert((multi_gpu && fft_type==BOX) || !multi_gpu);

    init(x0, x1, y0, y1, z0, z1, order, y_land_locked, z_land_locked);
}

//
// Create FFT plans
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::make_fft_plans() {

  if (fft_type == COLUMN) {
    // Set the size of the local FFT transforms
    int batch;
    int nfftx_local = x1 - x0 + 1;
    int nffty_local = y1 - y0 + 1;
    int nfftz_local = z1 - z0 + 1;
    
    batch = nffty_local * nfftz_local;
    cufftCheck(cufftPlanMany(&x_r2c_plan, 1, &nfftx_local,
			     NULL, 0, 0,
			     NULL, 0, 0, 
			     CUFFT_R2C, batch));
    cufftCheck(cufftSetCompatibilityMode(x_r2c_plan, CUFFT_COMPATIBILITY_NATIVE));
    
    batch = nfftz_local*(nfftx_local/2+1);
    cufftCheck(cufftPlanMany(&y_c2c_plan, 1, &nffty_local,
			     NULL, 0, 0,
			     NULL, 0, 0, 
			     CUFFT_C2C, batch));
    cufftCheck(cufftSetCompatibilityMode(y_c2c_plan, CUFFT_COMPATIBILITY_NATIVE));
    
    batch = (nfftx_local/2+1)*nffty_local;
    cufftCheck(cufftPlanMany(&z_c2c_plan, 1, &nfftz_local,
			     NULL, 0, 0,
			     NULL, 0, 0, 
			     CUFFT_C2C, batch));
    cufftCheck(cufftSetCompatibilityMode(z_c2c_plan, CUFFT_COMPATIBILITY_NATIVE));
    
    batch = nffty_local*nfftz_local;
    cufftCheck(cufftPlanMany(&x_c2r_plan, 1, &nfftx_local,
			     NULL, 0, 0,
			     NULL, 0, 0, 
			     CUFFT_C2R, batch));
    cufftCheck(cufftSetCompatibilityMode(x_c2r_plan, CUFFT_COMPATIBILITY_NATIVE));
  } else if (fft_type == SLAB) {
    int batch;
    int nfftx_local = x1 - x0 + 1;
    int nffty_local = y1 - y0 + 1;
    int nfftz_local = z1 - z0 + 1;

    int n[2] = {nffty_local, nfftx_local};

    batch = nfftz_local;
    cufftCheck(cufftPlanMany(&xy_r2c_plan, 2, n,
			     NULL, 0, 0,
			     NULL, 0, 0, 
			     CUFFT_R2C, batch));
    cufftCheck(cufftSetCompatibilityMode(xy_r2c_plan, CUFFT_COMPATIBILITY_NATIVE));

    batch = (nfftx_local/2+1)*nffty_local;
    cufftCheck(cufftPlanMany(&z_c2c_plan, 1, &nfftz_local,
			     NULL, 0, 0,
			     NULL, 0, 0, 
			     CUFFT_C2C, batch));
    cufftCheck(cufftSetCompatibilityMode(z_c2c_plan, CUFFT_COMPATIBILITY_NATIVE));

    batch = nfftz_local;
    cufftCheck(cufftPlanMany(&xy_c2r_plan, 2, n,
			     NULL, 0, 0,
			     NULL, 0, 0, 
			     CUFFT_C2R, batch));
    cufftCheck(cufftSetCompatibilityMode(xy_c2r_plan, CUFFT_COMPATIBILITY_NATIVE));
    
  } else if (fft_type == BOX) {
    if (multi_gpu) {
#if CUDA_VERSION >= 6000
      cufftCheck(cufftCreate(&r2c_plan));
      cufftCheck(cufftCreate(&c2r_plan));
      int ngpu = 2;
      int gpu[2] = {2, 3};
      cufftCheck(cufftXtSetGPUs(r2c_plan, ngpu, gpu));
      cufftCheck(cufftXtSetGPUs(c2r_plan, ngpu, gpu));

      size_t worksize_r2c[2];
      size_t worksize_c2r[2];

      cufftCheck(cufftMakePlan3d(r2c_plan, nfftz, nffty, nfftx, CUFFT_C2C, worksize_r2c));
      cufftCheck(cufftMakePlan3d(c2r_plan, nfftz, nffty, nfftx, CUFFT_C2C, worksize_c2r));
#endif
    } else {
      cufftCheck(cufftPlan3d(&r2c_plan, nfftz, nffty, nfftx, CUFFT_R2C));
      cufftCheck(cufftSetCompatibilityMode(r2c_plan, CUFFT_COMPATIBILITY_NATIVE));

      cufftCheck(cufftPlan3d(&c2r_plan, nfftz, nffty, nfftx, CUFFT_C2R));
      cufftCheck(cufftSetCompatibilityMode(c2r_plan, CUFFT_COMPATIBILITY_NATIVE));
    }
  }

}

//
// Class destructor
//
template <typename AT, typename CT, typename CT2>
Grid<AT, CT, CT2>::~Grid() {

  // Unbind grid texture
  cudaCheck(cudaUnbindTexture(grid_texref));

  delete accum_grid;
  delete charge_grid;
  delete solved_grid;
  deallocate<CT>(&data1);
  deallocate<CT>(&data2);

#if CUDA_VERSION >= 6000
  if (multi_gpu) {
    delete [] host_data;
    delete [] host_tmp;
    cufftCheck(cufftXtFree(multi_data));
  }
#endif

  if (fft_type == COLUMN) {
    delete xfft_grid;
    delete yfft_grid;
    delete zfft_grid;
    cufftCheck(cufftDestroy(x_r2c_plan));
    cufftCheck(cufftDestroy(y_c2c_plan));
    cufftCheck(cufftDestroy(z_c2c_plan));
    cufftCheck(cufftDestroy(x_c2r_plan));
  } else if (fft_type == SLAB) {
    delete xyfft_grid;
    delete zfft_grid;
    cufftCheck(cufftDestroy(xy_r2c_plan));
    cufftCheck(cufftDestroy(z_c2c_plan));
    cufftCheck(cufftDestroy(xy_c2r_plan));
  } else if (fft_type == BOX) {
    delete fft_grid;
    cufftCheck(cufftDestroy(r2c_plan));
    cufftCheck(cufftDestroy(c2r_plan));
  }
}

template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::print_info() {
  std::cout << "fft_type = ";
  if (fft_type == COLUMN) {
    std::cout << "COLUMN" << std::endl;
  } else if (fft_type == SLAB) {
    std::cout << "SLAB" << std::endl;
  } else {
    std::cout << "BOX" << std::endl;
  }
  std::cout << "order = " << order << std::endl;
  std::cout << "nfftx, nffty, nfftz = " << nfftx << " " << nffty << " " << nfftz << std::endl;
  std::cout << "x0...x1   = " << x0 << " ... " << x1 << std::endl;
  std::cout << "y0...y1   = " << y0 << " ... " << y1 << std::endl;
  std::cout << "z0...z1   = " << z0 << " ... " << z1 << std::endl;
  std::cout << "xlo...xhi = " << xlo << " ... " << xhi << std::endl;
  std::cout << "ylo...yhi = " << ylo << " ... " << yhi << std::endl;
  std::cout << "zlo...zhi = " << zlo << " ... " << zhi << std::endl;
  std::cout << "xsize, ysize, zsize = " << xsize << " " << ysize << " " << zsize << std::endl;
  std::cout << "data_size = " << data_size << std::endl;
}

template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::spread_charge(const int ncoord, const Bspline<CT> &bspline) {

  clear_gpu_array<AT>((AT *)accum_grid->data, xsize*ysize*zsize);

  dim3 nthread, nblock;

  nthread.x = 32;
  nthread.y = 4;
  nthread.z = 1;
  nblock.x = (ncoord - 1)/nthread.x + 1;
  nblock.y = 1;
  nblock.z = 1;

  size_t shmem_size = sizeof(spread_t)*nthread.x;

  switch(order) {
  case 4:
    spread_charge_4<AT> <<< nblock, nthread, shmem_size >>>(ncoord, 
							    bspline.gix, bspline.giy, bspline.giz,
							    bspline.charge,
							    (float4 *)bspline.thetax,
							    (float4 *)bspline.thetay,
							    (float4 *)bspline.thetaz,
							    nfftx, nffty, nfftz,
							    (AT *)accum_grid->data);
    break;

  default:
    std::cerr<<"Grid::spread_charge: order "<<order<<" not implemented"<<std::endl;
    exit(1);
  }
  cudaCheck(cudaGetLastError());

  // Reduce charge data back to a float/double value
  nthread.x = 512;
  nthread.y = 1;
  nthread.z = 1;
  nblock.x = (nfftx*nffty*nfftz - 1)/nthread.x + 1;
  nblock.y = 1;
  nblock.z = 1;
  reduce_data<AT, CT> <<< nblock, nthread >>>(xsize*ysize*zsize,
					      (AT *)accum_grid->data,
					      charge_grid->data);
  cudaCheck(cudaGetLastError());

}

//
// Perform scalar sum without calculating virial or energy (faster)
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::scalar_sum(const double *recip, const double kappa,
				   CT* prefac_x, CT* prefac_y, CT* prefac_z) {

  const double pi = 3.14159265358979323846;
  int nthread = 512;
  int nblock = 10;
  int shmem_size = sizeof(CT)*(nfftx + nffty + nfftz);

  int nfft1, nfft2, nfft3;
  int size1, size2, size3;
  CT *prefac1, *prefac2, *prefac3;
  CT recip1, recip2, recip3;
  CT2 *datap;

  if (fft_type == COLUMN || fft_type == SLAB) {
    nfft1 = nfftz;
    nfft2 = nfftx;
    nfft3 = nffty;
    size1 = nfftz;
    size2 = nfftx/2+1;
    size3 = nffty;
    prefac1 = prefac_z;
    prefac2 = prefac_x;
    prefac3 = prefac_y;
    recip1 = (CT)recip[8];
    recip2 = (CT)recip[0];
    recip3 = (CT)recip[4];
    datap = zfft_grid->data;
  } else if (fft_type == BOX) {
    nfft1 = nfftx;
    nfft2 = nffty;
    nfft3 = nfftz;
    size1 = nfftx/2+1;
    size2 = nffty;
    size3 = nfftz;
    prefac1 = prefac_x;
    prefac2 = prefac_y;
    prefac3 = prefac_z;
    recip1 = (CT)recip[0];
    recip2 = (CT)recip[4];
    recip3 = (CT)recip[8];
    datap = fft_grid->data;
  }

  bool ortho = (recip[1] == 0.0 && recip[2] == 0.0 && recip[3] == 0.0 &&
		recip[5] == 0.0 && recip[6] == 0.0 && recip[7] == 0.0);

  double inv_vol = recip[0]*recip[4]*recip[8];
  CT piv_inv = (CT)(inv_vol/pi);
  CT fac = (CT)(pi*pi/(kappa*kappa));

  bool global_base = (x0 == 0 && y0 == 0 && z0 == 0);

  int nf1 = nfft1/2 + (nfft1 % 2);
  int nf2 = nfft2/2 + (nfft2 % 2);
  int nf3 = nfft3/2 + (nfft3 % 2);

  if (ortho) {
    scalar_sum_ortho_kernel<CT, CT2>
      <<< nblock, nthread, shmem_size >>> (nfft1, nfft2, nfft3,
					   size1, size2, size3,
					   nf1, nf2, nf3,
					   recip1, recip2, recip3,
					   prefac1, prefac2, prefac3,
					   fac, piv_inv, global_base, datap);
  } else {
    std::cerr<<"Grid::scalar_sum: only orthorombic boxes are currently supported"<<std::endl;
    exit(1);
  }

  cudaCheck(cudaGetLastError());

}

//
// Gathers forces from the grid
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::gather_force(const int ncoord, const double* recip,
				     const Bspline<CT> &bspline,
				     const int stride, CT* force) {

  dim3 nthread(32, 2, 1);
  dim3 nblock((ncoord - 1)/nthread.x + 1, 1, 1);
  size_t shmem_size = sizeof(gather_t<CT>)*nthread.x + sizeof(float3)*nthread.x*nthread.y;

  // CCELEC is 1/ (4 pi eps ) in AKMA units, conversion from SI
  // units: CCELEC = e*e*Na / (4*pi*eps*1Kcal*1A)
  //
  //      parameter :: CCELEC=332.0636D0 ! old value of dubious origin
  //      parameter :: CCELEC=331.843D0  ! value from 1986-1987 CRC Handbook
  //                                   ! of Chemistry and Physics
  //  real(chm_real), parameter ::  &
  //       CCELEC_amber    = 332.0522173D0, &
  //       CCELEC_charmm   = 332.0716D0   , &
  //       CCELEC_discover = 332.054D0    , &
  //       CCELEC_namd     = 332.0636D0   

  const double ccelec = 332.0716;

  CT recip_loc[9];
  recip_loc[0] = (CT)(recip[0]*(double)nfftx*ccelec);
  recip_loc[1] = (CT)(recip[1]*(double)nfftx*ccelec);
  recip_loc[2] = (CT)(recip[2]*(double)nfftx*ccelec);
  recip_loc[3] = (CT)(recip[3]*(double)nffty*ccelec);
  recip_loc[4] = (CT)(recip[4]*(double)nffty*ccelec);
  recip_loc[5] = (CT)(recip[5]*(double)nffty*ccelec);
  recip_loc[6] = (CT)(recip[6]*(double)nfftz*ccelec);
  recip_loc[7] = (CT)(recip[7]*(double)nfftz*ccelec);
  recip_loc[8] = (CT)(recip[8]*(double)nfftz*ccelec);

  bool ortho = (recip[1] == 0.0 && recip[2] == 0.0 && recip[3] == 0.0 &&
		recip[5] == 0.0 && recip[6] == 0.0 && recip[7] == 0.0);

  if (ortho) {
    switch(order) {
    case 4:
      gather_force_4_ortho_kernel<AT, CT> 
	<<< nblock, nthread, shmem_size >>>(ncoord,
					    nfftx, nffty, nfftz,
					    nfftx, nffty, nfftz,
					    recip_loc[0], recip_loc[4], recip_loc[8],
					    bspline.gix, bspline.giy, bspline.giz, bspline.charge,
					    (float4 *)bspline.thetax,
					    (float4 *)bspline.thetay,
					    (float4 *)bspline.thetaz,
					    (float4 *)bspline.dthetax,
					    (float4 *)bspline.dthetay,
					    (float4 *)bspline.dthetaz,
					    stride, force);
      break;
    default:
      std::cerr<<"Grid::gather_force: order "<<order<<" not implemented"<<std::endl;
      exit(1);
    }
  } else {
      std::cerr<<"Grid::gather_force: only orthorombic boxes are currently supported"<<std::endl;
      std::cerr<<recip[1]<<std::endl;
      std::cerr<<recip[2]<<std::endl;
      std::cerr<<recip[3]<<std::endl;
      std::cerr<<recip[5]<<std::endl;
      std::cerr<<recip[6]<<std::endl;
      std::cerr<<recip[7]<<std::endl;
      exit(1);    
  }

  cudaCheck(cudaGetLastError());
}

//
// FFT x coordinate Real -> Complex
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::x_fft_r2c(CT2 *data) {

  if (fft_type == COLUMN) {
    cufftCheck(cufftExecR2C(x_r2c_plan,
			    (cufftReal *)data,
			    (cufftComplex *)data));
  } else {
    std::cerr << "Grid::x_fft_r2c, only COLUMN type FFT can call this function" << std::endl;
    exit(1);
  }

}

//
// FFT x coordinate Complex -> Real
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::x_fft_c2r(CT2 *data) {

  if (fft_type == COLUMN) {
    cufftCheck(cufftExecC2R(x_c2r_plan,
			    (cufftComplex *)data,
			    (cufftReal *)data));
  } else {
    std::cerr << "Grid::x_fft_r2c, only COLUMN type FFT can call this function" << std::endl;
    exit(1);
  }

}

//
// FFT y coordinate Complex -> Complex
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::y_fft_c2c(CT2 *data, const int direction) {

  if (fft_type == COLUMN) {
    cufftCheck(cufftExecC2C(y_c2c_plan,
			    (cufftComplex *)data,
			    (cufftComplex *)data,
			    direction));
  } else {
    std::cerr << "Grid::x_fft_r2c, only COLUMN type FFT can call this function" << std::endl;
    exit(1);
  }

}

//
// FFT z coordinate Complex -> Complex
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::z_fft_c2c(CT2 *data, const int direction) {

  if (fft_type == COLUMN) {
    cufftCheck(cufftExecC2C(z_c2c_plan,
			    (cufftComplex *)data,
			    (cufftComplex *)data,
			    direction));
  } else {
    std::cerr << "Grid::x_fft_r2c, only COLUMN type FFT can call this function" << std::endl;
    exit(1);
  }

}

//
// 3D FFT Real -> Complex
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::r2c_fft() {

  if (fft_type == COLUMN) {
    // data2(x, y, z)
    x_fft_r2c(xfft_grid->data);
    xfft_grid->transpose_xyz_yzx(yfft_grid);

    // data1(y, z, x)
    y_fft_c2c(yfft_grid->data, CUFFT_FORWARD);
    yfft_grid->transpose_xyz_yzx(zfft_grid);

    // data2(z, x, y)
    z_fft_c2c(zfft_grid->data, CUFFT_FORWARD);
  } else if (fft_type == SLAB) {
    cufftCheck(cufftExecR2C(xy_r2c_plan,
			    (cufftReal *)charge_grid->data,
			    (cufftComplex *)xyfft_grid->data));
    xyfft_grid->transpose_xyz_zxy(zfft_grid);
    cufftCheck(cufftExecC2C(z_c2c_plan,
			    (cufftComplex *)zfft_grid->data,
			    (cufftComplex *)zfft_grid->data, CUFFT_FORWARD));
  } else if (fft_type == BOX) {
    if (multi_gpu) {
#if CUDA_VERSION >= 6000
      // Transform from Real -> Complex
      cudaCheck(cudaMemcpy(host_tmp, charge_grid->data, sizeof(CT)*xsize*ysize*zsize,
			   cudaMemcpyDeviceToHost));
      for (int z=0;z < zsize;z++)
	for (int y=0;y < ysize;y++)
	  for (int x=0;x < xsize;x++) {
	    host_data[x + (y + z*ysize)*xsize].x = host_tmp[x + (y + z*ysize)*xsize];
	    host_data[x + (y + z*ysize)*xsize].y = 0;
	  }


      cufftCheck(cufftXtMemcpy(r2c_plan, multi_data, host_data, CUFFT_COPY_HOST_TO_DEVICE));
      cufftCheck(cufftXtExecDescriptorC2C(r2c_plan,
					  multi_data,
					  multi_data, CUFFT_FORWARD));

      // Copy data back to a single GPU buffer in fft_grid->data
      cufftCheck(cufftXtMemcpy(r2c_plan, host_data, multi_data, CUFFT_COPY_DEVICE_TO_HOST));

      CT2 *tmp = (CT2 *)host_tmp;
      for (int z=0;z < zsize;z++)
	for (int y=0;y < ysize;y++)
	  for (int x=0;x < xsize;x++) {
	    tmp[x + (y + z*ysize)*(xsize/2+1)].x = host_data[x + (y + z*ysize)*xsize].x;
	    tmp[x + (y + z*ysize)*(xsize/2+1)].y = host_data[x + (y + z*ysize)*xsize].y;
	  }

      cudaCheck(cudaMemcpy(fft_grid->data, tmp, sizeof(CT2)*(xsize/2+1)*ysize*zsize,
			   cudaMemcpyHostToDevice));
#endif
    } else {
      cufftCheck(cufftExecR2C(r2c_plan,
			      (cufftReal *)charge_grid->data,
			      (cufftComplex *)fft_grid->data));
    }
  }

}

//
// 3D FFT Complex -> Real
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::c2r_fft() {

  if (fft_type == COLUMN) {
    // data2(z, x, y)
    z_fft_c2c(zfft_grid->data, CUFFT_INVERSE);
    zfft_grid->transpose_xyz_zxy(yfft_grid);

    // data1(y, x, z)
    y_fft_c2c(yfft_grid->data, CUFFT_INVERSE);
    yfft_grid->transpose_xyz_zxy(xfft_grid);

    // data2(x, y, z)
    x_fft_c2r(xfft_grid->data);
  } else if (fft_type == SLAB) {
    cufftCheck(cufftExecC2C(z_c2c_plan,
			    (cufftComplex *)zfft_grid->data,
			    (cufftComplex *)zfft_grid->data, CUFFT_INVERSE));
    zfft_grid->transpose_xyz_yzx(xyfft_grid);
    cufftCheck(cufftExecC2R(xy_c2r_plan,
			    (cufftComplex *)xyfft_grid->data,
			    (cufftReal *)xyfft_grid->data));
  } else if (fft_type == BOX) {
    cufftCheck(cufftExecC2R(c2r_plan,
			    (cufftComplex *)fft_grid->data,
			    (cufftReal *)fft_grid->data));
  }

}

//
// Explicit instances of Grid
//
template class Grid<long long int, float, float2>;
