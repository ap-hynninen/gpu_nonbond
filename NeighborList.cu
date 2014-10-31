#include <stdio.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "NeighborList.h"

// IF defined, uses strict (Factor = 1.0f) memory reallocation. Used for debuggin memory problems.
#define STRICT_MEMORY_REALLOC

static __device__ NeighborListParam_t d_nlist_param;

static int BitCount(unsigned int u)
 {
         unsigned int uCount;

         uCount = u
                  - ((u >> 1) & 033333333333)
                  - ((u >> 2) & 011111111111);
         return
           ((uCount + (uCount >> 3))
            & 030707070707) % 63;
 }

/*
static int BitCount_ref(unsigned int u) {
  unsigned int x = u;
  int res = 0;
  while (x != 0) {
    res += (x & 1);
    x >>= 1;
  }
  return res;
}
*/

//
// The entire warp enters here
// If IvsI = true, search within I zone
//
template <bool IvsI>
__device__
void get_cell_bounds_z(const int icell, const int ncell, const float minx,
		       const float x0, const float x1, const float* __restrict__ bx,
		       const float rcut, int& jcell0, int& jcell1) {

  int jcell_start_left, jcell_start_right;

  if (IvsI) {
    // Search within a single zone (I)
    if (icell < 0) {
      // This is one of the image cells on the left =>
      // set the left cell boundary (jcell0) to 1 and start looking for the right
      // boundary from 1
      jcell_start_left = -1;         // with this value, we don't look for cells on the left
      jcell_start_right = 0;         // start looking for cells at right from 0
      jcell0 = 0;                    // left boundary set to minimum value
      jcell1 = -1;                   // set to "no cells" value
    } else if (icell >= ncell) {
      // This is one of the image cells on the right =>
      // set the right cell boundary (icell1) to ncell and start looking for the left
      // boundary from ncell
      jcell_start_left = ncell-1;    // start looking for cells at left from ncell
      jcell_start_right = ncell;     // with this value, we don't look for cells on the right
      jcell0 = ncell;                // set to "no cells" value
      jcell1 = ncell-1;              // right boundary set to maximum value
    } else {
      jcell_start_left = icell - 1;
      jcell_start_right = icell + 1;
      jcell0 = icell;
      jcell1 = icell;
    }
  } else {
    // Search between two different zones
    if (bx[0] >= x1 || (bx[0] < x1 && bx[0] > x0)) {
      // j-zone is to the right of i-zone
      // => no left search, start right search from 0
      jcell_start_left = -1;
      jcell_start_right = 0;
      jcell0 = 0;
      jcell1 = -1;
    } else if (bx[ncell] <= x0 || (bx[ncell] > x0 && bx[ncell] < x1)) {
      // j-zone is to the left of i-zone
      // => no right search, start left search from ncell
      jcell_start_left = ncell-1;
      jcell_start_right = ncell;
      jcell0 = ncell;
      jcell1 = ncell-1;
    } else {
      // i-zone is between j-zones
      // => safe choice is to search the entire range
      jcell_start_left = ncell-1;
      jcell_start_right = 0;
      jcell0 = ncell-1;
      jcell1 = 0;
    }
  }

  //
  // Check cells at left, stop once the distance to the cell right boundary
  // is greater than the cutoff.
  //
  // Cell right boundary is at bx[i]
  //
  for (int j=jcell_start_left;j >= 0;j--) {
    float d = x0 - bx[j];
    if (d > rcut) break;
    jcell0 = j;
  }

  //
  // Check cells at right, stop once the distance to the cell left boundary
  // is greater than the cutoff.
  //
  // Cell left boundary is at bx[i-1]
  //
  for (int j=jcell_start_right;j < ncell;j++) {
    float bx_j = (j > 0) ? bx[j-1] : minx;
    float d = bx_j - x1;
    if (d > rcut) break;
    jcell1 = j;
  }

  // Cell bounds are jcell0:jcell1
}

//
// The entire warp enters here
// If IvsI = true, search within I zone
//
template <bool IvsI>
__device__
void get_cell_bounds_xy(const int ncell, const float minx,
			const float x0, const float x1,
			const float inv_dx, const float rcut,
			int& jcell0, int& jcell1) {

  if (IvsI) {
    // Search within a single zone (I)

    //
    // Check cells at left, stop once the distance to the cell right boundary 
    // is greater than the cutoff.
    //
    // Cell right boundary is at bx
    // portion inside i-cell is (x0-bx)
    // => what is left of rcut on the left of i-cell is rcut-(x0-bx)
    //
    //float bx = minx + icell*dx;
    //jcell0 = max(0, icell - (int)ceilf((rcut - (x0 - bx))/dx));

    //
    // Check cells at right, stop once the distance to the cell left boundary
    // is greater than the cutoff.
    //
    // Cell left boundary is at bx
    // portion inside i-cell is (bx-x1)
    // => what is left of rcut on the right of i-cell is rcut-(bx-x1)
    //
    //bx = minx + (icell+1)*dx;
    //jcell1 = min(ncell-1, icell + (int)ceilf((rcut - (bx - x1))/dx));

    // Find first left boundary that is < x0-rcut
    jcell0 = max(0, (int)floorf((x0-rcut-minx)*inv_dx));

    // Find first right boundary that is > x1+rcut
    jcell1 = min(ncell-1, (int)ceilf((x1+rcut-minx)*inv_dx) - 1);

    //
    // Take care of the boundaries:
    //
    //if (icell < 0) jcell0 = 0;
    //if (icell >= ncell) jcell1 = ncell - 1;

  } else {
    //
    // Search between zones izone and jzone
    // (x0, x1) are for izone
    // (dx, minx, ncell) are for jzone
    //

    //
    // jzone left boundaries are given by: minx + jcell*dx
    // jzone right boundaries are given by: minx + (jcell+1)*dx
    //
    // izone overlap region is: x0-rcut ... x1+rcut
    //

    // Find first left boundary that is < x0-rcut
    jcell0 = max(0, (int)floorf((x0-rcut-minx)*inv_dx));

    // Find first right boundary that is > x1+rcut
    jcell1 = min(ncell-1, (int)ceilf((x1+rcut-minx)*inv_dx) - 1);
  }

  // Cell bounds are jcell0:jcell1
      
}

//
// Finds minimum of z0 and maximum of z1 across warp using __shfl -command
//
__forceinline__ __device__ void minmax_shfl(int z0, int z1, int &z0_min, int &z1_max) {
#if __CUDA_ARCH__ >= 300
  z0_min = z0;
  z1_max = z1;
  for (int i=16;i >= 1;i/=2) {
    z0_min = min(z0_min, __shfl_xor(z0, i));
    z1_max = max(z1_max, __shfl_xor(z1, i));
  }
#endif
}

__forceinline__ __device__ int min_shfl(int val) {
#if __CUDA_ARCH__ >= 300
  for (int i=16;i >= 1;i/=2) val = min(val, __shfl_xor(val, i));
#else
  val = 0;
#endif
  return val;
}

__forceinline__ __device__ int max_shfl(int val) {
#if __CUDA_ARCH__ >= 300
  for (int i=16;i >= 1;i/=2) val = max(val, __shfl_xor(val, i));
#else
  val = 0;
#endif
  return val;
}

__forceinline__ __device__ int min_shmem(int val, const int wid, volatile int* shbuf) {
  shbuf[wid] = val;
  for (int i=16;i >= 1;i/=2) {
    int n = shbuf[i ^ wid];
    shbuf[wid] = min(shbuf[wid], n);
  }
  return shbuf[wid];
}

__forceinline__ __device__ int max_shmem(int val, const int wid, volatile int* shbuf) {
  shbuf[wid] = val;
  for (int i=16;i >= 1;i/=2) {
    int n = shbuf[i ^ wid];
    shbuf[wid] = max(shbuf[wid], n);
  }
  return shbuf[wid];
}

//
// Broadcasts value from a single lane to all lanes
//
__forceinline__ __device__ int bcast_shfl(int val, const int srclane) {
#if __CUDA_ARCH__ >= 300
  return __shfl(val, srclane);
#else
  return 0;
#endif
}

__forceinline__ __device__ int bcast_shmem(int val, const int srclane, const int wid, 
					   volatile int* shbuf) {
  if (wid == srclane) shbuf[0] = val;
  return shbuf[0];
}

#if __CUDA_ARCH__ >= 300
//
// Checks that the value of integer is the warp, used for debugging
//
__forceinline__ __device__ bool check_int(int val) {
  int val0 = bcast_shfl(val, 0);
  return __all(val == val0);
}
#endif

//
// Calculates inclusive plus scan across warp
//
__forceinline__ __device__ int incl_scan_shfl(int val, const int wid, const int scansize=warpsize) {
#if __CUDA_ARCH__ >= 300
  for (int i=1;i < scansize;i*=2) {
    int n = __shfl_up(val, i, scansize);
    if (wid >= i) val += n;
  }
#else
  val = 0;
#endif
  return val;
}

__forceinline__ __device__ int incl_scan_shmem(int val, const int wid, volatile int* shbuf,
					       const int scansize=warpsize) {
  shbuf[wid] = val;
  for (int i=1;i < scansize;i*=2) {
    int n = (wid >= i) ? shbuf[wid - i] : 0;
    shbuf[wid] += n;
  }
  return shbuf[wid];
}

//
// Calculates the sum and places the result in all threads
//
__forceinline__ __device__ int sum_shfl(int val) {
#if __CUDA_ARCH__ >= 300
  for (int i=16;i >= 1;i /= 2)
    val += __shfl_xor(val, i);
#else
  val = 0;
#endif
  return val;
}

__forceinline__ __device__ int sum_shmem(int val, const int wid, volatile int* shbuf) {
  shbuf[wid] = val;
  for (int i=16;i >= 1;i /= 2) {
    int n = shbuf[i ^ wid];
    shbuf[wid] += n;
  }
  return val;
}

//
// Calculates exclusive plus-scan across warp for binary (0 or 1) values
//
// wid = warp ID = threadIdx.x % warpsize
//
__forceinline__ __device__ int binary_excl_scan(int val, int wid) {
  return __popc( __ballot(val) & ((1 << wid) - 1) );
}

//
// Calculates reduction across warp for binary (0 or 1) values
//
__forceinline__ __device__ int binary_reduce(int val) {
  return __popc(__ballot(val));
}

//
// Calculates distance exclusion mask using a single warp
//
// exclusion bits:
// 0 = no exclusion
// 1 = exclusion
//
// wid = warp thread index (0...warpSize-1)
//
template <int tilesize>
__device__ int get_dist_excl_mask(const int wid,
				  const int istart, const int iend,
				  const int jstart, const int jend,
				  const int ish,
				  const float boxx, const float boxy, const float boxz,
				  const float rcut2,
				  const float4* __restrict__ xyzq,
				  volatile float3* __restrict__ sh_xyzi
				  ) {

  // Load atom i coordinates to shared memory
  // NOTE: volatile -keyword 
  float4 xyzq_i;

  const unsigned int load_ij = threadIdx.x % tilesize;

  if (tilesize == 32 || wid < 16) {
    if (istart + load_ij <= iend) {
      xyzq_i = xyzq[istart + load_ij];
    } else {
      xyzq_i.x = -100000000.0f;
      xyzq_i.y = -100000000.0f;
      xyzq_i.z = -100000000.0f;
    }
    sh_xyzi[load_ij].x = xyzq_i.x;
    sh_xyzi[load_ij].y = xyzq_i.y;
    sh_xyzi[load_ij].z = xyzq_i.z;
  }

  // Load atom j coordinates
  float xj, yj, zj;
  //  const unsigned int loadj = (wid + (wid/TILESIZE)*(TILESIZE-1)) % TILESIZE;
  //  const unsigned int loadj = threadIdx.x % TILESIZE;
  if (jstart + load_ij <= jend) {
    float4 xyzq_j = xyzq[jstart + load_ij];
    xj = xyzq_j.x;
    yj = xyzq_j.y;
    zj = xyzq_j.z;
  } else {
    xj = 100000000.0f;
    yj = 100000000.0f;
    zj = 100000000.0f;
  }

  int q_samecell = (istart == jstart);

  // Calculate shift
  int ish_t = ish;
  float zsh = (ish_t/9 - 1)*boxz;
  ish_t -= (ish_t/9)*9;
  float ysh = (ish_t/3 - 1)*boxy;
  ish_t -= (ish_t/3)*3;
  float xsh = (ish_t - 1)*boxx;

  xj -= xsh;
  yj -= ysh;
  zj -= zsh;
  
  unsigned int excl = 0;
  int t;
  if (tilesize == 32) {

    for (t=0;t < (num_excl<tilesize>::val);t++) {
      int i = ((threadIdx.x + t) % tilesize);
      float dx = sh_xyzi[i].x - xj;
      float dy = sh_xyzi[i].y - yj;
      float dz = sh_xyzi[i].z - zj;
      float r2 = dx*dx + dy*dy + dz*dz;
      excl |= ((r2 >= rcut2) | (q_samecell && (wid <= i)) ) << t;
    }

  } else {

    /*
    for (t=0;t < (num_excl<tilesize>::val);t++) {
      int load_i = (wid + t*2 + (wid/tilesize)*(tilesize-1)) % tilesize;
      int ii = sh_start + load_i;
      float dx = sh_xyzi[ii].x - xj;
      float dy = sh_xyzi[ii].y - yj;
      float dz = sh_xyzi[ii].z - zj;
      float r2 = dx*dx + dy*dy + dz*dz;
      excl |= ((r2 >= rcut2) | (q_samecell && (load_ij <= load_i)) ) << t;
    }
    // excl is a 8 bit exclusion mask.
    // The full 32 bit exclusion mask is contained in 4 threads:
    // thread 0 contains the lowest 8 bits
    // thread 1 contains the next 8 bits, etc..
    
    excl <<= (threadIdx.x % num_thread_per_excl)*(num_excl<tilesize>::val);
    
    // Combine excl using shared memory
    const unsigned int sh_excl_ind = (threadIdx.x/warpsize)*(num_excl<tilesize>::val) + 
      (threadIdx.x % warpsize)/num_thread_per_excl;
    
    sh_excl[sh_excl_ind] = 0;
    __syncthreads();
    
    atomicOr(&sh_excl[sh_excl_ind], excl);
    
    // Make sure shared memory is written
    __syncthreads();
    
    // index to tile_excl.excl[] (0...7)
    const unsigned int excl_ind = (threadIdx.x % warpsize)/num_thread_per_excl;
    
    tile_indj[wid] = jstart;
    
    if ((threadIdx.x % num_thread_per_excl) == 0) {
      tile_excl[wid].excl[excl_ind] = sh_excl[sh_excl_ind];
    }
    */
  }

  return excl;
}

//
// Sort atoms into z-columns
//
// col_natom[0..ncellx*ncelly-1] = number of atoms in each column
// atom_icol[istart..iend]     = column index for atoms 
//
__global__ void calc_z_column_index_kernel(const float4* __restrict__ xyzq,
					   int* __restrict__ col_natom,
					   int* __restrict__ atom_icol,
					   int3* __restrict__ col_xy_zone) {

  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  
  int ind0 = 0;
  for (int izone=0;izone < 8;izone++) {
    if (i < d_nlist_param.zone_patom[izone+1]) {
      float4 xyzq_val = xyzq[i];
      float x = xyzq_val.x;
      float y = xyzq_val.y;
      float3 min_xyz = d_nlist_param.min_xyz[izone];
      int ix = (int)((x - min_xyz.x)*d_nlist_param.inv_celldx[izone]);
      int iy = (int)((y - min_xyz.y)*d_nlist_param.inv_celldy[izone]);
      int ind = ind0 + ix + iy*d_nlist_param.ncellx[izone];
      atomicAdd(&col_natom[ind], 1);
      atom_icol[i] = ind;
      int3 col_xy_zone_val;
      col_xy_zone_val.x = ix;
      col_xy_zone_val.y = iy;
      col_xy_zone_val.z = izone;
      col_xy_zone[ind] = col_xy_zone_val;
      break;
    }
    ind0 += d_nlist_param.ncellx[izone]*d_nlist_param.ncelly[izone];
  }

}

//
// Computes z column position using parallel exclusive prefix sum
// Also computes the cell_patom, col_ncellz, col_cell, and ncell
//
// NOTE: Must have nblock = 1, we loop over buckets to avoid multiple kernel calls
//
template <int tilesize>
__global__ void calc_z_column_pos_kernel(const int ncol_tot,
					 const int3* __restrict__ col_xy_zone,
					 int* __restrict__ col_natom,
					 int* __restrict__ col_patom,
					 int* __restrict__ cell_patom,
					 int* __restrict__ col_ncellz,
					 int4* __restrict__ cell_xyz_zone,
					 int* __restrict__ col_cell) {
  // Shared memory
  // Requires: blockDim.x*sizeof(int2)
  extern __shared__ int2 shpos2[];

  if (threadIdx.x == 0) {
    col_patom[0] = 0;
  }

  int2 offset = make_int2(0, 0);
  for (int base=0;base < ncol_tot;base += blockDim.x) {
    int i = base + threadIdx.x;
    int2 tmpval;
    tmpval.x = (i < ncol_tot) ? col_natom[i] : 0;  // Number of atoms in this column
    tmpval.y = (i < ncol_tot) ? (tmpval.x - 1)/tilesize + 1 : 0; // Number of z-cells in this column
    if (i < ncol_tot) col_ncellz[i] = tmpval.y;    // Set col_ncellz[icol]
    shpos2[threadIdx.x] = tmpval;
    if (i < ncol_tot) col_natom[i] = 0;
    __syncthreads();

    for (int d=1;d < blockDim.x; d *= 2) {
      int2 tmp = (threadIdx.x >= d) ? shpos2[threadIdx.x-d] : make_int2(0, 0);
      __syncthreads();
      shpos2[threadIdx.x].x += tmp.x;
      shpos2[threadIdx.x].y += tmp.y;
      __syncthreads();
    }

    if (i < ncol_tot) {
      // Write col_patom in global memory
      int2 val1 = shpos2[threadIdx.x];
      val1.x += offset.x;
      val1.y += offset.y;
      col_patom[i+1] = val1.x;
      // Write cell_patom in global memory
      // OPTIMIZATION NOTE: Is this looping too slow? Should we move this into a separate kernel?
      int2 val0 = (threadIdx.x > 0) ? shpos2[threadIdx.x - 1] : make_int2(0, 0);
      val0.x += offset.x;
      val0.y += offset.y;
      int icell0 = val0.y;
      int icell1 = val1.y;
      int iatom  = val0.x;
      // Write col_cell
      col_cell[i] = icell0;
      // col_xy_zone[icol].x = x coordinate for each column
      // col_xy_zone[icol].y = y coordinate for each column
      // col_xy_zone[icol].z = zone for each column
      int4 cell_xyz_zone_val;
      int3 col_xy_zone_val = col_xy_zone[i];
      cell_xyz_zone_val.x = col_xy_zone_val.x;   // icellx
      cell_xyz_zone_val.y = col_xy_zone_val.y;   // icelly
      cell_xyz_zone_val.z = 0;                   // icellz (set in the loop below)
      cell_xyz_zone_val.w = col_xy_zone_val.z;   // izone
      for (int icell=icell0;icell < icell1;icell++,iatom+=tilesize,cell_xyz_zone_val.z++) {
	cell_patom[icell] = iatom;
	cell_xyz_zone[icell] = cell_xyz_zone_val;
      }
    }
    
    // Add the last value to the offset for the next block
    int2 lastval = shpos2[blockDim.x-1];
    offset.x += lastval.x;
    offset.y += lastval.y;

    // Sync threads so that the next iteration can start writing in shared memory
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    // Cap off cell_patom
    cell_patom[offset.y] = offset.x;
    // Write ncell into global GPU buffer
    d_nlist_param.ncell = offset.y;
    // Clear nexcl
    d_nlist_param.nexcl = 0;
  }

  // Set zone_cell = starting cell for each zone
  if (threadIdx.x < 8) {
    int icol = d_nlist_param.zone_col[threadIdx.x];
    d_nlist_param.zone_cell[threadIdx.x] = (icol < ncol_tot) ? col_cell[icol] :  d_nlist_param.ncell;
  }

}

//
// Calculates ncellz_max[izone].
//
// blockDim.x = max number of columns over all zones
// Each thread block calculates one zone (blockIdx.x = izone)
//
__global__ void calc_ncellz_max_kernel(const int* __restrict__ col_ncellz) {

  // Shared memory
  // Requires: blockDim.x*sizeof(int)
  extern __shared__ int sh_col_ncellz[];

  // ncol[izone] gives the cumulative sum of ncellx[izone]*ncelly[izone]
  int start = d_nlist_param.ncol[blockIdx.x];
  int end   = d_nlist_param.ncol[blockIdx.x+1] - 1;

  int ncellz_max = 0;

  for (;start <= end;start += blockDim.x) {
    // Load col_ncellz into shared memory
    int pos = start + threadIdx.x;
    int col_ncellz_val = 0;
    if (pos <= end) col_ncellz_val = col_ncellz[pos];
    sh_col_ncellz[threadIdx.x] = col_ncellz_val;
    __syncthreads();
      
    // Reduce
    int n = end - start;
    for (int d=1;d < n;d *= 2) {
      int t = threadIdx.x + d;
      int val = (t < n) ? sh_col_ncellz[t] : 0;
      __syncthreads();
      sh_col_ncellz[threadIdx.x] = max(sh_col_ncellz[threadIdx.x], val);
      __syncthreads();
    }
    
    // Store into register
    if (threadIdx.x == 0) ncellz_max = max(ncellz_max, sh_col_ncellz[0]);
  }

  // Write into global memory
  if (threadIdx.x == 0) d_nlist_param.ncellz_max[blockIdx.x] = ncellz_max;
}

/*
//
// Calculates celldz_min[izone], where izone = blockIdx.x = 0...7
//
__global__ void calc_celldz_min_kernel() {

  // Shared memory
  // Requires: blockDim.x*sizeof(float)
  extern __shared__ float sh_celldz_min[];

  // ncol[izone] gives the cumulative sum of ncellx[izone]*ncelly[izone]
  int start = d_nlist_param.ncell[blockIdx.x];
  int end   = d_nlist_param.ncell[blockIdx.x+1] - 1;

  float celldz_min = (float)(1.0e20);

  for (;start <= end;start += blockDim.x) {
    // Load value into shared memory
    float celldz_min_val = (float)(1.0e20);
    int pos = start + threadIdx.x;
    if (pos <= end) celldz_min_val = ;
    sh_celldz_min[threadIdx.x] = celldz_min_val;
    __synthreads();

    // Reduce
    int n = end - start;
    for (int d=1;d < n;d *= 2) {
      int t = threadIdx.x + d;
      float val = (t < n) ? sh_celldz_min[t] : (float)(1.0e20);
      __syncthreads();
      sh_celldz_min[threadIdx.x] = min(sh_celldz_min[threadIdx.x], val);
      __syncthreads();
    }

    // Store into register
    if (threadIdx.x == 0) celldz_min = min(celldz_min, sh_celldz_min[0]);
  }

  // Write into global memory
  if (threadIdx.x == 0) d_nlist_param.celldz_min[blockIdx.x] = celldz_min;

}
*/

//
// Finds the min_xyz and max_xyz for zone "izone"
//
__global__ void calc_minmax_xyz_kernel(const int ncoord, const int izone,
				       const float4* __restrict__ xyzq) {

  // Shared memory
  // Requires: 6*blockDim.x*sizeof(float)
  extern __shared__ float sh_minmax_xyz[];
  volatile float* sh_min_x = &sh_minmax_xyz[0];
  volatile float* sh_min_y = &sh_minmax_xyz[blockDim.x];
  volatile float* sh_min_z = &sh_minmax_xyz[blockDim.x*2];
  volatile float* sh_max_x = &sh_minmax_xyz[blockDim.x*3];
  volatile float* sh_max_y = &sh_minmax_xyz[blockDim.x*4];
  volatile float* sh_max_z = &sh_minmax_xyz[blockDim.x*5];

  // Load data into shared memory
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  float4 xyzq_i = xyzq[min(i,ncoord-1)];
  float x = xyzq_i.x;
  float y = xyzq_i.y;
  float z = xyzq_i.z;
  sh_min_x[threadIdx.x] = x;
  sh_min_y[threadIdx.x] = y;
  sh_min_z[threadIdx.x] = z;
  sh_max_x[threadIdx.x] = x;
  sh_max_y[threadIdx.x] = y;
  sh_max_z[threadIdx.x] = z;
  __syncthreads();

  // Reduce
  for (int d=1;d < blockDim.x;d *= 2) {
    int t = threadIdx.x + d;
    float min_x = (t < blockDim.x) ? sh_min_x[t] : (float)(1.0e20);
    float min_y = (t < blockDim.x) ? sh_min_y[t] : (float)(1.0e20);
    float min_z = (t < blockDim.x) ? sh_min_z[t] : (float)(1.0e20);
    float max_x = (t < blockDim.x) ? sh_max_x[t] : (float)(-1.0e20);
    float max_y = (t < blockDim.x) ? sh_max_y[t] : (float)(-1.0e20);
    float max_z = (t < blockDim.x) ? sh_max_z[t] : (float)(-1.0e20);
    __syncthreads();
    sh_min_x[threadIdx.x] = min(sh_min_x[threadIdx.x], min_x);
    sh_min_y[threadIdx.x] = min(sh_min_y[threadIdx.x], min_y);
    sh_min_z[threadIdx.x] = min(sh_min_z[threadIdx.x], min_z);
    sh_max_x[threadIdx.x] = max(sh_max_x[threadIdx.x], max_x);
    sh_max_y[threadIdx.x] = max(sh_max_y[threadIdx.x], max_y);
    sh_max_z[threadIdx.x] = max(sh_max_z[threadIdx.x], max_z);
    __syncthreads();
  }

  // Store into global memory
  if (threadIdx.x == 0) {
    atomicMin(&d_nlist_param.min_xyz[izone].x, sh_min_x[0]);
    atomicMin(&d_nlist_param.min_xyz[izone].y, sh_min_y[0]);
    atomicMin(&d_nlist_param.min_xyz[izone].z, sh_min_z[0]);
    atomicMax(&d_nlist_param.max_xyz[izone].x, sh_max_x[0]);
    atomicMax(&d_nlist_param.max_xyz[izone].y, sh_max_y[0]);
    atomicMax(&d_nlist_param.max_xyz[izone].z, sh_max_z[0]);
  }

}

//
// Re-order atoms according to pos. Non-deterministic version (because of atomicAdd())
//
__global__ void reorder_atoms_z_column_kernel(const int ncoord,
					      const int* atom_icol,
					      int* col_natom,
					      const int* col_patom,
					      const float4* __restrict__ xyzq_in,
					      float4* __restrict__ xyzq_out,
					      int* __restrict__ ind_sorted) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  
  if (i < ncoord) {
    // Column index
    int icol = atom_icol[i];
    int pos = col_patom[icol];
    int n = atomicAdd(&col_natom[icol], 1);
    // new position = pos + n
    int newpos = pos + n;
    ind_sorted[newpos] = i;
    xyzq_out[newpos] = xyzq_in[i];
  }

}

//
// Reorders loc2glo
//
__global__ void build_loc2glo_kernel(const int ncoord,
				     const int* __restrict__ ind_sorted,
				     const int* __restrict__ loc2glo_in,
				     int* __restrict__ loc2glo_out) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  
  if (i < ncoord) {
    int j = ind_sorted[i];
    loc2glo_out[i] = loc2glo_in[j];
  }
}


//
// Builds glo2loc using loc2glo
//
__global__ void build_glo2loc_kernel(const int ncoord,
				     const int* __restrict__ loc2glo,
				     int* __restrict__ glo2loc) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  
  if (i < ncoord) {
    int ig = loc2glo[i];
    glo2loc[ig] = i;
  }
}

//
// Builds atom_pcell. Single warp takes care of single cell
//
__global__ void build_atom_pcell_kernel(const int* __restrict__ cell_patom,
					int* __restrict__ atom_pcell) {
  const int icell = (threadIdx.x + blockIdx.x*blockDim.x)/warpsize;
  const int wid = threadIdx.x % warpsize;

  if (icell < d_nlist_param.ncell) {
    int istart = cell_patom[icell];
    int iend   = cell_patom[icell+1] - 1;
    if (istart + wid <= iend) atom_pcell[istart + wid] = icell;
  }

}

//
// Sorts atoms according to z coordinate
//
// Uses bitonic sort, see:
// http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
//
// Each thread block sorts a single z column.
// Each z-column can only have up to blockDim.x number of atoms
//
struct keyval_t {
  float key;
  int val;
};
__global__ void sort_z_column_kernel(const int* __restrict__ col_patom,
				     float4* __restrict__ xyzq,
				     int* __restrict__ ind_sorted) {

  // Shared memory
  // Requires: blockDim.x*sizeof(keyval_t)
  extern __shared__ keyval_t sh_keyval[];

  int col_patom0 = col_patom[blockIdx.x];
  int n = col_patom[blockIdx.x+1] - col_patom0;

  // Read keys and values into shared memory
  keyval_t keyval;
  keyval.key = (threadIdx.x < n) ? xyzq[threadIdx.x + col_patom0].z : 1.0e38;
  keyval.val = (threadIdx.x < n) ? (threadIdx.x + col_patom0) : (n-1);
  sh_keyval[threadIdx.x] = keyval;
  __syncthreads();

  for (int k = 2;k <= blockDim.x;k *= 2) {
    for (int j = k/2; j > 0;j /= 2) {
      int ixj = threadIdx.x ^ j;
      if (ixj > threadIdx.x && ixj < blockDim.x) {
	// asc = true for ascending order
	bool asc = ((threadIdx.x & k) == 0);
	
	// Read data
	keyval_t keyval1 = sh_keyval[threadIdx.x];
	keyval_t keyval2 = sh_keyval[ixj];
	
	float lo_key = asc ? keyval1.key : keyval2.key;
	float hi_key = asc ? keyval2.key : keyval1.key;
	
	if (lo_key > hi_key) {
	  // keys are in wrong order => exchange
	  sh_keyval[threadIdx.x] = keyval2;
	  sh_keyval[ixj]         = keyval1;
	}
	
	//if ((i&k)==0 && get(i)>get(ixj)) exchange(i,ixj);
	//if ((i&k)!=0 && get(i)<get(ixj)) exchange(i,ixj);
      }
      __syncthreads();
    }
  }

  // sh_keyval[threadIdx.x].val gives the mapping:
  //
  // xyzq_new[threadIdx.x + col_patom0]        = xyzq[sh_keyval[threadIdx.x].val]
  // loc2glo_new[threadIdx.x + col_patom0] = loc2glo[sh_keyval[threadIdx.x].val]
  //

  float4 xyzq_val;
  int ind_val;
  if (threadIdx.x < n) {
    int i = sh_keyval[threadIdx.x].val;    
    ind_val = ind_sorted[i];
    xyzq_val = xyzq[i];
  }
  __syncthreads();
  if (threadIdx.x < n) {
    int newpos = threadIdx.x + col_patom0;
    ind_sorted[newpos] = ind_val;
    xyzq[newpos] = xyzq_val;
  }

}

//
// Calculates bounding box (bb) and cell z-boundaries (cell_bz)
// NOTE: Each thread calculates one bounding box
//
template <int tilesize>
__global__ void calc_bb_cell_bz_kernel(const int* __restrict__ cell_patom,
				       const float4* __restrict__ xyzq,
				       bb_t* __restrict__ bb,
				       float* __restrict__ cell_bz) {

  const int icell = threadIdx.x + blockIdx.x*blockDim.x;

  if (icell < d_nlist_param.ncell) {
    int istart = cell_patom[icell];
    int iend   = cell_patom[icell+1] - 1;
    float4 xyzq_val = xyzq[istart];
    float minx = xyzq_val.x;
    float miny = xyzq_val.y;
    float minz = xyzq_val.z;
    float maxx = xyzq_val.x;
    float maxy = xyzq_val.y;
    float maxz = xyzq_val.z;

    for (int i=istart+1;i <= iend;i++) {
      xyzq_val = xyzq[i];
      minx = min(minx, xyzq_val.x);
      miny = min(miny, xyzq_val.y);
      minz = min(minz, xyzq_val.z);
      maxx = max(maxx, xyzq_val.x);
      maxy = max(maxy, xyzq_val.y);
      maxz = max(maxz, xyzq_val.z);
    }
    // Set the cell z-boundary equal to the z-coordinate of the last atom
    cell_bz[icell] = xyzq_val.z;
    bb_t bb_val;
    bb_val.x = 0.5f*(minx + maxx);
    bb_val.y = 0.5f*(miny + maxy);
    bb_val.z = 0.5f*(minz + maxz);
    bb_val.wx = 0.5f*(maxx - minx);
    bb_val.wy = 0.5f*(maxy - miny);
    bb_val.wz = 0.5f*(maxz - minz);
    bb[icell] = bb_val;
  }
  
  // Zero n_ientry and n_tile -counters in preparation for neighbor list build
  if (icell == 0) {
    d_nlist_param.n_ientry = 0;
    d_nlist_param.n_tile = 0;
    //d_nlist_param.tmp = 0;
  }

}

//#define INLINE_OFF

//
//
//
template <int tilesize>
#ifndef INLINE_OFF
__forceinline__
#endif
__device__ void flush_atomj(const int wid, const int istart,
			    volatile int* __restrict__ sh_jlist,
			    const int* __restrict__ cell_patom,
			    const int min_atomj, const int max_atomj,
			    const int n_atomj, volatile int* __restrict__ sh_atomj,
			    const int min_excl_atom, const int max_excl_atom,
			    const int n_excl_atom, const int* __restrict__ excl_atom,
			    const int jtile_start,
			    tile_excl_t<tilesize>* __restrict__ tile_excl) {
  if ((min_atomj <= max_excl_atom) && (max_atomj >= min_excl_atom)) {
    int atomj = (wid < n_atomj) ? (sh_atomj[wid] >> n_jlist_max_shift) : -1;
    for (int ibase=0;ibase < n_excl_atom;ibase+=warpsize) {
      int i = ibase + wid;
      // Load excluded atom from global memory and check if there are any possible exclusions
      int excl_atomi = (i < n_excl_atom) ? (excl_atom[i] >> 5) : -1;
      int has_excl = __ballot((excl_atomi >= min_atomj) && (excl_atomi <= max_atomj));
      // Loop through possible exclusions
      while (has_excl) {
	// Get bit position for the exclusion
	int bitpos = __ffs(has_excl) - 1;
	i = ibase + bitpos;
	excl_atomi = excl_atom[i];
	// Check excl_atomi vs. sh_atomj[0...warpsize-1]
	if ((excl_atomi >> 5) == atomj) {
	  // Thread wid found exclusion between atomj and (excl_atomi & 31)
	  // NOTE: Only a single thread per warp enters here
	  int i_jlist = (sh_atomj[wid] & n_jlist_max_mask);
	  int jtile = jtile_start + i_jlist;
	  int jcell = sh_jlist[i_jlist];
	  int jstart = cell_patom[jcell];
	  int excl_ind  = atomj - jstart;
	  int excl_shift = ( (excl_atomi & 31) - excl_ind + tilesize) % tilesize;
	  unsigned int excl_mask = 1 << excl_shift;
	  tile_excl[jtile].excl[excl_ind] |= excl_mask;
	}
	// Remove bit from has_excl
	has_excl ^= (1 << bitpos);
      }
    }
  }
}
#undef INLINE_OFF

//
// Flush jlist into global memory
//
template <int tilesize>
__device__ void flush_jlist(const int wid, const int istart, const int iend,
			    const int n_jlist, volatile int* __restrict__ sh_jlist,
			    const int ish,
			    const float rcut2, const float xi, const float yi, const float zi,
			    const float4* __restrict__ xyzq,
			    const int* __restrict__ cell_patom,
			    volatile int* __restrict__ sh_atomj,
			    const int min_excl_atom, const int max_excl_atom,
			    const int n_excl_atom, const int* __restrict__ excl_atom,
			    int* __restrict__ tile_indj,
			    tile_excl_t<tilesize>* __restrict__ tile_excl,
			    ientry_t* __restrict__ ientry
#if __CUDA_ARCH__ < 300
			    ,volatile int* __restrict__ shflmem,
			    volatile float3* __restrict__ sh_xyzj
#endif
			    ) {

  // Allocate space on the global tile_excl and tile_indj -lists
  // NOTE: we are allocating space for n_jlist entries. However, not all of these are used
  //       because some of the i-j tiles will be empty. If we don't want to keep these
  //       "ghost" tiles in the list, we need to setup another shared memory buffer for
  //       exclusion masks and then only add the tiles that are non-empty.
  int jtile_start;
  if (wid == 0) jtile_start = atomicAdd(&d_nlist_param.n_tile, n_jlist);
#if __CUDA_ARCH__ >= 300
  jtile_start = bcast_shfl(jtile_start, 0);
#else
  jtile_start = bcast_shmem(jtile_start, 0, wid, shflmem);
#endif

  int min_atomj = 1 << 30;
  int max_atomj = 0;
  int n_atomj = 0;
  int n_jlist_new = 0;
  // Loop through j-cells
  for (int i_jlist=0;i_jlist < n_jlist;i_jlist++) {
    int jcell = sh_jlist[i_jlist];
    //if (jcell < 0 || jcell >= d_nlist_param.ncell) atomicOr(&d_nlist_param.tmp, jcell);
    int jstart = cell_patom[jcell];
    int jend   = cell_patom[jcell + 1] - 1;

    //---------------------------------------------------------------------------------------
    //
    // Exclusion check with jcell
    //

    // Load j-cell atoms
    float4 xyzq_j;
    if (jstart + wid <= jend) xyzq_j = xyzq[jstart + wid];
#if __CUDA_ARCH__ >= 300
    float xj = xyzq_j.x;
    float yj = xyzq_j.y;
    float zj = xyzq_j.z;
#else
    sh_xyzj[wid].x = xyzq_j.x;
    sh_xyzj[wid].y = xyzq_j.y;
    sh_xyzj[wid].z = xyzq_j.z;
#endif

    bool first = true;
    for (int j=0;j <= jend-jstart;j++) {
#if __CUDA_ARCH__ >= 300
      float xt = __shfl(xj, j);
      float yt = __shfl(yj, j);
      float zt = __shfl(zj, j);
#else
      float xt = sh_xyzj[j].x;
      float yt = sh_xyzj[j].y;
      float zt = sh_xyzj[j].z;
#endif
      float dx = xi - xt;
      float dy = yi - yt;
      float dz = zi - zt;
      
      float r2 = dx*dx + dy*dy + dz*dz;

      if (__any((r2 < rcut2))) {

	if (first) {
	  first = false;
	  // ----------------------------
	  // Set initial exclusion masks
	  // ----------------------------
	  int jtile = jtile_start + n_jlist_new;
	  // NOTE: In case i,j cells are less than tilesize atoms, add exclusions
	  int ni = (iend-istart+1);
	  unsigned int mask = (jstart + wid <= jend) ? 0 : 0xffffffff;   // j contribution
	  int up = (ni >= wid) ? ni-wid : tilesize + ni-wid;
	  int dw = (wid >= ni) ? wid-ni : tilesize + wid-ni;
	  unsigned int imask = (1 << (tilesize-ni)) - 1;
	  mask |= (imask << up) | (imask >> dw);                // i contribution
	  // Diagonal tile, exclude i >= j
	  if (istart == jstart) {
	    mask |= (0xffffffff >> wid);
	  }
	  tile_excl[jtile].excl[wid] = mask;
	  // --------------------------
	  // Keep in sh_jlist
	  // --------------------------
	  if (wid == 0) tile_indj[jtile] = jstart;
	  // Re-store jcell so that flush_atomj can read it off
	  sh_jlist[n_jlist_new] = jcell;
	  n_jlist_new++;
	}

	// This j-atom is within rcut of one of the i-atoms => add to exclusion check list
	// Add j-atom to the exclusion check list
	int atomj = jstart + j;
	min_atomj = min(min_atomj, atomj);
	max_atomj = max(max_atomj, atomj);
	sh_atomj[n_atomj++] = (atomj << n_jlist_max_shift) | (n_jlist_new-1);
	
	// Check sh_atomj[0...warpsize-1] for exclusions with any
	// of the i atoms in excl_atom[0...n_excl_atom-1]
	if (n_atomj == warpsize) {
	  // Check for topological exclusions
	  flush_atomj<tilesize>(wid, istart, sh_jlist, cell_patom,
				min_atomj, max_atomj, n_atomj, sh_atomj,
				min_excl_atom, max_excl_atom, n_excl_atom, excl_atom,
				jtile_start, tile_excl);
	  min_atomj = 1 << 30;
	  max_atomj = 0;
	  n_atomj = 0;
	} // if (natomj == warpsize)
      } // if (__any((r2 < rcut2)))
    } // for (int j=0;j <= jend-jstart;j++)

    //---------------------------------------------------------------------------------------

  } // for (int i_jlist=0;i_jlist < n_jlist;i_jlist++)

  if (n_atomj > 0) {
    flush_atomj<tilesize>(wid, istart, sh_jlist, cell_patom,
			  min_atomj, max_atomj, n_atomj, sh_atomj,
			  min_excl_atom, max_excl_atom, n_excl_atom, excl_atom,
			  jtile_start, tile_excl);
  }

  // Add to ientry list in global memory
  if (wid == 0) {
    int ientry_ind = atomicAdd(&d_nlist_param.n_ientry, 1);
    int jtile_end = jtile_start + n_jlist_new - 1;
    ientry_t ientry_val;
    ientry_val.indi    = istart;
    ientry_val.ish     = ish;
    ientry_val.startj  = jtile_start;
    ientry_val.endj    = jtile_end;
    ientry[ientry_ind] = ientry_val;
  }

}

//
// Build neighborlist for one zone at the time
//
// NOTE: One warp takes care of one cell
//
//template < int tilesize, bool IvsI >
template < int tilesize >
__global__
void build_kernel(const int maxNumExcl,
		  const int4* __restrict__ cell_xyz_zone,
		  const int* __restrict__ col_ncellz,
		  const int* __restrict__ col_cell,
		  const float* __restrict__ cell_bz,
		  const int* __restrict__ cell_patom,
		  const int* __restrict__ loc2glo,
		  const int* __restrict__ glo2loc,
		  const int* __restrict__ atom_excl_pos,
		  const int* __restrict__ atom_excl,
		  const float4* __restrict__ xyzq,
		  const float boxx, const float boxy, const float boxz,
		  const float rcut, const float rcut2,
		  const bb_t* __restrict__ bb,
		  int* __restrict__ excl_atom_heap,
		  int* __restrict__ tile_indj,
		  tile_excl_t<tilesize>* __restrict__ tile_excl,
		  ientry_t* __restrict__ ientry) {

  // Shared memory
  extern __shared__ char shbuf[];

  // Index of the i-cell
  const int icell = (threadIdx.x + blockIdx.x*blockDim.x)/warpsize;
  // Warp index
  const int wid = threadIdx.x % warpsize;

  if (icell >= d_nlist_param.ncell) return;

  // Get (icellx, icelly, icellz, izone):
  int4 icell_xyz_zone = cell_xyz_zone[icell];
  //int icellx = icell_xyz_zone.x;
  //int icelly = icell_xyz_zone.y;
  int icellz = icell_xyz_zone.z;
  //int izone  = IvsI ? 0 : icell_xyz_zone.w;
  int izone  = icell_xyz_zone.w;
  bool IvsI = (izone == 0) ? true : false;

  int n_jzone = IvsI ? 1 : d_nlist_param.n_int_zone[izone];
  
  if (n_jzone == 0) return;

  // Load bounding box
  bb_t ibb = bb[icell];

  // ----------------------------------------------------------------
  // Calculate shared memory pointers:
  //
  // Total memory requirement:
  // (blockDim.x/warpsize)*( (!IvsI)*n_jzone*sizeof(int2) + n_jlist_max*sizeof(int) 
  //                         + tilesize*sizeof(float3))
  //
  // Required space:
  // shflmem:         blockDim.x*sizeof(int)                           (Only for __CUDA_ARCH__ < 300)  
  // sh_jcellxy_min:  (blockDim.x/warpsize)*n_jzone*sizeof(int2)       (Only for IvsI = false)
  // sh_jlist:        (blockDim.x/warpsize)*n_jlist_max*sizeof(int)
  // sh_xyzj:         (blockDim.x/warpsize)*tilesize*sizeof(float3)    (Only for __CUDA_ARCH__ < 300)
  // sh_atomj:        blockDim.x*sizeof(int)
  //
  // NOTE: Each warp has its own sh_jcellxy_min[]
  int shbuf_pos = 0;
#if __CUDA_ARCH__ < 300
  // Shuffle memory buffer
  volatile int* shflmem = (int *)&shbuf[(threadIdx.x/warpsize)*warpsize*sizeof(int)];
  shbuf_pos += blockDim.x*sizeof(int);
  // j coordinates (x, y, z) for flush_jlist
  volatile float3* sh_xyzj = (float3 *)&shbuf[shbuf_pos + 
					      (threadIdx.x/warpsize)*tilesize*sizeof(float3)];
  shbuf_pos += (blockDim.x/warpsize)*tilesize*sizeof(float3);
#endif

  // jcellx and jcelly minimum values
  volatile int2 *sh_jcellxy_min;
  //if (!IvsI) {
    sh_jcellxy_min = (int2 *)&shbuf[shbuf_pos + 
				    (threadIdx.x/warpsize)*n_jzone*sizeof(int2)];
    shbuf_pos += (blockDim.x/warpsize)*n_jzone*sizeof(int2);
    //}

  // Temporary j-cell list. Each warp has its own jlist
  volatile int *sh_jlist = (int *)&shbuf[shbuf_pos +
					 (threadIdx.x/warpsize)*n_jlist_max*sizeof(int)];
  shbuf_pos += (blockDim.x/warpsize)*n_jlist_max*sizeof(int);

  // j atoms for flush_jlist
  volatile int* sh_atomj = (int *)&shbuf[shbuf_pos + 
					 (threadIdx.x/warpsize)*warpsize*sizeof(int)];
  shbuf_pos += blockDim.x*sizeof(int);
  // ----------------------------------------------------------------

  for (int ii=0;ii < n_jlist_max;ii++) sh_jlist[ii] = -1;

  //
  // Load exclusions for atoms in icell
  //

  // Allocate space for exclusions in global memory
  // Each warp (icell) has tilesize*maxNumExcl amount of space
  int* __restrict__ excl_atom = &excl_atom_heap[icell*tilesize*maxNumExcl];

  int istart = cell_patom[icell];
  int iend   = cell_patom[icell+1] - 1;
  int iatom = istart + wid;
  int jstart = 0;
  int jend = -1;
  float4 xyzq_i;
  if (iatom <= iend) {
    int ig = loc2glo[iatom];
    jstart = atom_excl_pos[ig];
    jend   = atom_excl_pos[ig+1] - 1;
    xyzq_i = xyzq[istart + wid];
  }
  float xi = xyzq_i.x;
  float yi = xyzq_i.y;
  float zi = xyzq_i.z;
  int jlen = jend - jstart + 1;
#if __CUDA_ARCH__ >= 300
  int pos = incl_scan_shfl(jlen, wid);
#else
  int pos = incl_scan_shmem(jlen, wid, shflmem);
#endif
  // Get the total number of excluded atoms by broadcasting the last value
  // across all threads in the warp
#if __CUDA_ARCH__ >= 300
  int n_excl_atom = bcast_shfl(pos, warpsize-1);
#else
  int n_excl_atom = bcast_shmem(pos, warpsize-1, wid, shflmem);
#endif
  // Get the exclusive sum position
  pos -= jlen;
  // Loop through excluded atoms:
  // Find min and max indices
  // Store atom indices to excl_atom -buffer
  int min_excl_atom = (1 << 30);                    // (= big number)
  int max_excl_atom = 0;
  int nexcl = 0;
  for (int jatom=jstart;jatom <= jend;jatom++) {
    int atom = glo2loc[atom_excl[jatom]];
    // Atoms that are not on this node are marked in glo2loc[] by value -1
    if (atom >= 0) {
      min_excl_atom = min(min_excl_atom, atom);
      max_excl_atom = max(max_excl_atom, atom);
    }
    // Store excluded atom index (atom) and atom i index
    excl_atom[pos + nexcl++] = (atom << 5) | wid;
  }
  // Reduce min_excl_atom and max_excl_atom across the warp
#if __CUDA_ARCH__ >= 300
  min_excl_atom = min_shfl(min_excl_atom);
  max_excl_atom = max_shfl(max_excl_atom);
#else
  min_excl_atom = min_shmem(min_excl_atom, wid, shflmem);
  max_excl_atom = max_shmem(max_excl_atom, wid, shflmem);
#endif

  for (int imx=d_nlist_param.imx_lo;imx <= d_nlist_param.imx_hi;imx++) {
    float imbbx0 = ibb.x + imx*boxx;
    int n_jcellx = 0;
    int jcellx_min, jcellx_max;
    if (IvsI) {
      get_cell_bounds_xy<true>(d_nlist_param.ncellx[0], d_nlist_param.min_xyz[0].x,
			       imbbx0-ibb.wx, imbbx0+ibb.wx,
			       d_nlist_param.inv_celldx[0], rcut, jcellx_min, jcellx_max);
      n_jcellx = max(0, jcellx_max - jcellx_min + 1);
      if (n_jcellx == 0) continue;
    } else {
      if (wid < n_jzone) {
	int jzone = d_nlist_param.int_zone[izone][wid];
	int jcellx0_t, jcellx1_t;
	get_cell_bounds_xy<false>(d_nlist_param.ncellx[jzone], d_nlist_param.min_xyz[jzone].x,
				  imbbx0-ibb.wx, imbbx0+ibb.wx,
				  d_nlist_param.inv_celldx[jzone], rcut, jcellx0_t, jcellx1_t);
	n_jcellx = max(0, jcellx1_t-jcellx0_t+1);
	sh_jcellxy_min[wid].x = jcellx0_t;
      }
      if (__all(n_jcellx == 0)) continue;
    }
    
    for (int imy=d_nlist_param.imy_lo;imy <= d_nlist_param.imy_hi;imy++) {
      float imbby0 = ibb.y + imy*boxy;
      int n_jcelly = 0;
      int jcelly_min, jcelly_max;
      if (IvsI) {
	get_cell_bounds_xy<true>(d_nlist_param.ncelly[0], d_nlist_param.min_xyz[0].y,
				 imbby0-ibb.wy, imbby0+ibb.wy,
				 d_nlist_param.inv_celldy[0], rcut, jcelly_min, jcelly_max);
	n_jcelly = max(0, jcelly_max - jcelly_min + 1);
	if (n_jcelly == 0) continue;
      } else {
	if (wid < n_jzone) {
	  int jzone = d_nlist_param.int_zone[izone][wid];
	  int jcelly0_t, jcelly1_t;
	  get_cell_bounds_xy<false>(d_nlist_param.ncelly[jzone], d_nlist_param.min_xyz[jzone].y,
				    imbby0-ibb.wy, imbby0+ibb.wy,
				    d_nlist_param.inv_celldy[jzone], rcut, jcelly0_t, jcelly1_t);
	  n_jcelly = max(0, jcelly1_t-jcelly0_t+1);
	  sh_jcellxy_min[wid].y = jcelly0_t;
	}
	if (__all(n_jcelly == 0)) continue;
      }

      for (int imz=d_nlist_param.imz_lo;imz <= d_nlist_param.imz_hi;imz++) {
	float imbbz0 = ibb.z + imz*boxz;
	int ish = imx+1 + 3*(imy+1 + 3*(imz+1));

	float imxi = xi + imx*boxx;
	float imyi = yi + imy*boxy;
	float imzi = zi + imz*boxz;

	int jzone_counter;
	if (!IvsI) jzone_counter = 0;
	do {
	  int n_jlist = 0;
	  int n_jcellx_t = n_jcellx;
	  int n_jcelly_t = n_jcelly;
	  int jzone;
	  if (!IvsI) {
#if __CUDA_ARCH__ >= 300
	    n_jcellx_t = bcast_shfl(n_jcellx_t, jzone_counter);
	    n_jcelly_t = bcast_shfl(n_jcelly_t, jzone_counter);
#else
	    n_jcellx_t = bcast_shmem(n_jcellx_t, jzone_counter, wid, shflmem);
	    n_jcelly_t = bcast_shmem(n_jcelly_t, jzone_counter, wid, shflmem);
#endif
	    jcellx_min = sh_jcellxy_min[jzone_counter].x;
	    jcelly_min = sh_jcellxy_min[jzone_counter].y;
	    jzone = d_nlist_param.int_zone[izone][jzone_counter];
	  }
	  int total_xy = n_jcellx_t*n_jcelly_t;
	  if (total_xy > 0) {
	    int jcellz_min = 1<<30;
	    int jcellz_max = 0;
	    for (int ibase=0;ibase < total_xy;ibase+=warpsize) {
	      int i = ibase + wid;
	      // Find new (jcellz0_t, jcellz1_t) -range
	      int jcellz0_t = 1<<30;
	      int jcellz1_t = 0;
	      if (i < total_xy) {
		int jcelly = i/n_jcellx_t;
		int jcellx = i - jcelly*n_jcellx_t;
		jcellx += jcellx_min;
		jcelly += jcelly_min;
		int jcol = jcellx + jcelly*d_nlist_param.ncellx[IvsI ? 0 : jzone] + 
		  (IvsI ? 0 : d_nlist_param.zone_col[jzone]);
		// jcell0 = beginning of cells for column jcol
		int jcell0 = col_cell[jcol];
		if (IvsI) {
		  get_cell_bounds_z<true>(icellz + imz*col_ncellz[jcol], col_ncellz[jcol],
					  d_nlist_param.min_xyz[IvsI ? 0 : jzone].z,
					  imbbz0-ibb.wz, imbbz0+ibb.wz,
					  &cell_bz[jcell0], rcut, jcellz0_t, jcellz1_t);
		} else {
		  get_cell_bounds_z<false>(icellz + imz*col_ncellz[jcol], col_ncellz[jcol],
					   d_nlist_param.min_xyz[IvsI ? 0 : jzone].z,
					   imbbz0-ibb.wz, imbbz0+ibb.wz,
					   &cell_bz[jcell0], rcut, jcellz0_t, jcellz1_t);
		}
	      } // if (i < total_xy)
	      jcellz_min = min(jcellz_min, jcellz0_t);
	      jcellz_max = max(jcellz_max, jcellz1_t);
	    } // for (int ibase...)

	    // Here all threads have their own (jcellz_min, jcellz_max),
	    // find the minimum and maximum among all threads:
#if __CUDA_ARCH__ >= 300
	    jcellz_min = min_shfl(jcellz_min);
	    jcellz_max = max_shfl(jcellz_max);
#else
	    jcellz_min = min_shmem(jcellz_min, wid, shflmem);
	    jcellz_max = max_shmem(jcellz_max, wid, shflmem);
#endif

	    int n_jcellz_max = jcellz_max - jcellz_min + 1;
	    int total_xyz = total_xy*n_jcellz_max;

	    if (total_xyz > 0) {

	      //
	      // Final loop that goes through the cells
	      //
	      // Cells are ordered in (y, x, z). (i.e. z first, x second, y third)
	      //

	      for (int ibase=0;ibase < total_xyz;ibase+=warpsize) {
		int i = ibase + wid;
		int ok = 0;
		int jcell;
		if (i < total_xyz) {
		  // Calculate (jcellx, jcelly, jcellz)
		  int it = i;	    
		  int jcelly = it/(n_jcellx_t*n_jcellz_max);
		  it -= jcelly*(n_jcellx_t*n_jcellz_max);
		  int jcellx = it/n_jcellz_max;
		  int jcellz = it - jcellx*n_jcellz_max;
		  jcellx += jcellx_min;
		  jcelly += jcelly_min;
		  jcellz += jcellz_min;
		  // Calculate column index "jcol" and final cell index "jcell"
		  int jcol = jcellx + jcelly*d_nlist_param.ncellx[IvsI ? 0 : jzone] + 
		    (IvsI ? 0 : d_nlist_param.zone_col[jzone]);
		  jcell = col_cell[jcol] + jcellz;
		  // NOTE: jcellz can be out of bounds here, so we need to check
		  if ( ((IvsI && (icell <= jcell)) || !IvsI) && jcellz >= 0 && jcellz < col_ncellz[jcol]) {
		    // Read bounding box for j-cell
		    bb_t jbb = bb[jcell];
		    // Calculate distance between i-cell and j-cell bounding boxes
		    float dx = max(0.0f, fabsf(imbbx0 - jbb.x) - ibb.wx - jbb.wx);
		    float dy = max(0.0f, fabsf(imbby0 - jbb.y) - ibb.wy - jbb.wy);
		    float dz = max(0.0f, fabsf(imbbz0 - jbb.z) - ibb.wz - jbb.wz);
		    float r2 = dx*dx + dy*dy + dz*dz;
		    ok = (r2 < rcut2);
		  }
		} // if (i < total_xyz)
		//
		// Add j-cells into temporary list (in shared memory)
		//
		// First reduce to calculate position for each thread in warp
		int pos = binary_excl_scan(ok, wid);
		int n_jlist_add = binary_reduce(ok);

		//#define DO_NOT_FLUSH
#ifndef DO_NOT_FLUSH
		// Flush if the sh_jlist[] buffer would become full
		if ((n_jlist + n_jlist_add) > n_jlist_max) {
		  flush_jlist<tilesize>(wid, istart, iend, n_jlist, sh_jlist, ish,
					rcut2, imxi, imyi, imzi, xyzq, cell_patom,
					sh_atomj,
					min_excl_atom, max_excl_atom, n_excl_atom, excl_atom,
					tile_indj, tile_excl, ientry
#if __CUDA_ARCH__ < 300
					,shflmem, sh_xyzj
#endif
					);
		  n_jlist = 0;
		}


		// Add to list
		if (ok) sh_jlist[n_jlist + pos] = jcell;
		n_jlist += n_jlist_add;

#endif
	      } // for (int ibase...)

#ifndef DO_NOT_FLUSH
	      if (n_jlist > 0) {
		flush_jlist<tilesize>(wid, istart, iend, n_jlist, sh_jlist, ish,
				      rcut2, imxi, imyi, imzi, xyzq, cell_patom,
				      sh_atomj,
				      min_excl_atom, max_excl_atom, n_excl_atom, excl_atom,
				      tile_indj, tile_excl, ientry
#if __CUDA_ARCH__ < 300
				      ,shflmem, sh_xyzj
#endif
				      );
	      }
#endif
#undef DO_NOT_FLUSH
	    } // if (total_xyz > 0)
	  } // if (total_xy > 0)

	  if (!IvsI) jzone_counter++;
	} while (!IvsI && (jzone_counter < n_jzone));

      } // for (int imz=imz_lo;imz <= imz_hi;imz++)
    } // for (int imy=imy_lo;imy <= imy_hi;imy++)
  } // for (int imx=imx_lo;imx <= imx_hi;imx++)

}

//----------------------------------------------------------------------------------------
//
// Builds tilex exclusion mask from ijlist[] based on distance and index
// Builds exclusion mask based on atom-atom distance and index (i >= j excluded)
//
// Uses 32 threads to calculate the distances for a single ijlist -entry.
//
const int nwarp_build_excl_dist = 8;

template < int tilesize >
__global__ void build_excl_kernel(const unsigned int base_tid, const int n_ijlist,
				  const int3 *ijlist,
				  const int *cell_patom, const float4 *xyzq,
				  int *tile_indj,
				  tile_excl_t<tilesize> *tile_excl,
				  const float boxx, const float boxy, const float boxz,
				  const float rcut2) {
  const int num_thread_per_excl = (32/(num_excl<tilesize>::val));

  // Global thread index
  const unsigned int gtid = threadIdx.x + blockDim.x*blockIdx.x + base_tid;
  // Global warp index
  const unsigned int wid = gtid / warpsize;
  // Local thread index (0...warpsize-1)
  const unsigned int tid = gtid % warpsize;
  // local thread index (0...tilesize-1)
  const unsigned int stid = gtid % tilesize;

  // Shared memory
  extern __shared__ char shmem[];
  volatile float3 *sh_xyzi = (float3 *)&shmem[0];    // nwarp_build_excl_dist*tilesize
  unsigned int *sh_excl = (unsigned int *)&sh_xyzi[nwarp_build_excl_dist*tilesize];

  //  __shared__ float3 sh_xyzi[nwarp_build_excl_dist*tilesize];
  //#if (tilesize == 16)
  //  __shared__ unsigned int sh_excl[nwarp_build_excl_dist*num_excl];
  //#endif

  if (wid >= n_ijlist) return;

  // Each warp computes one ijlist entry
  int3 ijlist_val = ijlist[wid];
  int icell = ijlist_val.x - 1;
  int ish   = ijlist_val.y;
  int jcell = ijlist_val.z - 1;

  int istart = cell_patom[icell] - 1;
  int iend   = cell_patom[icell+1] - 2;

  int jstart = cell_patom[jcell] - 1;
  int jend   = cell_patom[jcell+1] - 2;

  const unsigned int load_ij = threadIdx.x % tilesize;
  const int sh_start = (threadIdx.x/warpsize)*tilesize;

  // Load atom i coordinates to shared memory
  // NOTE: volatile qualifier in "sh_xyzi" guarantees that values are actually read/written from
  //       shared memory. Therefore, no __syncthreads() is needed.
  float4 xyzq_i;

  if (tilesize == 32 || tid < 16) {
    if (istart + load_ij <= iend) {
      xyzq_i = xyzq[istart + load_ij];
    } else {
      xyzq_i.x = -100000000.0f;
      xyzq_i.y = -100000000.0f;
      xyzq_i.z = -100000000.0f;
    }
    sh_xyzi[sh_start + load_ij].x = xyzq_i.x;
    sh_xyzi[sh_start + load_ij].y = xyzq_i.y;
    sh_xyzi[sh_start + load_ij].z = xyzq_i.z;
  }

  // Load atom j coordinates
  float xj, yj, zj;
  //  const unsigned int loadj = (tid + (tid/TILESIZE)*(TILESIZE-1)) % TILESIZE;
  //  const unsigned int loadj = threadIdx.x % TILESIZE;
  if (jstart + load_ij <= jend) {
    float4 xyzq_j = xyzq[jstart + load_ij];
    xj = xyzq_j.x;
    yj = xyzq_j.y;
    zj = xyzq_j.z;
  } else {
    xj = 100000000.0f;
    yj = 100000000.0f;
    zj = 100000000.0f;
  }

  // Calculate shift
  float zsh = (ish/9 - 1)*boxz;
  ish -= (ish/9)*9;
  float ysh = (ish/3 - 1)*boxy;
  ish -= (ish/3)*3;
  float xsh = (ish - 1)*boxx;

  xj -= xsh;
  yj -= ysh;
  zj -= zsh;
  
  // Make sure shared memory has been written
  // NOTE: since we're only operating within the warp, this __syncthreads() is just to make sure
  //       all values are actually written in shared memory and not kept in registers etc.
  //__syncthreads();

  int q_samecell = (icell == jcell);

  unsigned int excl = 0;
  int t;

  if (tilesize == 32) {

    for (t=0;t < (num_excl<tilesize>::val);t++) {
      int i = ((threadIdx.x + t) % tilesize);
      int ii = sh_start + i;
      float dx = sh_xyzi[ii].x - xj;
      float dy = sh_xyzi[ii].y - yj;
      float dz = sh_xyzi[ii].z - zj;
      float r2 = dx*dx + dy*dy + dz*dz;
      excl |= ((r2 >= rcut2) | (q_samecell && (tid <= i)) ) << t;
    }
    tile_indj[wid] = jstart;
    tile_excl[wid].excl[stid] = excl;

  } else {

    for (t=0;t < (num_excl<tilesize>::val);t++) {
      int load_i = (tid + t*2 + (tid/tilesize)*(tilesize-1)) % tilesize;
      int ii = sh_start + load_i;
      float dx = sh_xyzi[ii].x - xj;
      float dy = sh_xyzi[ii].y - yj;
      float dz = sh_xyzi[ii].z - zj;
      float r2 = dx*dx + dy*dy + dz*dz;
      excl |= ((r2 >= rcut2) | (q_samecell && (load_ij <= load_i)) ) << t;
    }
    // excl is a 8 bit exclusion mask.
    // The full 32 bit exclusion mask is contained in 4 threads:
    // thread 0 contains the lowest 8 bits
    // thread 1 contains the next 8 bits, etc..
    
    excl <<= (threadIdx.x % num_thread_per_excl)*(num_excl<tilesize>::val);
    
    // Combine excl using shared memory
    const unsigned int sh_excl_ind = (threadIdx.x/warpsize)*(num_excl<tilesize>::val) + 
      (threadIdx.x % warpsize)/num_thread_per_excl;
    
    sh_excl[sh_excl_ind] = 0;
    __syncthreads();
    
    atomicOr(&sh_excl[sh_excl_ind], excl);
    
    // Make sure shared memory is written
    __syncthreads();
    
    // index to tile_excl.excl[] (0...7)
    const unsigned int excl_ind = (threadIdx.x % warpsize)/num_thread_per_excl;
    
    tile_indj[wid] = jstart;
    
    if ((threadIdx.x % num_thread_per_excl) == 0) {
      tile_excl[wid].excl[excl_ind] = sh_excl[sh_excl_ind];
    }
  }

}

//----------------------------------------------------------------------------------------
//
// Combines tile_excl_top on GPU
// One thread takes care of one integer in the exclusion mask, therefore:
//
// 32x32 tile, 32 integers per tile
// 16x16 tile, 8 integers per tile
//
template <int tilesize>
__global__ void add_tile_top_kernel(const int ntile_top,
				    const int *tile_ind_top,
				    const tile_excl_t<tilesize> *tile_excl_top,
				    tile_excl_t<tilesize> *tile_excl) {
  // Global thread index
  const unsigned int gtid = threadIdx.x + blockDim.x*blockIdx.x;
  // Index to tile_ind_top[]
  const unsigned int i = gtid / (num_excl<tilesize>::val);
  // Index to exclusion mask
  const unsigned int ix = gtid % (num_excl<tilesize>::val);

  if (i < ntile_top) {
    int ind = tile_ind_top[i];
    tile_excl[ind].excl[ix] |= tile_excl_top[i].excl[ix];
  }

}

//########################################################################################
//########################################################################################
//########################################################################################

//
// Class creator
//
template <int tilesize>
NeighborList<tilesize>::NeighborList(const int ncoord_glo, const CudaTopExcl& topExcl,
				     const int nx, const int ny, const int nz) : topExcl(topExcl) {
  this->ncoord_glo = ncoord_glo;
  this->init(nx, ny, nz);
}

template <int tilesize>
NeighborList<tilesize>::NeighborList(const int ncoord_glo, const CudaTopExcl& topExcl,
				     const char *filename,
				     const int nx, const int ny, const int nz) : topExcl(topExcl) {
  this->ncoord_glo = ncoord_glo;
  this->init(nx, ny, nz);
  load(filename);
}

//
// Class destructor
//
template <int tilesize>
NeighborList<tilesize>::~NeighborList() {
  if (tile_excl != NULL) deallocate< tile_excl_t<tilesize> > (&tile_excl);
  if (ientry != NULL) deallocate<ientry_t>(&ientry);
  if (tile_indj != NULL) deallocate<int>(&tile_indj);
  // Sparse
  if (pairs != NULL) deallocate< pairs_t<tilesize> > (&pairs);
  if (ientry_sparse != NULL) deallocate<ientry_t>(&ientry_sparse);
  if (tile_indj_sparse != NULL) deallocate<int>(&tile_indj_sparse);
  // Neighbor list building
  if (col_natom != NULL) deallocate<int>(&col_natom);
  if (col_patom != NULL) deallocate<int>(&col_patom);
  if (atom_icol != NULL) deallocate<int>(&atom_icol);
  if (ind_sorted != NULL) deallocate<int>(&ind_sorted);
  if (cell_patom != NULL) deallocate<int>(&cell_patom);
  if (atom_pcell != NULL) deallocate<int>(&atom_pcell);
  if (col_ncellz != NULL) deallocate<int>(&col_ncellz);
  if (col_xy_zone != NULL) deallocate<int3>(&col_xy_zone);
  if (col_cell != NULL) deallocate<int>(&col_cell);
  if (cell_xyz_zone != NULL) deallocate<int4>(&cell_xyz_zone);
  if (cell_bz != NULL) deallocate<float>(&cell_bz);
  if (excl_atom_heap != NULL) deallocate<int>(&excl_atom_heap);
  if (cell_excl_pos != NULL) deallocate<int>(&cell_excl_pos);
  if (cell_excl != NULL) deallocate<int>(&cell_excl);
  if (bb != NULL) deallocate<bb_t>(&bb);
  deallocate_host<NeighborListParam_t>(&h_nlist_param);
}

template <int tilesize>
void NeighborList<tilesize>::init(const int nx, const int ny, const int nz) {
  n_ientry = 0;
  n_tile = 0;

  tile_excl = NULL;
  tile_excl_len = 0;

  ientry = NULL;
  ientry_len = 0;

  tile_indj = NULL;
  tile_indj_len = 0;

  // Sparse
  n_ientry_sparse = 0;
  n_tile_sparse = 0;

  pairs_len = 0;
  pairs = NULL;
  
  ientry_sparse_len = 0;
  ientry_sparse = NULL;

  tile_indj_sparse_len = 0;
  tile_indj_sparse = NULL;

  // Neighbor list building
  col_natom_len = 0;
  col_natom = NULL;

  col_patom_len = 0;
  col_patom = NULL;

  ind_sorted_len = 0;
  ind_sorted = NULL;

  atom_icol_len = 0;
  atom_icol = NULL;

  col_ncellz_len = 0;
  col_ncellz = NULL;

  col_xy_zone_len = 0;
  col_xy_zone = NULL;

  col_cell_len = 0;
  col_cell = NULL;

  cell_patom_len = 0;
  cell_patom = NULL;

  atom_pcell_len = 0;
  atom_pcell = NULL;

  cell_xyz_zone_len = 0;
  cell_xyz_zone = NULL;

  cell_bz_len = 0;
  cell_bz = NULL;

  excl_atom_heap_len = 0;
  excl_atom_heap = NULL;

  cell_excl_pos_len = 0;
  cell_excl_pos = NULL;

  cell_excl_len = 0;
  cell_excl = NULL;

  bb_len = 0;
  bb = NULL;

  allocate_host<NeighborListParam_t>(&h_nlist_param, 1);

  h_nlist_param->imx_lo = 0;
  h_nlist_param->imx_hi = 0;
  h_nlist_param->imy_lo = 0;
  h_nlist_param->imy_hi = 0;
  h_nlist_param->imz_lo = 0;
  h_nlist_param->imz_hi = 0;
  if (nx == 1) {
    h_nlist_param->imx_lo = -1;
    h_nlist_param->imx_hi = 1;
  }
  if (ny == 1) {
    h_nlist_param->imy_lo = -1;
    h_nlist_param->imy_hi = 1;
  }
  if (nz == 1) {
    h_nlist_param->imz_lo = -1;
    h_nlist_param->imz_hi = 1;
  }

  test = false;
}

//
// Setup n_int_zone[0:7] and int_zone[0:7][0:7]
// zone ordering is: I,FZ,FY,EX,FX,EZ,EY,C = 0,...7
//
template <int tilesize>
void NeighborList<tilesize>::set_int_zone(const int *zone_natom, int *n_int_zone,
					  int int_zone[][8]) {
  const int I=0,FZ=1,FY=2,EX=3/*,FX=4,EZ=5,EY=6,C=7*/;
  // Setup interaction order that maximizes communication-computation overlap
  const int zones[8][5] = { {I, -1, -1, -1, -1},  // I-I
			    {I, -1, -1, -1, -1},  // I-FZ
			    {I, FZ, -1, -1, -1},  // I-FY, FZ-FY
			    {I, -1, -1, -1, -1},  // I-EX
			    {I, FZ, FY, EX, -1},  // I-FX, FZ-FX, FY-FX, EX-FX
			    {I, FZ, -1, -1, -1},  // I-EZ, FZ-EZ
			    {I, FY, -1, -1, -1},  // I-EY, FY-EY
			    {I, -1, -1, -1, -1}}; // I-C

  n_int_zone_max = 0;
  for (int izone=0;izone < 8;izone++) {
    n_int_zone[izone] = 0;
    if (zone_natom[izone] > 0) {
      int j = 0;
      while (zones[izone][j] > -1) {
	if (zone_natom[zones[izone][j]] > 0) {
	  int_zone[izone][n_int_zone[izone]] = zones[izone][j];
	  n_int_zone[izone]++;
	}
	j++;
      }
    }
    n_int_zone_max = max(n_int_zone_max, n_int_zone[izone]);
  }

}

//
// Setup xy-cell sizes
//
template <int tilesize>
void NeighborList<tilesize>::set_cell_sizes(const int *zone_natom,
					    const float3 *max_xyz, const float3 *min_xyz,
					    int *ncellx, int *ncelly, int *ncellz_max,
					    float *celldx, float *celldy, float *celldz_min) {

  for (int izone=0;izone < 8;izone++) {
    if (zone_natom[izone] > 0) {
      // NOTE: we increase the cell sizes here by 0.001 to make sure no atoms drop outside cells
      float xsize = max_xyz[izone].x - min_xyz[izone].x + 0.001f;
      float ysize = max_xyz[izone].y - min_xyz[izone].y + 0.001f;
      float zsize = max_xyz[izone].z - min_xyz[izone].z + 0.001f;
      float delta = powf(xsize*ysize*zsize*tilesize/(float)zone_natom[izone], 1.0f/3.0f);
      ncellx[izone] = max(1, (int)(xsize/delta));
      ncelly[izone] = max(1, (int)(ysize/delta));
      // Approximation for ncellz = 2 x "uniform distribution of atoms"
      ncellz_max[izone] = max(1, 2*zone_natom[izone]/(ncellx[izone]*ncelly[izone]*tilesize));
      celldx[izone] = xsize/(float)(ncellx[izone]);
      celldy[izone] = ysize/(float)(ncelly[izone]);
      celldz_min[izone] = zsize/(float)(ncellz_max[izone]);
      if (test) {
	std::cerr << izone << ": " << min_xyz[izone].z << " ... " << max_xyz[izone].z << std::endl;
      }
    } else {
      ncellx[izone] = 0;
      ncelly[izone] = 0;
      ncellz_max[izone] = 0;
      celldx[izone] = 1.0f;
      celldy[izone] = 1.0f;
      celldz_min[izone] = 1.0f;
    }
  }

  /*
  std::cout << "celldx = " << celldx[0] << " ncellx[0] = " << ncellx[0] 
	    << " xsize = " << max_xyz[0].x - min_xyz[0].x + 0.001f << std::endl;

  std::cout << "celldy = " << celldy[0] << " ncelly[0] = " << ncelly[0] 
	    << " ysize = " << max_xyz[0].y - min_xyz[0].y + 0.001f << std::endl;
  */

}

//
// Tests for z columns
//
template <int tilesize>
bool NeighborList<tilesize>::test_z_columns(const int* zone_patom,
					    const int* ncellx, const int* ncelly,
					    const int ncol_tot,
					    const float3* min_xyz,
					    const float* inv_celldx, const float* inv_celldy,
					    const float4* xyzq, const float4* xyzq_sorted,
					    const int* col_patom) {

  int ncoord = zone_patom[8];
  float4 *h_xyzq = new float4[ncoord];
  copy_DtoH_sync<float4>(xyzq, h_xyzq, ncoord);
  float4 *h_xyzq_sorted = new float4[ncoord];
  copy_DtoH_sync<float4>(xyzq_sorted, h_xyzq_sorted, ncoord);

  int *h_col_patom = new int[ncol_tot+1];
  copy_DtoH_sync<int>(col_patom, h_col_patom, ncol_tot+1);
  int *h_ind_sorted = new int[ncoord];
  copy_DtoH_sync<int>(ind_sorted, h_ind_sorted, ncoord);

  bool ok = true;

  int izone, i, j;
  float x, y, xj, yj;
  int ix, iy, ind, lo_ind, hi_ind;
  int ind0 = 0;
  for (izone=0;izone < 8;izone++) {
    int istart = zone_patom[izone];
    int iend   = zone_patom[izone+1] - 1;
    if (iend >= istart) {
      float x0 = min_xyz[izone].x;
      float y0 = min_xyz[izone].y;
      for (i=istart;i <= iend;i++) {
	x = h_xyzq_sorted[i].x;
	y = h_xyzq_sorted[i].y;
	ix = (int)((x - x0)*inv_celldx[izone]);
	iy = (int)((y - y0)*inv_celldy[izone]);
	ind = ind0 + ix + iy*ncellx[izone];
	lo_ind = h_col_patom[ind];
	hi_ind = h_col_patom[ind+1] - 1;
	if (i < lo_ind || i > hi_ind) {
	  std::cout << "test_z_columns FAILED at i=" << i << " izone = " << izone << std::endl;
	  std::cout << "ind, lo_ind, hi_ind = " << ind << " " << lo_ind << " " << hi_ind << std::endl;
	  std::cout << "x,y = " << x << " " << y << " x0,y0 = " << x0 << " " << y0 << std::endl;
	  std::cout << "inv_celldx/y = " << inv_celldx[izone] << " " << inv_celldy[izone] << std::endl;
	  std::cout << "ix,iy =" << ix << " " << iy << " ind0 = " << ind0
		    << " ncellx = " << ncellx[izone] << " ncelly = " << ncelly[izone] << std::endl;
	  exit(1);
	}
      }
      for (i=istart;i <= iend;i++) {
	j = h_ind_sorted[i];
	x = h_xyzq_sorted[i].x;
	y = h_xyzq_sorted[i].y;
	xj = h_xyzq[j].x;
	yj = h_xyzq[j].y;
	if (x != xj || y != yj) {
	  std::cout << "test_z_columns FAILED at i=" << i << std::endl;
	  std::cout << "x,y   =" << x << " " << y << std::endl;
	  std::cout << "xj,yj =" << xj << " " << yj << std::endl;
	  exit(1);
	}
      }
      ind0 += ncellx[izone]*ncelly[izone];
    }
  }

  if (ok) std::cout << "test_z_columns OK" << std::endl;

  delete [] h_xyzq;
  delete [] h_xyzq_sorted;
  delete [] h_col_patom;
  delete [] h_ind_sorted;

  return ok;
}

//
// Tests sort
//
template <int tilesize>
bool NeighborList<tilesize>::test_sort(const int* zone_patom,
				       const int* ncellx, const int* ncelly,
				       const int ncol_tot, const int ncell_max,
				       const float3* min_xyz,
				       const float* inv_celldx, const float* inv_celldy,
				       const float4* xyzq, const float4* xyzq_sorted,
				       const int* col_patom, const int* cell_patom) {

  cudaCheck(cudaDeviceSynchronize());

  int ncoord = zone_patom[8];
  float4 *h_xyzq = new float4[ncoord];
  copy_DtoH_sync<float4>(xyzq, h_xyzq, ncoord);
  float4 *h_xyzq_sorted = new float4[ncoord];
  copy_DtoH_sync<float4>(xyzq_sorted, h_xyzq_sorted, ncoord);
  int *h_col_patom = new int[ncol_tot+1];
  copy_DtoH_sync<int>(col_patom, h_col_patom, ncol_tot+1);
  int *h_ind_sorted = new int[ncoord];
  copy_DtoH_sync<int>(ind_sorted, h_ind_sorted, ncoord);
  int *h_cell_patom = new int[ncell_max];
  copy_DtoH_sync<int>(cell_patom, h_cell_patom, ncell_max);

  bool ok = true;
  
  int izone, i, j, k, prev_ind;
  float x, y, z, prev_z;
  float xj, yj, zj;
  int ix, iy, ind, lo_ind, hi_ind;

  k = 0;
  for (i=1;i < ncol_tot+1;i++) {
    for (j=h_col_patom[i-1];j < h_col_patom[i];j+=32) {
      if (j != h_cell_patom[k]) {
	std::cout << "test_sort FAILED at i=" << i << std::endl;
	std::cout << "j,k=" << j << " " << k << "cell_patom[k]=" << h_cell_patom[k] << std::endl;
	exit(1);
      }
      k++;
    }
  }
  int ind0 = 0;
  for (izone=0;izone < 8;izone++) {
    int istart = zone_patom[izone];
    int iend   = zone_patom[izone+1] - 1;
    if (iend >= istart) {
      float x0 = min_xyz[izone].x;
      float y0 = min_xyz[izone].y;
      prev_z = min_xyz[izone].z;
      prev_ind = ind0;
      for (i=istart;i <= iend;i++) {
	x = h_xyzq_sorted[i].x;
	y = h_xyzq_sorted[i].y;
	z = h_xyzq_sorted[i].z;
	  
	ix = (int)((x - x0)*inv_celldx[izone]);
	iy = (int)((y - y0)*inv_celldy[izone]);
	ind = ind0 + ix + iy*ncellx[izone];

	if (prev_ind != ind) {
	  prev_z = min_xyz[izone].z;
	}

	lo_ind = h_col_patom[ind];
	hi_ind = h_col_patom[ind+1] - 1;
	if (i < lo_ind || i > hi_ind) {
	  std::cout << "test_sort FAILED at i=" << i << std::endl;
	  std::cout << "ind, lo_ind, hi_ind = " << ind << " " << lo_ind << " " << hi_ind << std::endl;
	  exit(1);
	}
	if (z < prev_z) {
	  std::cout << "test_sort FAILED at i=" << i << std::endl;
	  std::cout << "prev_z, z = " << prev_z << " " << z << std::endl;
	  exit(1);
	}
	prev_z = z;
	prev_ind = ind;
      }
      
      for (i=istart;i <= iend;i++) {
	j = h_ind_sorted[i];
	x = h_xyzq_sorted[i].x;
	y = h_xyzq_sorted[i].y;
	z = h_xyzq_sorted[i].z;
	xj = h_xyzq[j].x;
	yj = h_xyzq[j].y;
	zj = h_xyzq[j].z;
	if (x != xj || y != yj || z != zj) {
	  std::cout << "test_sort FAILED at i=" << i << std::endl;
	  std::cout << "x,y,z   =" << x << " " << y << " " << z << std::endl;
	  std::cout << "xj,yj,zj=" << xj << " " << yj << " " << zj << std::endl;
	  exit(1);
	}
      }
      ind0 += ncellx[izone]*ncelly[izone];
    }
  }

  if (ok) std::cout << "test_sort OK" << std::endl;

  delete [] h_xyzq;
  delete [] h_xyzq_sorted;
  delete [] h_col_patom;
  delete [] h_cell_patom;
  delete [] h_ind_sorted;

  return ok;
}

//
// Copies h_nlist_param (CPU) -> d_nlist_param (GPU)
//
template <int tilesize>
void NeighborList<tilesize>::set_nlist_param(cudaStream_t stream) {
  cudaCheck(cudaMemcpyToSymbolAsync(d_nlist_param, h_nlist_param, sizeof(NeighborListParam_t),
  				    0, cudaMemcpyHostToDevice, stream));
}

//
// Copies d_nlist_param (GPU) -> h_nlist_param (CPU)
//
template <int tilesize>
void NeighborList<tilesize>::get_nlist_param() {
  cudaCheck(cudaMemcpyFromSymbol(h_nlist_param, d_nlist_param, sizeof(NeighborListParam_t),
				 0, cudaMemcpyDeviceToHost));
}

//
// Resets n_tile and n_ientry variables for build() -call
//
template <int tilesize>
void NeighborList<tilesize>::reset() {
  get_nlist_param();
  cudaCheck(cudaDeviceSynchronize());
  h_nlist_param->n_tile = 0;
  h_nlist_param->n_ientry = 0;
  set_nlist_param(0);
  cudaCheck(cudaDeviceSynchronize());
}

//
// Returns an estimate for the number of tiles
//
template <int tilesize>
void NeighborList<tilesize>::get_tile_ientry_est(int *n_int_zone, int int_zone[][8],
						 int *ncellx, int *ncelly, int *ncellz_max,
						 float *celldx, float *celldy, float *celldz_min,
						 float rcut, int &n_tile_est, int &n_ientry_est) {
  n_tile_est = 0;
  // Loop over all zone-zone interactions
  for (int izone=0;izone < 8;izone++) {
    for (int j=0;j < n_int_zone[izone];j++) {
      int jzone = int_zone[izone][j];
      if (izone != jzone) {
	// Calculate the amount of volume overlap on zone j
	double dx_j, dy_j, dz_j;
	calc_volume_overlap(h_nlist_param->min_xyz[izone].x,
			    h_nlist_param->min_xyz[izone].y,
			    h_nlist_param->min_xyz[izone].z,
			    h_nlist_param->max_xyz[izone].x,
			    h_nlist_param->max_xyz[izone].y,
			    h_nlist_param->max_xyz[izone].z, rcut,
			    h_nlist_param->min_xyz[jzone].x,
			    h_nlist_param->min_xyz[jzone].y,
			    h_nlist_param->min_xyz[jzone].z,
			    h_nlist_param->max_xyz[jzone].x,
			    h_nlist_param->max_xyz[jzone].y,
			    h_nlist_param->max_xyz[jzone].z, dx_j, dy_j, dz_j);
	// Calculate the amount of volume overlap on zone i
	double dx_i, dy_i, dz_i;
	calc_volume_overlap(h_nlist_param->min_xyz[jzone].x,
			    h_nlist_param->min_xyz[jzone].y,
			    h_nlist_param->min_xyz[jzone].z,
			    h_nlist_param->max_xyz[jzone].x,
			    h_nlist_param->max_xyz[jzone].y,
			    h_nlist_param->max_xyz[jzone].z, rcut,
			    h_nlist_param->min_xyz[izone].x,
			    h_nlist_param->min_xyz[izone].y,
			    h_nlist_param->min_xyz[izone].z,
			    h_nlist_param->max_xyz[izone].x,
			    h_nlist_param->max_xyz[izone].y,
			    h_nlist_param->max_xyz[izone].z, dx_i, dy_i, dz_i);
	// Number of cells in each direction that are needed to fill the overlap volume
	int ncellx_j = (int)ceil(dx_j/celldx[jzone]);
	int ncelly_j = (int)ceil(dy_j/celldy[jzone]);
	int ncellz_j = (int)ceil(dz_j/celldz_min[jzone]);
	int ncell_j = ncellx_j*ncelly_j*ncellz_j;
	int ncellx_i = (int)ceil(dx_i/celldx[izone]);
	int ncelly_i = (int)ceil(dy_i/celldy[izone]);
	int ncellz_i = (int)ceil(dz_i/celldz_min[izone]);
	int ncell_i = ncellx_i*ncelly_i*ncellz_i;
	n_tile_est += ncell_j*ncell_i;
      } else {
	int ncell_i = ncellx[izone]*ncelly[izone]*ncellz_max[izone];
	// Estimate the number of neighbors in each direction for the positive direction and multiply
	// by the number of cells
	int n_neigh_ij = ((int)ceilf(rcut/celldx[izone])+1)*((int)ceilf(rcut/celldy[izone])+1)
	  *((int)ceilf(rcut/celldz_min[izone])+1)*ncell_i;
	n_tile_est += n_neigh_ij;
      }
    }
  }

  // Assume every i-j tile is in a separate ientry (worst case)
  n_ientry_est = n_tile_est;
}

//
// Calculates overlap between volumes
//
template <int tilesize>
double NeighborList<tilesize>::calc_volume_overlap(double Ax0, double Ay0, double Az0, 
						   double Ax1, double Ay1, double Az1, double rcut,
						   double Bx0, double By0, double Bz0, 
						   double Bx1, double By1, double Bz1,
						   double& dx, double& dy, double& dz) {
  double x0 = Ax0-rcut;
  double y0 = Ay0-rcut;
  double z0 = Az0-rcut;
  double x1 = Ax1+rcut;
  double y1 = Ay1+rcut;
  double z1 = Az1+rcut;

  dx = min(x1, Bx1) - max(x0, Bx0);
  dy = min(y1, By1) - max(y0, By0);
  dz = min(z1, Bz1) - max(z0, Bz0);
  dx = (dx > 0.0) ? dx : 0.0;
  dy = (dy > 0.0) ? dy : 0.0;
  dz = (dz > 0.0) ? dz : 0.0;

  return dx*dy*dz;
}


//
// Sorts atoms, when minimum and maximum coordinate values are known
//
template <int tilesize>
void NeighborList<tilesize>::sort(const int *zone_patom,
				  const float3 *max_xyz, const float3 *min_xyz,
				  float4 *xyzq,
				  float4 *xyzq_sorted,
				  int *loc2glo,
				  cudaStream_t stream) {
  int ncoord = zone_patom[8];
  assert(ncoord <= max_ncoord);
  int ncol_tot;

  if (ncoord > ncoord_glo) {
    std::cerr << "NeighborList::sort(1), Invalid value for ncoord" << std::endl;
    exit(1);
  }

  // -------------------------- Setup -----------------------------
  sort_setup(zone_patom, max_xyz, min_xyz, ncol_tot, stream);
  // --------------------------------------------------------------

  // ------------------ Allocate/Reallocate memory ----------------
  sort_alloc_realloc(ncol_tot, ncoord);
  // --------------------------------------------------------------

  // ---------------------- Do actual sorting ---------------------
  sort_core(ncol_tot, ncoord, xyzq, xyzq_sorted, stream);
  // --------------------------------------------------------------

  // ------------------ Build indices etc. after sort -------------
  sort_build_indices(ncoord, xyzq_sorted, loc2glo, stream);
  // --------------------------------------------------------------

  // Test sort
  if (test) {
    test_sort(h_nlist_param->zone_patom, h_nlist_param->ncellx, h_nlist_param->ncelly,
	      ncol_tot, ncell_max, min_xyz, h_nlist_param->inv_celldx, h_nlist_param->inv_celldy,
	      xyzq, xyzq_sorted, col_patom, cell_patom);
  }

}

//
// Sorts atoms
//
template <int tilesize>
void NeighborList<tilesize>::sort(const int *zone_patom,
				  float4 *xyzq,
				  float4 *xyzq_sorted,
				  int *loc2glo,
				  cudaStream_t stream) {
  const int ncoord = zone_patom[8];
  assert(ncoord <= max_ncoord);
  int ncol_tot;

  if (ncoord > ncoord_glo) {
    std::cerr << "NeighborList::sort(1), Invalid value for ncoord" << std::endl;
    exit(1);
  }

  for (int izone=0;izone < 8;izone++) {
    h_nlist_param->min_xyz[izone].x = (float)1.0e20;
    h_nlist_param->min_xyz[izone].y = (float)1.0e20;
    h_nlist_param->min_xyz[izone].z = (float)1.0e20;
    h_nlist_param->max_xyz[izone].x = (float)(-1.0e20);
    h_nlist_param->max_xyz[izone].y = (float)(-1.0e20);
    h_nlist_param->max_xyz[izone].z = (float)(-1.0e20);
  }

  set_nlist_param(stream);

  for (int izone=0;izone < 8;izone++) {
    int nstart = zone_patom[izone];
    int ncoord_zone = zone_patom[izone+1] - nstart;
    if (ncoord_zone > 0) {
      int nthread = 512;
      int nblock = (ncoord_zone-1)/nthread+1;
      int shmem_size = 6*nthread*sizeof(float);
      calc_minmax_xyz_kernel<<< nblock, nthread, shmem_size, stream >>>
	(ncoord_zone, izone, &xyzq[nstart]);
    }
  }
  cudaCheck(cudaStreamSynchronize(stream));

  get_nlist_param();

  /*
  std::cout << "min_xyz = " << h_nlist_param->min_xyz[0].x << " "
	    << h_nlist_param->min_xyz[0].y << " "
	    << h_nlist_param->min_xyz[0].z << " " << std::endl;

  std::cout << "max_xyz = " << h_nlist_param->max_xyz[0].x << " "
	    << h_nlist_param->max_xyz[0].y << " "
	    << h_nlist_param->max_xyz[0].z << " " << std::endl;
  */

  // -------------------------- Setup -----------------------------
  sort_setup(zone_patom, h_nlist_param->max_xyz, h_nlist_param->min_xyz, ncol_tot, stream);
  // --------------------------------------------------------------

  // ------------------ Allocate/Reallocate memory ----------------
  sort_alloc_realloc(ncol_tot, ncoord);
  // --------------------------------------------------------------

  // ---------------------- Do actual sorting ---------------------
  sort_core(ncol_tot, ncoord, xyzq, xyzq_sorted, stream);
  // --------------------------------------------------------------

  // ------------------ Build indices etc. after sort -------------
  sort_build_indices(ncoord, xyzq_sorted, loc2glo, stream);
  // --------------------------------------------------------------

  // Test sort
  if (test) {
    test_sort(h_nlist_param->zone_patom, h_nlist_param->ncellx, h_nlist_param->ncelly,
	      ncol_tot, ncell_max, h_nlist_param->min_xyz,
	      h_nlist_param->inv_celldx, h_nlist_param->inv_celldy,
	      xyzq, xyzq_sorted, col_patom, cell_patom);
  }

}

//
// Setups sort parameters: nlist_param, ncol_tot, ncell_max
//
// NOTE: ncell_max is an approximate upper bound for the number of cells,
//       it is possible to blow this bound, so we should check for it
template < int tilesize >
void NeighborList<tilesize>::sort_setup(const int *zone_patom,
					const float3 *max_xyz, const float3 *min_xyz,
					int &ncol_tot, cudaStream_t stream) {

  int zone_natom[8];
  //
  // Calculate zone_natom
  //
  // zone_natom[izone] = number of atoms in zone "izone"
  for (int izone=0;izone < 8;izone++) {
    zone_natom[izone] = zone_patom[izone+1] - zone_patom[izone];
  }

  set_int_zone(zone_natom, h_nlist_param->n_int_zone, h_nlist_param->int_zone);
  set_cell_sizes(zone_natom, max_xyz, min_xyz,
		 h_nlist_param->ncellx, h_nlist_param->ncelly, h_nlist_param->ncellz_max,
		 h_nlist_param->celldx, h_nlist_param->celldy, h_nlist_param->celldz_min);

  // Setup nlist_param and copy it to GPU
  int ncol = 0;
  int max_ncellxy = 0;
  h_nlist_param->zone_col[0] = 0;
  for (int izone=0;izone < 8;izone++) {
    h_nlist_param->zone_patom[izone] = zone_patom[izone];
    h_nlist_param->ncol[izone] = ncol;
    max_ncellxy = max(max_ncellxy, h_nlist_param->ncellx[izone]*h_nlist_param->ncelly[izone]);
    ncol += h_nlist_param->ncellx[izone]*h_nlist_param->ncelly[izone];
    h_nlist_param->inv_celldx[izone] = 1.0f/h_nlist_param->celldx[izone];
    h_nlist_param->inv_celldy[izone] = 1.0f/h_nlist_param->celldy[izone];
    h_nlist_param->min_xyz[izone].x = min_xyz[izone].x;
    h_nlist_param->min_xyz[izone].y = min_xyz[izone].y;
    h_nlist_param->min_xyz[izone].z = min_xyz[izone].z;
    if (izone > 0) {
      h_nlist_param->zone_col[izone] = h_nlist_param->zone_col[izone-1] +
	h_nlist_param->ncellx[izone-1]*h_nlist_param->ncelly[izone-1];
    }
  }
  h_nlist_param->ncol[8] = ncol;
  h_nlist_param->zone_patom[8] = zone_patom[8];

  // Copy h_nlist_param => d_nlist_param
  set_nlist_param(stream);

  ncol_tot = 0;
  ncell_max = 0;
  for (int izone=0;izone < 8;izone++) {
    int ncellxy = h_nlist_param->ncellx[izone]*h_nlist_param->ncelly[izone];
    ncol_tot += ncellxy;
    ncell_max += ncellxy*h_nlist_param->ncellz_max[izone];
  }

  // Wait till set_nlist_param finishes 
  cudaCheck(cudaStreamSynchronize(stream));
}

//
// Allocates / Re-allocates memory for sort
//
template <int tilesize>
void NeighborList<tilesize>::sort_alloc_realloc(const int ncol_tot, const int ncoord) {

#ifdef STRICT_MEMORY_REALLOC
  float fac = 1.0f;
#else
  float fac = 1.2f;
#endif

  reallocate<int>(&atom_icol, &atom_icol_len, ncoord, fac);
  reallocate<int>(&atom_pcell, &atom_pcell_len, ncoord, fac);

  reallocate<int>(&cell_patom, &cell_patom_len, ncell_max+1, fac);
  reallocate<int4>(&cell_xyz_zone, &cell_xyz_zone_len, ncell_max, fac);
  reallocate<float>(&cell_bz, &cell_bz_len, ncell_max, fac);
  reallocate<bb_t>(&bb, &bb_len, ncell_max, fac);

  reallocate<int>(&col_natom, &col_natom_len, ncol_tot, fac);
  reallocate<int>(&col_patom, &col_patom_len, ncol_tot+1, fac);
  reallocate<int>(&col_ncellz, &col_ncellz_len, ncol_tot, fac);
  reallocate<int3>(&col_xy_zone, &col_xy_zone_len, ncol_tot, fac);
  reallocate<int>(&col_cell, &col_cell_len, ncol_tot, fac);

  reallocate<int>(&ind_sorted, &ind_sorted_len, ncoord, fac);
}

//
// Builds indices etc. after sort. xyzq is the sorted array
//
template <int tilesize>
void NeighborList<tilesize>::sort_build_indices(const int ncoord, float4 *xyzq, int *loc2glo,
						cudaStream_t stream) {
  int nthread, nblock, shmem_size;

  //
  // Build loc2glo (really we are reordering it with ind_sorted)
  //
  // Make a copy of loc2glo to glo2loc
  // NOTE: This is temporary, glo2loc will be used for a different purpose later
  copy_DtoD<int>(loc2glo, topExcl.get_glo2loc(), ncoord, stream);
  nthread = 512;
  nblock = (ncoord - 1)/nthread + 1;
  build_loc2glo_kernel<<< nblock, nthread, 0, stream >>>(ncoord, ind_sorted,
							 topExcl.get_glo2loc(), loc2glo);
  cudaCheck(cudaGetLastError());

  // Build glo2loc
  // NOTE: We mark atoms that do not exist with -1
  set_gpu_array<int>(topExcl.get_glo2loc(), topExcl.get_ncoord(), -1, stream);
  nthread = 512;
  nblock = (ncoord - 1)/nthread + 1;
  build_glo2loc_kernel<<< nblock, nthread, 0, stream >>>(ncoord, loc2glo, topExcl.get_glo2loc());
  cudaCheck(cudaGetLastError());

  // Build atom_pcell
  nthread = 1024;
  nblock = (ncell_max - 1)/(nthread/warpsize) + 1;
  build_atom_pcell_kernel<<< nblock, nthread, 0, stream >>>(cell_patom, atom_pcell);
  cudaCheck(cudaGetLastError());

  // Build bounding box (bb) and cell boundaries (cell_bz)
  nthread = 512;
  nblock = (ncell_max-1)/nthread + 1;
  shmem_size = 0;
  calc_bb_cell_bz_kernel<tilesize> <<< nblock, nthread, shmem_size, stream >>>
    (cell_patom, xyzq, bb, cell_bz);
  cudaCheck(cudaGetLastError());

  /*
  //
  // Calculate ncellz_max[izone]
  // NOTE: This is only needed in order to get a better estimate for n_tile_est
  //
  nthread = min(((max_ncellxy-1)/warpsize+1)*warpsize, get_max_nthread());
  nblock = 8;
  shmem_size = nthread*sizeof(int);
  calc_ncellz_max_kernel<<< nblock, nthread, shmem_size, stream >>>(col_ncellz);
  */

}

//
// Sorts atoms, core subroutine.
//
template <int tilesize>
void NeighborList<tilesize>::sort_core(const int ncol_tot, const int ncoord,
				       float4 *xyzq,
				       float4 *xyzq_sorted,
				       cudaStream_t stream) {

  int nthread, nblock, shmem_size;

  // Clear col_natom
  clear_gpu_array<int>(col_natom, ncol_tot, stream);

  // Make a copy of loc2glo to glo2loc
  // NOTE: This is temporary, glo2loc will be used for a different purpose later
  //copy_DtoD<int>(loc2glo, glo2loc, ncoord, stream);

  //
  // Calculate number of atoms in each z-column (col_natom)
  // and the column index for each atom (atom_icol)
  //
  nthread = 512;
  nblock = (ncoord-1)/nthread+1;
  calc_z_column_index_kernel<<< nblock, nthread, 0, stream >>>
    (xyzq, col_natom, atom_icol, col_xy_zone);
  cudaCheck(cudaGetLastError());

  //
  // Calculate positions in z columns
  // NOTE: Clears col_natom and sets (col_patom, cell_patom, col_ncellz, d_nlist_param.ncell)
  //
  nthread = min(((ncol_tot-1)/tilesize+1)*tilesize, get_max_nthread());
  shmem_size = nthread*sizeof(int2);
  calc_z_column_pos_kernel<tilesize> <<< 1, nthread, shmem_size, stream >>>
    (ncol_tot, col_xy_zone, col_natom, col_patom, cell_patom, col_ncellz, cell_xyz_zone,
     col_cell);

  //
  // Reorder atoms into z-columns
  // NOTE: also sets up startcell_zone[izone]
  //
  nthread = 512;
  nblock = (ncoord-1)/nthread+1;
  reorder_atoms_z_column_kernel<<< nblock, nthread, 0, stream >>>
    (ncoord, atom_icol, col_natom, col_patom, xyzq, xyzq_sorted, ind_sorted);
  cudaCheck(cudaGetLastError());

  // Test z columns
  if (test) {
    cudaCheck(cudaDeviceSynchronize());
    test_z_columns(h_nlist_param->zone_patom, h_nlist_param->ncellx, h_nlist_param->ncelly,
		   ncol_tot, h_nlist_param->min_xyz, h_nlist_param->inv_celldx, h_nlist_param->inv_celldy,
		   xyzq, xyzq_sorted, col_patom);
  }

  // Now sort according to z coordinate
  nthread = 0;
  nblock = 0;
  for (int izone=0;izone < 8;izone++) {
    nblock += h_nlist_param->ncellx[izone]*h_nlist_param->ncelly[izone];
    nthread = max(nthread, h_nlist_param->ncellz_max[izone]*tilesize);
  }
  if (nthread < get_max_nthread()) {
    shmem_size = nthread*sizeof(keyval_t);
    sort_z_column_kernel<<< nblock, nthread, shmem_size, stream >>>
      (col_patom, xyzq_sorted, ind_sorted);
    cudaCheck(cudaGetLastError());
  } else {
    std::cerr << "Neighborlist::sort_core, this version of sort_z_column_kernel not implemented yet"
	      << std::endl;
    exit(1);
  }

}

//
// Sets ientry from host memory array
//
template <int tilesize>
void NeighborList<tilesize>::set_ientry(int n_ientry, ientry_t *h_ientry) {

  this->n_ientry = n_ientry;

  // Allocate & reallocate d_ientry
#ifdef STRICT_MEMORY_REALLOC
  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.0f);
#else
  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.4f);
#endif

  // Copy to device
  copy_HtoD_sync<ientry_t>(h_ientry, ientry, n_ientry);
}

//
// Builds neighborlist
//
template <int tilesize>
void NeighborList<tilesize>::build(const float boxx, const float boxy, const float boxz,
				   const float rcut,
				   const float4 *xyzq, const int *loc2glo,
				   cudaStream_t stream) {
  int nthread, nblock, shmem_size;

  get_nlist_param();
  
  int n_tile_est, n_ientry_est;
  get_tile_ientry_est(h_nlist_param->n_int_zone, h_nlist_param->int_zone,
		      h_nlist_param->ncellx, h_nlist_param->ncelly, h_nlist_param->ncellz_max,
		      h_nlist_param->celldx, h_nlist_param->celldy, h_nlist_param->celldz_min,
		      rcut, n_tile_est, n_ientry_est);
  //std::cout << "n_ientry_est = " << n_ientry_est << " n_tile_est = " << n_tile_est << std::endl;

  if (test) {
    std::cout << "ncell = " << h_nlist_param->ncell << " ncell_max = " << ncell_max
	      << " n_tile_est = " << n_tile_est << std::endl;
    for (int izone=0;izone < 8;izone++) {
      std::cout << izone << ": " << h_nlist_param->ncellx[izone]
		<< " " << h_nlist_param->ncelly[izone]
		<< " " << h_nlist_param->ncellz_max[izone]
		<< std::endl;
    }
  }

  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry_est, 1.0f);
  reallocate<tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, n_tile_est, 1.0f);
  reallocate<int>(&tile_indj, &tile_indj_len, n_tile_est, 1.0f);

#ifdef STRICT_MEMORY_REALLOC
  reallocate<int>(&excl_atom_heap, &excl_atom_heap_len, ncell_max*tilesize*topExcl.getMaxNumExcl(), 1.0f);
#else
  reallocate<int>(&excl_atom_heap, &excl_atom_heap_len, ncell_max*tilesize*topExcl.getMaxNumExcl(), 1.2f);
#endif

  //clear_gpu_array< tile_excl_t<tilesize> >(tile_excl, tile_excl_len, stream);

  // Shared memory requirements:
  // (blockDim.x/warpsize)*( (!IvsI)*n_jzone*sizeof(int2) + n_jlist_max*sizeof(int) 
  //                         + tilesize*sizeof(float3))

  // I vs. I
  nthread = 512;
  //nblock = (ncell_max-1)/(nthread/warpsize) + 1;
  nblock = (h_nlist_param->ncell-1)/(nthread/warpsize) + 1;
  shmem_size = (nthread/warpsize)*n_jlist_max*sizeof(int);     // sh_jlist[]
  shmem_size += nthread*sizeof(int);                           // sh_atomj[]
  if (get_cuda_arch() < 300) {
    shmem_size += nthread*sizeof(int);                         // shflmem[]
    shmem_size += (nthread/warpsize)*tilesize*sizeof(float3);  // sh_xyzj[]
  }
  // For !IvsI, shmem_size += (nthread/warpsize)*n_int_zone_max*sizeof(int2)
  shmem_size += (nthread/warpsize)*n_int_zone_max*sizeof(int2);// sh_jcellxy_min[]

  //std::cout << "NeighborList::build, shmem_size = " << shmem_size << std::endl;
  build_kernel<tilesize>
    <<< nblock, nthread, shmem_size, stream >>>
    (topExcl.getMaxNumExcl(), cell_xyz_zone, col_ncellz, col_cell, cell_bz, cell_patom,
     loc2glo, topExcl.get_glo2loc(), topExcl.getAtomExclPos(), topExcl.getAtomExcl(),
     xyzq, boxx, boxy, boxz, rcut, rcut*rcut, bb, excl_atom_heap,
     tile_indj, tile_excl, ientry);
  cudaCheck(cudaGetLastError());

  /*
  // Rest
  nthread = 512;
  nblock = (ncell_max-1)/(nthread/warpsize) + 1;
  shmem_size = (nthread/warpsize)*( n_jlist_max*sizeof(int) + tilesize*sizeof(float3)) + 
    nthread*sizeof(int);
  if (get_cuda_arch() < 300) shmem_size += nthread*sizeof(int);
  shmem_size += (nthread/warpsize)*n_int_zone_max*sizeof(int2)
  //std::cout << "NeighborList::build, shmem_size = " << shmem_size << std::endl;
  build_kernel<tilesize, false>
    <<< nblock, nthread, shmem_size, stream >>>
    (max_nexcl, cell_xyz_zone, col_ncellz, col_cell, cell_bz, cell_patom, loc2glo, glo2loc,
     atom_excl_pos, atom_excl, xyzq, boxx, boxy, boxz, rcut, rcut*rcut, bb, excl_atom_heap,
     tile_indj, tile_excl, ientry);
  cudaCheck(cudaGetLastError());
  */

  cudaCheck(cudaDeviceSynchronize());
  get_nlist_param();

  n_ientry = h_nlist_param->n_ientry;
  n_tile = h_nlist_param->n_tile;

  if (n_tile > n_tile_est) {
    std::cout << "NeighborList::build, Limit blown: n_tile > n_tile_est"<< std::endl;
    exit(1);
  }

  if (n_ientry > n_ientry_est) {
    std::cout << "NeighborList::build, Limit blown: n_ientry > n_ientry_est"<< std::endl;
    exit(1);
  }

  if (test) test_build(boxx, boxy, boxz, rcut, xyzq, loc2glo);
}

struct tileinfo_t {
  int excl;
  double dx, dy, dz;
  double r2;
};

template <int tilesize>
bool compare(tileinfo_t* tile1, tileinfo_t* tile2, std::vector<int2>& ijvec) {
  ijvec.clear();
  bool ok = true;
  for (int jt=0;jt < tilesize;jt++) {
    for (int it=0;it < tilesize;it++) {
      if (tile1[it + jt*tilesize].excl != tile2[it + jt*tilesize].excl) {
	int2 ijval;
	ijval.x = it;
	ijval.y = jt;
	ijvec.push_back(ijval);
	ok = false;
      }
    }
  }
  return ok;
}

template <int tilesize>
void set_excl(tileinfo_t* tile1) {
  for (int jt=0;jt < tilesize;jt++) {
    for (int it=0;it < tilesize;it++) {
      tile1[it + jt*tilesize].excl = 1;
    }
  }
}

template<int tilesize>
void print_excl(tileinfo_t* tile1) {
  for (int jt=0;jt < tilesize;jt++) {
    for (int it=0;it < tilesize;it++) {
      fprintf(stderr,"%d ",tile1[it + jt*tilesize].excl);
    }
    fprintf(stderr,"\n");
  }
}

std::ostream& operator<< (std::ostream &o, const bb_t& b) {
  o << "x,y,z= " << b.x << " " << b.y << " "<< b.z
    << " wx,wy,wz= " << b.wx << " " << b.wy << " "<< b.wz;
  return o;
}

//
// Test neighbor list building with a simple N^2 algorithm
//
template <int tilesize>
void NeighborList<tilesize>::test_build(const double boxx, const double boxy, const double boxz,
					const double rcut, const float4 *xyzq, const int* loc2glo) {

  cudaCheck(cudaDeviceSynchronize());
  get_nlist_param();

  int n_ientry = h_nlist_param->n_ientry;
  int n_tile = h_nlist_param->n_tile;
  int ncell = h_nlist_param->ncell;

  int *h_atom_excl_pos = new int[topExcl.getAtomExclPosLen()];
  int *h_atom_excl = new int[topExcl.getAtomExclLen()];
  copy_DtoH_sync<int>(topExcl.getAtomExclPos(), h_atom_excl_pos, topExcl.getAtomExclPosLen());
  copy_DtoH_sync<int>(topExcl.getAtomExcl(), h_atom_excl, topExcl.getAtomExclLen());

  int ncoord = h_nlist_param->zone_patom[8];

  int *h_loc2glo = new int[ncoord];
  copy_DtoH_sync<int>(loc2glo, h_loc2glo, ncoord);

  float4* h_xyzq = new float4[ncoord];
  copy_DtoH_sync<float4>(xyzq, h_xyzq, ncoord);

  bb_t *h_bb = new bb_t[ncell];
  copy_DtoH_sync<bb_t>(bb, h_bb, ncell);

  float* h_cell_bz = new float[ncell];
  copy_DtoH_sync<float>(cell_bz, h_cell_bz, ncell);

  double rcut2 = rcut*rcut;

  //float boxxf = (float)boxx;
  //float boxyf = (float)boxy;
  //float boxzf = (float)boxz;

  double hboxx = 0.5*boxx;
  double hboxy = 0.5*boxy;
  double hboxz = 0.5*boxz;
  //float hboxxf = (float)hboxx;
  //float hboxyf = (float)hboxy;
  //float hboxzf = (float)hboxz;

  // Calculate number of pairs
  int npair_cpu = calc_cpu_pairlist<double>(h_nlist_param->zone_patom, h_xyzq, h_loc2glo,
					    h_atom_excl_pos, h_atom_excl,
					    boxx, boxy, boxz, rcut);

  std::cout << "npair_cpu=" << npair_cpu << std::endl;

  ientry_t* h_ientry = new ientry_t[n_ientry];
  tile_excl_t<tilesize>* h_tile_excl = new tile_excl_t<tilesize>[n_tile];
  int* h_tile_indj = new int[n_tile];
  int* h_cell_patom = new int[ncell+1];

  copy_DtoH_sync<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH_sync<tile_excl_t<tilesize> >(tile_excl, h_tile_excl, n_tile);
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, n_tile);
  copy_DtoH_sync<int>(cell_patom, h_cell_patom, ncell+1);

  // Calculate number of pairs on the GPU list
  int npair_gpu = calc_gpu_pairlist<double>(n_ientry, h_ientry, h_tile_indj, h_tile_excl,
					    h_xyzq, boxx, boxy, boxz, rcut);

  std::cout << "npair_gpu=" << npair_gpu << std::endl;

  tileinfo_t *tileinfo = new tileinfo_t[tilesize*tilesize];
  tileinfo_t *tileinfo2 = new tileinfo_t[tilesize*tilesize];
  std::vector<int2> ijvec;

  //
  // Go through all cell pairs and check that the gpu caught all of them
  //
  int npair_gpu2 = 0;
  int ncell_pair = 0;
  bool okloop = true;
  for (int izone=0;izone < 8;izone++) {
    for (int jzone=0;jzone < 8;jzone++) {
      if (izone == 1 && jzone != 5) continue;
      if (izone == 2 && jzone != 1 && jzone != 6) continue;
      if (izone == 4 && jzone != 1 && jzone != 2 && jzone != 3) continue;

      int icell_start = h_nlist_param->zone_cell[izone];
      int icell_end   = (izone+1 < 8) ? h_nlist_param->zone_cell[izone+1] : h_nlist_param->ncell;

      for (int icell=icell_start;icell < icell_end;icell++) {
	int jcell_start = h_nlist_param->zone_cell[jzone];
	int jcell_end   = (jzone+1 < 8) ? h_nlist_param->zone_cell[jzone+1] : h_nlist_param->ncell;
	if (izone == 0 && jzone == 0) {
	  jcell_start = icell;
	}
	for (int jcell=jcell_start;jcell < jcell_end;jcell++) {
	  int istart = h_cell_patom[icell];
	  int iend   = h_cell_patom[icell+1]-1;
	  int jstart = h_cell_patom[jcell];
	  int jend   = h_cell_patom[jcell+1]-1;
	  int npair_tile1 = 0;
	  bool pair = false;
	  double min_diff = 1.0e10;
	  set_excl<tilesize>(tileinfo);
	  for (int i=istart;i <= iend;i++) {
	    double xi = h_xyzq[i].x;
	    double yi = h_xyzq[i].y;
	    double zi = h_xyzq[i].z;
	    int ig = h_loc2glo[i];
	    int excl_start = h_atom_excl_pos[ig];
	    int excl_end   = h_atom_excl_pos[ig+1]-1;
	    for (int j=jstart;j <= jend;j++) {
	      tileinfo_t tileinfo_val;
	      tileinfo_val.excl = 1;
	      if (icell != jcell || i < j) {
		double xj = h_xyzq[j].x;
		double yj = h_xyzq[j].y;
		double zj = h_xyzq[j].z;
		double dx = xi - xj;
		double dy = yi - yj;
		double dz = zi - zj;
		double shx = 0.0;
		double shy = 0.0;
		double shz = 0.0;
		if (dx > hboxx) {
		  shx = -boxx;
		} else if (dx < -hboxx) {
		  shx = boxx;
		}
		if (dy > hboxy) {
		  shy = -boxy;
		} else if (dy < -hboxy) {
		  shy = boxy;
		}
		if (dz > hboxz) {
		  shz = -boxz;
		} else if (dz < -hboxz) {
		  shz = boxz;
		}
		double xis = xi + shx;
		double yis = yi + shy;
		double zis = zi + shz;
		dx = xis - xj;
		dy = yis - yj;
		dz = zis - zj;
		double r2 = dx*dx + dy*dy + dz*dz;
		min_diff = min(min_diff, fabs(r2-rcut2));

		int jg = h_loc2glo[j];
		bool excl_flag = false;
		for (int excl=excl_start;excl <= excl_end;excl++) {
		  if (h_atom_excl[excl] == jg) {
		    excl_flag = true;
		    break;
		  }
		}
		if (excl_flag == false) {
		  tileinfo_val.excl = 0;
		} else {
		  tileinfo_val.excl = 1;
		}
		if (r2 < rcut2 && !excl_flag) {
		  npair_gpu2++;
		  npair_tile1++;
		  pair = true;
		}
		tileinfo_val.dx = dx;
		tileinfo_val.dy = dy;
		tileinfo_val.dz = dz;
		tileinfo_val.r2 = r2;
	      }
	      int it = i-istart;
	      int jt = j-jstart;
	      tileinfo[it + jt*tilesize] = tileinfo_val;
	    }
	  } // for (int i=istart;i <= iend;i++)

	  if (pair) {
	    // Pair of cells with atoms starting at istart and jstart
	    bool found_this_pair = false;
	    int ind, jtile;
	    for (ind=0;ind < n_ientry;ind++) {
	      if (h_ientry[ind].indi != istart &&
		  h_ientry[ind].indi != jstart) continue;
	      int startj = h_ientry[ind].startj;
	      int endj   = h_ientry[ind].endj;
	      for (jtile=startj;jtile <= endj;jtile++) {
		if ((h_ientry[ind].indi == istart && h_tile_indj[jtile] == jstart) ||
		    (h_ientry[ind].indi == jstart && h_tile_indj[jtile] == istart)) {
		  found_this_pair = true;
		  break;
		}
	      }
	      if (found_this_pair) break;
	    }

	    if (found_this_pair) {
	      // Check the tile we found (ind, jtile)
	      int istart0, jstart0;
	      istart0 = h_ientry[ind].indi;
	      jstart0 = h_tile_indj[jtile];
	
	      int ish     = h_ientry[ind].ish;
	      int ish_tmp = ish;
	      double shz = (ish_tmp/9 - 1)*boxz;
	      ish_tmp -= (ish_tmp/9)*9;
	      double shy = (ish_tmp/3 - 1)*boxy;
	      ish_tmp -= (ish_tmp/3)*3;
	      double shx = (ish_tmp - 1)*boxx;

	      int npair_tile2 = 0;
	      for (int i=istart0;i < istart0+tilesize;i++) {
		double xi = (double)h_xyzq[i].x + shx;
		double yi = (double)h_xyzq[i].y + shy;
		double zi = (double)h_xyzq[i].z + shz;
		for (int j=jstart0;j < jstart0+tilesize;j++) {
		  int bitpos = ((i-istart0) - (j-jstart0) + tilesize) % tilesize;
		  unsigned int excl = h_tile_excl[jtile].excl[j-jstart0] >> bitpos;
		  double xj = h_xyzq[j].x;
		  double yj = h_xyzq[j].y;
		  double zj = h_xyzq[j].z;
		  double dx = xi - xj;
		  double dy = yi - yj;
		  double dz = zi - zj;
		  double r2 = dx*dx + dy*dy + dz*dz;

		  int it, jt;
		  if (istart0 == istart) {
		    it = i-istart0;
		    jt = j-jstart0;
		  } else {
		    jt = i-istart0;
		    it = j-jstart0;
		  }

		  tileinfo_t tileinfo_val;
		  tileinfo_val.excl = (excl & 1);
		  tileinfo_val.dx = dx;
		  tileinfo_val.dy = dy;
		  tileinfo_val.dz = dz;
		  tileinfo_val.r2 = r2;
		  tileinfo2[it + jt*tilesize] = tileinfo_val;

		  if (r2 < rcut2 && !(excl & 1)) {
		    npair_tile2++;
		  }
		}
	      }

	      //if (abs(npair_tile1 - npair_tile2) > 0) {
	      if (!compare<tilesize>(tileinfo, tileinfo2, ijvec)) {

		bool ok = true;
		for (int k=0;k < ijvec.size();k++) {
		  int it = ijvec.at(k).x;
		  int jt = ijvec.at(k).y;
		  tileinfo_t tileinfo_val;
		  tileinfo_val = tileinfo[it + jt*tilesize];
		  if (tileinfo_val.r2 >= rcut2) {
		    ok = false;
		    break;
		  }
		}
		if (!ok) continue;

		//std::cerr << "tile pair ERROR: icell = " << icell << " jcell = " << jcell 
		//	  << " npair_tile1 = " << npair_tile1 << " npair_tile2 = " << npair_tile2
		//	  << std::endl;
		//std::cerr << " istart0 = " << istart0 << " jstart0 = " << jstart0 
		//	  << " izone = " << izone << " jzone = " << jzone << std::endl;
		//std::cerr << " istart,iend  = " << istart << " " << iend 
		//	  << " jstart,jend  = " << jstart << " " << jend
		//	  << " min_diff=" << min_diff << std::endl;

		//fprintf(stderr,"tileinfo:\n");
		//print_excl<tilesize>(tileinfo);
		//fprintf(stderr,"tileinfo2:\n");
		//print_excl<tilesize>(tileinfo2);

		for (int k=0;k < ijvec.size();k++) {
		  int it = ijvec.at(k).x;
		  int jt = ijvec.at(k).y;

		  tileinfo_t tileinfo_val;
		  tileinfo_val = tileinfo[it + jt*tilesize];
		  tileinfo_t tileinfo2_val;
		  tileinfo2_val = tileinfo2[it + jt*tilesize];

		  if (tileinfo_val.r2 < rcut2) {
		    fprintf(stderr,"----------------------------------------------\n");
		    fprintf(stderr,"it,jt=%d %d dx,dy,dz=%lf %lf %lf r2=%lf | %d %d\n",it,jt,
			    tileinfo_val.dx,tileinfo_val.dy,tileinfo_val.dz,tileinfo_val.r2,
			    tileinfo_val.excl, tileinfo2_val.excl);
		    int ig = h_loc2glo[it];
		    int excl_start = h_atom_excl_pos[ig];
		    int excl_end   = h_atom_excl_pos[ig+1]-1;
		    int jg = h_loc2glo[jt];
		    //bool excl_flag = false;
		    for (int excl=excl_start;excl <= excl_end;excl++) {
		      if (h_atom_excl[excl] == jg) {
			fprintf(stderr,"======================= EXCLUSION FOUND! ==================\n");
			break;
		      }
		    }
		  }
		}
		//exit(1);
	      }
	  
	    } else {
	      std::cerr << "tile pair with istart = " << istart << " jstart = " << jstart
			<< " NOT FOUND" << std::endl;
	      std::cerr << "min_diff = " << min_diff << " npair_tile1 = " << npair_tile1
			<< " ind = " << ind << std::endl;
	      std::cerr << h_bb[icell] << " | " << icell << std::endl;
	      std::cerr << h_bb[jcell] << " | " << jcell << std::endl;
	      //exit(1);
	      okloop = false;
	    }
	  }

	  if (pair) ncell_pair++;
	} // for (int jcell...)
      } // for (int icell...)
    }
  }

  delete [] tileinfo;
  delete [] tileinfo2;

  delete [] h_atom_excl_pos;
  delete [] h_atom_excl;

  delete [] h_loc2glo;

  delete [] h_xyzq;
  delete [] h_ientry;
  delete [] h_tile_excl;
  delete [] h_tile_indj;
  delete [] h_cell_patom;

  delete [] h_bb;
  delete [] h_cell_bz;

  if (npair_cpu != npair_gpu || !okloop) {
    std::cout << "##################################################" << std::endl;
    std::cout << "test_build FAILED" << std::endl;
    std::cout << "n_ientry = " << n_ientry << " n_tile = " << n_tile << std::endl;
    std::cout << "npair_cpu = " << npair_cpu << " npair_gpu = " << npair_gpu 
	      << " npair_gpu2 = " << npair_gpu2 << std::endl;
    std::cout << "##################################################" << std::endl;
  } else {
    std::cout << "test_build OK" << std::endl;
  }

  if (!okloop) exit(1);

}

//
// Calculates GPU pair list
//
template <int tilesize> template <typename T>
int NeighborList<tilesize>::calc_gpu_pairlist(const int n_ientry, const ientry_t* ientry,
					      const int* tile_indj,
					      const tile_excl_t<tilesize>* tile_excl, const float4* xyzq,
					      const double boxx, const double boxy, const double boxz,
					      const double rcut) {
  T rcut2 = rcut*rcut;
  T boxxT = boxx;
  T boxyT = boxy;
  T boxzT = boxz;

  int npair = 0;
  for (int ind=0;ind < n_ientry;ind++) {
    int istart = ientry[ind].indi;
    int ish    = ientry[ind].ish;
    int startj = ientry[ind].startj;
    int endj   = ientry[ind].endj;

    int ish_tmp = ish;
    T shz = (ish_tmp/9 - 1)*boxzT;
    ish_tmp -= (ish_tmp/9)*9;
    T shy = (ish_tmp/3 - 1)*boxyT;
    ish_tmp -= (ish_tmp/3)*3;
    T shx = (ish_tmp - 1)*boxxT;

    for (int jtile=startj;jtile <= endj;jtile++) {
      for (int i=istart;i < istart+tilesize;i++) {
	T xi = (T)xyzq[i].x + shx;
	T yi = (T)xyzq[i].y + shy;
	T zi = (T)xyzq[i].z + shz;
	int jstart = tile_indj[jtile];
	for (int j=jstart;j < jstart+tilesize;j++) {
	  int bitpos = ((i-istart) - (j-jstart) + tilesize) % tilesize;
	  int excl = tile_excl[jtile].excl[j-jstart] >> bitpos;
	  T xj = xyzq[j].x;
	  T yj = xyzq[j].y;
	  T zj = xyzq[j].z;
	  T dx = xi - xj;
	  T dy = yi - yj;
	  T dz = zi - zj;
	  T r2 = dx*dx + dy*dy + dz*dz;
	  if (r2 < rcut2 && !(excl & 1)) npair++;
	}
      }
    }
  }

  return npair;
}

//
// Calculates CPU pair list
//
template <int tilesize> template <typename T>
int NeighborList<tilesize>::calc_cpu_pairlist(const int* zone_patom, const float4* xyzq,
					      const int* loc2glo, const int* atom_excl_pos,
					      const int* atom_excl, const double boxx,
					      const double boxy, const double boxz, const double rcut) {
  T rcut2 = rcut*rcut;
  T boxxT = boxx;
  T boxyT = boxy;
  T boxzT = boxz;
  T hboxx = 0.5*boxx;
  T hboxy = 0.5*boxy;
  T hboxz = 0.5*boxz;

  int npair = 0;
  for (int izone=0;izone < 8;izone++) {
    for (int jzone=0;jzone < 8;jzone++) {
      if (izone == 1 && jzone != 5) continue;
      if (izone == 2 && jzone != 1 && jzone != 6) continue;
      if (izone == 4 && jzone != 1 && jzone != 2 && jzone != 3) continue;

      int istart = zone_patom[izone];
      int iend   = zone_patom[izone+1] - 1;
      int jstart = zone_patom[jzone];
      int jend   = zone_patom[jzone+1] - 1;

      for (int i=istart;i <= iend;i++) {
	T xi = xyzq[i].x;
	T yi = xyzq[i].y;
	T zi = xyzq[i].z;
	int ig = loc2glo[i];
	int excl_start = atom_excl_pos[ig];
	int excl_end   = atom_excl_pos[ig+1]-1;
	if (izone == 0 && jzone == 0) jstart = i + 1;
	for (int j=jstart;j <= jend;j++) {
	  T xj = xyzq[j].x;
	  T yj = xyzq[j].y;
	  T zj = xyzq[j].z;
	  T dx = xi - xj;
	  T dy = yi - yj;
	  T dz = zi - zj;
	  if (dx > hboxx) {
	    dx = (xi-boxxT) - xj;
	  } else if (dx < -hboxx) {
	    dx = (xi+boxxT) - xj;
	  }
	  if (dy > hboxy) {
	    dy = (yi-boxyT) - yj;
	  } else if (dy < -hboxy) {
	    dy = (yi+boxyT) - yj;
	  }
	  if (dz > hboxz) {
	    dz = (zi-boxzT) - zj;
	  } else if (dz < -hboxz) {
	    dz = (zi+boxzT) - zj;
	  }
	  T r2 = dx*dx + dy*dy + dz*dz;

	  if (r2 < rcut2) {
	    int jg = loc2glo[j];
	    bool excl_flag = false;
	    for (int excl=excl_start;excl <= excl_end;excl++) {
	      if (atom_excl[excl] == jg) {
	      	excl_flag = true;
		break;
	      }
	    }
	    if (excl_flag == false) npair++;
	  }

	}
	//
      }
    }
  }

  return npair;
}

/*
void test_excl_dist_index(const int n_ijlist, const int3 *d_ijlist,
			  const int *d_cell_patom, const float4 *d_xyzq,
			  int *d_tile_indj,
			  tile_excl_t *d_tile_excl,
			  const float boxx, const float boxy, const float boxz,
			  const float rcut2) {

  int3 *h_ijlist;
  int *h_cell_patom;
  float4 *h_xyzq;
  int *h_tile_indj;
  tile_excl_t *h_tile_excl;

  h_ijlist = (int3 *)malloc(n_ijlist*sizeof(int3));
  h_cell_patom = (int *)malloc(mdsim.ncell*sizeof(int));
  h_xyzq = (float4 *)malloc(mdsim.ncoord*sizeof(float4));
  h_tile_indj = (int *)malloc(n_ijlist*sizeof(int));
  h_tile_excl = (tile_excl_t *)malloc(n_ijlist*sizeof(tile_excl_t));

  cudaCheck(cudaMemcpy(h_ijlist, d_ijlist, sizeof(int3)*n_ijlist,
		       cudaMemcpyDeviceToHost));

  cudaCheck(cudaMemcpy(h_cell_patom, d_cell_patom, sizeof(int)*mdsim.ncell,
		       cudaMemcpyDeviceToHost));

  cudaCheck(cudaMemcpy(h_xyzq, d_xyzq, sizeof(float4)*mdsim.ncoord,
		       cudaMemcpyDeviceToHost));

  cudaCheck(cudaMemcpy(h_tile_indj, d_tile_indj, sizeof(int)*n_ijlist,
		       cudaMemcpyDeviceToHost));

  cudaCheck(cudaMemcpy(h_tile_excl, d_tile_excl, sizeof(tile_excl_t)*n_ijlist,
		       cudaMemcpyDeviceToHost));

  for (int wid=0;wid < n_ijlist;wid++) {

    int3 ijlist_val = h_ijlist[wid];
    int icell = ijlist_val.x - 1;
    int ish   = ijlist_val.y;
    int jcell = ijlist_val.z - 1;

    int istart = h_cell_patom[icell] - 1;
    int iend   = h_cell_patom[icell+1] - 2;

    int jstart = h_cell_patom[jcell] - 1;
    int jend   = h_cell_patom[jcell+1] - 2;

    int q_samecell = (icell == jcell);

    // Calculate shift
    float zsh = (ish/9 - 1)*boxz;
    ish -= (ish/9)*9;
    float ysh = (ish/3 - 1)*boxy;
    ish -= (ish/3)*3;
    float xsh = (ish - 1)*boxx;
    
    int i,j,ii,jj;

    for (ii=istart,i=0;ii <= iend;ii++,i++) {
      float4 xyzq_i = h_xyzq[ii];
      float xi = xyzq_i.x;
      float yi = xyzq_i.y;
      float zi = xyzq_i.z;
      for (jj=jstart,j=0;jj <= jend;jj++,j++) {
	float4 xyzq_j = h_xyzq[jj];
	float xj = xyzq_j.x - xsh;
	float yj = xyzq_j.y - ysh;
	float zj = xyzq_j.z - zsh;
	float dx = xi - xj;
	float dy = yi - yj;
	float dz = zi - zj;
	float r2 = dx*dx + dy*dy + dz*dz;
#if (TILESIZE == 16)
	int ttid = ((i+j) % 2)*16 + j;
	int iexcl = ttid/4;
	int tmp = i + 1 + j*15;
	int shbit = ((tmp/2) % 8) + (j % 4)*8;
#else
	int ij = i + j*TILESIZE - j;
	int iexcl = j;
	int shbit = (ij % TILESIZE);
#endif
	unsigned int ibit = 1 << shbit;
	unsigned int excl = ((r2 >= rcut2) | (q_samecell && (j <= i)) ) << shbit;
	unsigned int excl_gpu = h_tile_excl[wid].excl[iexcl];
	if ( ((excl_gpu & ibit) ^ excl) != 0 && fabsf(r2-rcut2) > 7.0e-5) {
	  printf("Error found in test_excl_dist_index:\n");
	  printf("wid = %d i = %d j = %d iexcl = %d shbit = %d\n",wid,i,j,iexcl,shbit);
	  printf("ii = %d jj = %d %d %d %d %d\n",ii,jj,r2 >= rcut2,
		 (q_samecell && (j <= i)),icell,jcell);
	  printf("%x ^ %x = %x \n",excl_gpu & ibit, excl, (excl_gpu & ibit) ^ excl);
	  printf("i:  %f %f %f\n",xi,yi,zi);
	  printf("j:  %f %f %f\n",xj,yj,zj);
	  printf("jo: %f %f %f\n",xyzq_j.x,xyzq_j.y,xyzq_j.z);
	  printf("sh: %f %f %f\n",xsh,ysh,zsh);
	  printf("dx: %1.8f %1.8f %1.8f\n",dx,dy,dz);
	  printf("r2: %f %e\n",r2,fabsf(r2-rcut2));
	  exit(1);
	}
      }
    }

  }

  free(h_ijlist);
  free(h_cell_patom);
  free(h_xyzq);
  free(h_tile_indj);
  free(h_tile_excl);

  printf("test_excl_dist_index OK\n");
}
*/

//
// Host wrapper for build_tilex_kernel
// Builds exclusion mask based on atom-atom distance and index (i >= j excluded)
//
template <int tilesize>
void NeighborList<tilesize>::build_excl(const float boxx, const float boxy, const float boxz,
					const float rcut,
					const int n_ijlist, const int3 *ijlist,
					const int *cell_patom,
					const float4 *xyzq,
					cudaStream_t stream) {

  if (n_ijlist == 0) return;

  // Allocate & re-allocate (d_tile_indj, d_tile_excl)
#ifdef STRICT_MEMORY_REALLOC
  reallocate<int>(&tile_indj, &tile_indj_len, n_ijlist, 1.0f);
  reallocate<tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, n_ijlist, 1.0f);
#else
  reallocate<int>(&tile_indj, &tile_indj_len, n_ijlist, 1.2f);
  reallocate<tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, n_ijlist, 1.2f);
#endif

  float rcut2 = rcut*rcut;

  int nthread = nwarp_build_excl_dist*warpsize;
  int nblock_tot = (n_ijlist-1)/(nthread/warpsize) + 1;
  size_t shmem_size = nwarp_build_excl_dist*tilesize*sizeof(float3); 

  if (tilesize == 16) {
    shmem_size += nwarp_build_excl_dist*(num_excl<tilesize>::val)*sizeof(unsigned int);
  }

  int3 max_nblock3 = get_max_nblock();
  unsigned int max_nblock = max_nblock3.x;
  unsigned int base_tid = 0;

  while (nblock_tot != 0) {

    int nblock = (nblock_tot > max_nblock) ? max_nblock : nblock_tot;
    nblock_tot -= nblock;

    build_excl_kernel<tilesize>
      <<< nblock, nthread, shmem_size, stream >>>
      (base_tid, n_ijlist, ijlist, cell_patom,
       xyzq, tile_indj, tile_excl,
       boxx, boxy, boxz,
       rcut2);

    base_tid += nblock*nthread;

    cudaCheck(cudaGetLastError());
  }

  /*
  if (mdsim.q_test != 0) {
    test_excl_dist_index(mdsim.n_ijlist, mdsim.ijlist, mdsim.cell_patom,
			 mdsim.xyzq.xyzq, mdsim.tile_indj, mdsim.tile_excl,
			 boxx, boxy, boxz,
			 rcut2);
  }
  */

}

//
// Host wrapper for add_tile_top_kernel
//
template <int tilesize>
void NeighborList<tilesize>::add_tile_top(const int ntile_top, const int *tile_ind_top,
					  const tile_excl_t<tilesize> *tile_excl_top,
					  cudaStream_t stream) {
  int nthread = 256;
  int nblock = (ntile_top*(num_excl<tilesize>::val) - 1)/nthread + 1;
  
  add_tile_top_kernel<tilesize>
    <<< nblock, nthread, 0, stream >>>
    (ntile_top, tile_ind_top, tile_excl_top, tile_excl);
  
  cudaCheck(cudaGetLastError());
}

//
// Splits neighbor list into dense and sparse parts
//
template <int tilesize>
void NeighborList<tilesize>::split_dense_sparse(int npair_cutoff) {

  ientry_t *h_ientry = new ientry_t[n_ientry];
  int *h_tile_indj = new int[n_tile];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[n_tile];

  ientry_t *h_ientry_dense = new ientry_t[n_ientry];
  int *h_tile_indj_dense = new int[n_tile];
  tile_excl_t<tilesize> *h_tile_excl_dense = new tile_excl_t<tilesize>[n_tile];

  ientry_t *h_ientry_sparse = new ientry_t[n_ientry];
  int *h_tile_indj_sparse = new int[n_tile];
  pairs_t<tilesize> *h_pairs = new pairs_t<tilesize>[n_tile];

  copy_DtoH_sync<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, n_tile);
  copy_DtoH_sync< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, n_tile);

  int n_ientry_dense = 0;
  int n_tile_dense = 0;
  n_ientry_sparse = 0;
  n_tile_sparse = 0;
  for (int i=0;i < n_ientry;i++) {
    bool sparse_i_tiles = true;
    int startj_dense = n_tile_dense;
    for (int j=h_ientry[i].startj;j <= h_ientry[i].endj;j++) {
      int npair = 0;
      for (int k=0;k < (num_excl<tilesize>::val);k++) {
	unsigned int n1bit = BitCount(h_tile_excl[j].excl[k]);
	npair += 32 - n1bit;
      }

      if (npair <= npair_cutoff) {
	// Sparse
	for (int k=0;k < (num_excl<tilesize>::val);k++) {
	  
	}
	h_tile_indj_sparse[n_tile_sparse] = h_tile_indj[j];
	n_tile_sparse++;
      } else {
	// Dense
	for (int k=0;k < (num_excl<tilesize>::val);k++) {
	  h_tile_excl_dense[n_tile_dense].excl[k] = h_tile_excl[j].excl[k];
	}
	h_tile_indj_dense[n_tile_dense] = h_tile_indj[j];
	n_tile_dense++;
	sparse_i_tiles = false;
      }

    }

    if (sparse_i_tiles) {
      // Sparse
    } else {
      h_ientry_dense[n_ientry_dense] = h_ientry[i];
      h_ientry_dense[n_ientry_dense].startj = startj_dense;
      h_ientry_dense[n_ientry_dense].endj = n_tile_dense - 1;
      n_ientry_dense++;
    }
  }

  n_ientry = n_ientry_dense;
  n_tile = n_tile_dense;

  copy_HtoD_sync<ientry_t>(h_ientry_dense, ientry, n_ientry);
  copy_HtoD_sync<int>(h_tile_indj_dense, tile_indj, n_tile);
  copy_HtoD_sync< tile_excl_t<tilesize> >(h_tile_excl_dense, tile_excl, n_tile);

  allocate<ientry_t>(&ientry_sparse, n_ientry_sparse);
  allocate<int>(&tile_indj_sparse, n_tile_sparse);
  allocate< pairs_t<tilesize> >(&pairs, n_tile_sparse);
  ientry_sparse_len = n_ientry_sparse;
  tile_indj_sparse_len = n_tile_sparse;
  pairs_len = n_tile_sparse;

  copy_HtoD_sync<ientry_t>(h_ientry_sparse, ientry_sparse, n_ientry_sparse);
  copy_HtoD_sync<int>(h_tile_indj_sparse, tile_indj_sparse, n_tile_sparse);
  copy_HtoD_sync< pairs_t<tilesize> >(h_pairs, pairs, n_tile_sparse);

  delete [] h_ientry;
  delete [] h_tile_indj;
  delete [] h_tile_excl;

  delete [] h_ientry_dense;
  delete [] h_tile_indj_dense;
  delete [] h_tile_excl_dense;

  delete [] h_ientry_sparse;
  delete [] h_tile_indj_sparse;
  delete [] h_pairs;

}

//
// Removes empty tiles
//
template <int tilesize>
void NeighborList<tilesize>::remove_empty_tiles() {

  ientry_t *h_ientry = new ientry_t[n_ientry];
  int *h_tile_indj = new int[n_tile];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[n_tile];

  ientry_t *h_ientry_noempty = new ientry_t[n_ientry];
  int *h_tile_indj_noempty = new int[n_tile];
  tile_excl_t<tilesize> *h_tile_excl_noempty = new tile_excl_t<tilesize>[n_tile];

  copy_DtoH_sync<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, n_tile);
  copy_DtoH_sync< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, n_tile);

  int n_ientry_noempty = 0;
  int n_tile_noempty = 0;
  for (int i=0;i < n_ientry;i++) {
    bool empty_i_tiles = true;
    int startj_noempty = n_tile_noempty;
    for (int j=h_ientry[i].startj;j <= h_ientry[i].endj;j++) {
      bool empty_tile = true;
      for (int k=0;k < (num_excl<tilesize>::val);k++) {
	unsigned int n1bit = BitCount(h_tile_excl[j].excl[k]);
	if (n1bit != 32) empty_tile = false;
      }

      if (!empty_tile) {
	for (int k=0;k < (num_excl<tilesize>::val);k++) {
	  h_tile_excl_noempty[n_tile_noempty].excl[k] = h_tile_excl[j].excl[k];
	}
	h_tile_indj_noempty[n_tile_noempty] = h_tile_indj[j];
	n_tile_noempty++;
	empty_i_tiles = false;
      }
    }

    if (!empty_i_tiles) {
      h_ientry_noempty[n_ientry_noempty] = h_ientry[i];
      h_ientry_noempty[n_ientry_noempty].startj = startj_noempty;
      h_ientry_noempty[n_ientry_noempty].endj = n_tile_noempty - 1;
      n_ientry_noempty++;
    }
  }

  n_ientry = n_ientry_noempty;
  n_tile = n_tile_noempty;

  copy_HtoD_sync<ientry_t>(h_ientry_noempty, ientry, n_ientry);
  copy_HtoD_sync<int>(h_tile_indj_noempty, tile_indj, n_tile);
  copy_HtoD_sync< tile_excl_t<tilesize> >(h_tile_excl_noempty, tile_excl, n_tile);

  delete [] h_ientry;
  delete [] h_tile_indj;
  delete [] h_tile_excl;

  delete [] h_ientry_noempty;
  delete [] h_tile_indj_noempty;
  delete [] h_tile_excl_noempty;

}

//
// Analyzes the neighbor list and prints info
//
template <int tilesize>
void NeighborList<tilesize>::analyze() {

  ientry_t *h_ientry = new ientry_t[n_ientry];
  int *h_tile_indj = new int[n_tile];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[n_tile];

  copy_DtoH_sync<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH_sync<int>(tile_indj, h_tile_indj, n_tile);
  copy_DtoH_sync< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, n_tile);

  std::cout << "Number of i-tiles = " << n_ientry << ", total number of tiles = " 
	    << n_tile << std::endl;

  std::ofstream file_npair("npair.txt", std::ofstream::out);
  std::ofstream file_nj("nj.txt", std::ofstream::out);

  unsigned int nexcl_bit = 0;
  unsigned int nexcl_bit_self = 0;
  unsigned int nempty_tile = 0;
  unsigned int nempty_line = 0;
  unsigned int npair_tot = 0;
  for (int i=0;i < n_ientry;i++) {
    file_nj << h_ientry[i].endj - h_ientry[i].startj + 1 << std::endl;
    for (int j=h_ientry[i].startj;j <= h_ientry[i].endj;j++) {
      int npair = 0;
      bool empty_tile = true;
      for (int k=0;k < (num_excl<tilesize>::val);k++) {
	unsigned int n1bit = BitCount(h_tile_excl[j].excl[k]);

	if (n1bit > 32) {
	  std::cerr << n1bit << " " << std::hex << h_tile_excl[j].excl[k] << std::endl;
	  exit(1);
	}

	if (n1bit == 32)
	  nempty_line++;
	else
	  empty_tile = false;

	nexcl_bit += n1bit;
	npair += 32 - n1bit;

	if (h_ientry[i].indi == h_tile_indj[j]) nexcl_bit_self += n1bit;
      }
      if (empty_tile) nempty_tile++;
      file_npair << npair << std::endl;
      npair_tot += npair;
    }
  }

  file_npair.close();
  file_nj.close();

  unsigned int n_tile_pairs = n_tile*tilesize*tilesize;
  std::cout << "Total number of pairs = " << npair_tot 
	    << " (" << (double)npair_tot*100.0/(double)n_tile_pairs << "% full)" << std::endl;
  std::cout << "Total number of pairs in tiles = " << n_tile_pairs << std::endl;
  std::cout << "Number of excluded pairs = " << nexcl_bit << " (" << 
    ((double)nexcl_bit*100)/(double)n_tile_pairs << "%)" << std::endl;
  std::cout << "Number of excluded pairs in self (i==j) tiles = " << nexcl_bit_self << " (" << 
    ((double)nexcl_bit_self*100)/(double)n_tile_pairs << "%)" << std::endl;
  std::cout << "Number of empty lines = " << nempty_line << " (" <<
    ((double)nempty_line*100)/((double)(n_tile*tilesize)) << "%)" << std::endl;
  std::cout << "Number of empty tiles = " << nempty_tile << " (" <<
    ((double)nempty_tile*100)/(double)n_tile << "%)" << std::endl;

  delete [] h_ientry;
  delete [] h_tile_indj;
  delete [] h_tile_excl;

}

//
// Load neighbor list from file
//
template <int tilesize>
void NeighborList<tilesize>::load(const char *filename) {

  ientry_t *h_ientry;
  int *h_tile_indj;
  tile_excl_t<tilesize> *h_tile_excl;

  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(filename);

    file >> n_ientry >> n_tile;

    h_ientry = new ientry_t[n_ientry];
    h_tile_indj = new int[n_tile];
    h_tile_excl = new tile_excl_t<tilesize>[n_tile];

    for (int i=0;i < n_ientry;i++) {
      file >> std::dec >> h_ientry[i].indi >> h_ientry[i].ish >> 
	h_ientry[i].startj >> h_ientry[i].endj;
      for (int j=h_ientry[i].startj;j <= h_ientry[i].endj;j++) {
	file >> std::dec >> h_tile_indj[j];
	for (int k=0;k < (num_excl<tilesize>::val);k++) {
	  file >> std::hex >> h_tile_excl[j].excl[k];
	}
      }
    }

    file.close();
  }
  catch(std::ifstream::failure e) {
    std::cerr << "Error opening/reading/closing file " << filename << std::endl;
    exit(1);
  }

#ifdef STRICT_MEMORY_REALLOC
  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.0f);
  reallocate<int>(&tile_indj, &tile_indj_len, n_tile, 1.0f);
  reallocate< tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, n_tile, 1.0f);
#else
  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.2f);
  reallocate<int>(&tile_indj, &tile_indj_len, n_tile, 1.2f);
  reallocate< tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, n_tile, 1.2f);
#endif

  copy_HtoD_sync<ientry_t>(h_ientry, ientry, n_ientry);
  copy_HtoD_sync<int>(h_tile_indj, tile_indj, n_tile);
  copy_HtoD_sync< tile_excl_t<tilesize> >(h_tile_excl, tile_excl, n_tile);

  delete [] h_ientry;
  delete [] h_tile_indj;
  delete [] h_tile_excl;
}

//
// Explicit instances of NeighborList
//
//template class NeighborList<16>;
template class NeighborList<32>;

template int NeighborList<32>::calc_gpu_pairlist<double>(const int n_ientry, const ientry_t* ientry,
							 const int* tile_indj,
							 const tile_excl_t<32>* tile_excl,
							 const float4* xyzq,
							 const double boxx, const double boxy,
							 const double boxz,
							 const double rcut);

template int NeighborList<32>::calc_cpu_pairlist<double>(const int* zone_patom, const float4* xyzq,
							 const int* loc2glo, const int* atom_excl_pos,
							 const int* atom_excl, const double boxx,
							 const double boxy, const double boxz,
							 const double rcut);

template int NeighborList<32>::calc_gpu_pairlist<float>(const int n_ientry, const ientry_t* ientry,
							const int* tile_indj,
							const tile_excl_t<32>* tile_excl,
							const float4* xyzq,
							const double boxx, const double boxy,
							const double boxz,
							const double rcut);

template int NeighborList<32>::calc_cpu_pairlist<float>(const int* zone_patom, const float4* xyzq,
							const int* loc2glo, const int* atom_excl_pos,
							const int* atom_excl, const double boxx,
							const double boxy, const double boxz,
							const double rcut);
