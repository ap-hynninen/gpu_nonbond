#include <iostream>
#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "NeighborList.h"

//
// Calculates tilex index for each atom
//
__global__ void calc_tilex_ind_kernel(const int istart, const int iend,
				      const float4* __restrict__ xyzq,
				      const int ind0,
				      const int ncellx,
				      const int ncelly,
				      const int ncellz,
				      const float x0,
				      const float y0,
				      const float z0,
				      const float inv_dx,
				      const float inv_dy,
				      const float inv_dz,
				      int* __restrict__ tilex_key,
				      int* __restrict__ tilex_val) {

  const int ind = threadIdx.x + blockIdx.x*blockDim.x + istart;
  
  if (ind <= iend) {
    float4 xyzq_val = xyzq[ind];
    float x = xyzq_val.x;
    float y = xyzq_val.y;
    float z = xyzq_val.z;
    int ix = (int)((x - x0)*inv_dx);
    int iy = (int)((y - y0)*inv_dy);
    int iz = (int)((z - z0)*inv_dz);
    int key = ind0 + (ix + iy*ncellx)*ncellz + iz;

    tilex_key[ind] = key;
    tilex_val[ind] = ind;
  }

}

//
// Sort atoms into z-columns
//
// col_n[0..ncellx*ncelly-1] = number of atoms in each column
// col_ind[istart..iend]     = column index for atoms 
//
__global__ void calc_z_column_index_kernel(const int istart, const int iend,
					   const float4* __restrict__ xyzq,
					   const int ind0,
					   const int ncellx,
					   const int ncelly,
					   const float x0,
					   const float y0,
					   const float inv_dx,
					   const float inv_dy,
					   int* __restrict__ col_n,
					   int* __restrict__ col_ind) {

  const int i = threadIdx.x + blockIdx.x*blockDim.x + istart;
  
  if (i <= iend) {
    float4 xyzq_val = xyzq[i];
    float x = xyzq_val.x;
    float y = xyzq_val.y;
    int ix = (int)((x - x0)*inv_dx);
    int iy = (int)((y - y0)*inv_dy);
    int ind = ind0 + ix + iy*ncellx;
    atomicAdd(&col_n[ind], 1);
    col_ind[i] = ind;
  }
  
}

//
// Computes z column position using parallel exclusive prefix sum
// NOTE: Must have nblock = 1, we loop over buckets to avoid multiple kernel calls
//
__global__ void calc_z_column_pos_kernel(const int ncol_tot,
					 int* __restrict__ col_n,
					 int* __restrict__ col_pos) {
  // Shared memory
  // Requires: blockDim.x*sizeof(int)
  extern __shared__ int shpos[];

  if (threadIdx.x == 0) col_pos[0] = 0;

  int offset = 0;
  for (int base=0;base < ncol_tot;base += blockDim.x) {
    int i = base + threadIdx.x;
    shpos[threadIdx.x] = (i < ncol_tot) ? col_n[i] : 0;
    if (i < ncol_tot) col_n[i] = 0;
    __syncthreads();

    for (int d=1;d < blockDim.x; d *= 2) {
      int tmp = (threadIdx.x >= d) ? shpos[threadIdx.x-d] : 0;
      __syncthreads();
      shpos[threadIdx.x] += tmp;
      __syncthreads();
    }

    if (i < ncol_tot) col_pos[i+1] = shpos[threadIdx.x] + offset;

    offset += shpos[blockDim.x-1];
  }

}

struct keyval_t {
  float key;
  int val;
};

//
// Sorts atoms according to z coordinate
//
// Uses bitonic sort, see:
// http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
//
// Each thread block sorts a single z column
//
__global__ void sort_z_column_kernel(const int* __restrict__ col_pos,
				     float4* __restrict__ xyzq) {

  // Shared memory
  // Requires: blockDim.x*sizeof(keyval_t)
  extern __shared__ keyval_t sh_keyval[];

  int col_pos0 = col_pos[blockIdx.x];
  int n = col_pos[blockIdx.x+1] - col_pos0;

  // Read keys and values into shared memory
  keyval_t keyval;
  keyval.key = (threadIdx.x < n) ? xyzq[threadIdx.x + col_pos0].z : 1.0e38;
  keyval.val = (threadIdx.x < n) ? (threadIdx.x + col_pos0) : (n-1);
  sh_keyval[threadIdx.x] = keyval;
  __syncthreads();

  for (int k = 2;k <= blockDim.x;k *= 2) {
    for (int j = k/2; j > 0;j /= 2) {
      int ixj = threadIdx.x ^ j;
      if (ixj > threadIdx.x && ixj < n) {
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

  float4 xyzq_val;
  if (threadIdx.x < n) xyzq_val = xyzq[sh_keyval[threadIdx.x].val];
  __syncthreads();
  if (threadIdx.x < n) xyzq[threadIdx.x + col_pos0] = xyzq_val;

}


//
// Re-order atoms according to pos
//
__global__ void reorder_atoms_z_column_kernel(const int ncoord,
					      const int* col_ind,
					      int* col_n,
					      const int* col_pos,
					      const float4* __restrict__ xyzq_in,
					      float4* __restrict__ xyzq_out) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  
  if (i < ncoord) {
    int ind = col_ind[i];
    int pos = col_pos[ind];
    int n = atomicAdd(&col_n[ind], 1);
    // new position = pos + n
    float4 xyzq_val = xyzq_in[i];
    xyzq_out[pos+n] = xyzq_val;
  }

}

//
// Re-order atoms according to tilex_val
//
__global__ void reorder_atoms_kernel(const int ncoord,
				     const int* tilex_val,
				     const float4* __restrict__ xyzq_in,
				     float4* __restrict__ xyzq_out) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  
  if (tid < ncoord) {
    int ind = tilex_val[tid];
    float4 xyzq_val = xyzq_in[ind];
    xyzq_out[tid] = xyzq_val;
  }

}

//
//
//
template <int tilesize>
void NeighborList<tilesize>::set_cell_sizes(const int *zonelist,
					    const float3 *max_xyz, const float3 *min_xyz,
					    int *ncellx, int *ncelly, int *ncellz,
					    float *celldx, float *celldy, float *celldz) {

  for (int izone=0;izone < 8;izone++) {
    int nstart;
    if (izone > 0) {
      nstart = zonelist[izone-1] + 1;
    } else {
      nstart = 1;
    }
    // ncoord_zone = number of atoms in this zone
    int ncoord_zone = zonelist[izone] - nstart + 1;
    if (ncoord_zone > 0) {
      // NOTE: we increase the cell sizes here by 0.001 to make sure no atoms drop outside cells
      float xsize = max_xyz[izone].x - min_xyz[izone].x + 0.001f;
      float ysize = max_xyz[izone].y - min_xyz[izone].y + 0.001f;
      float zsize = max_xyz[izone].z - min_xyz[izone].z + 0.001f;
      float delta = powf(xsize*ysize*zsize*tilesize/(float)ncoord_zone, 1.0f/3.0f);
      ncellx[izone] = max(1, (int)(xsize/delta));
      ncelly[izone] = max(1, (int)(ysize/delta));
      // Approximation for ncellz = "uniform distribution of atoms"
      ncellz[izone] = max(1, ncoord_zone/(ncellx[izone]*ncelly[izone]*tilesize));
      celldx[izone] = xsize/(float)(ncellx[izone]);
      celldy[izone] = ysize/(float)(ncelly[izone]);
      celldz[izone] = zsize/(float)(ncellz[izone]);
    } else {
      ncellx[izone] = 0;
      ncelly[izone] = 0;
      ncellz[izone] = 0;
      celldx[izone] = 1.0f;
      celldy[izone] = 1.0f;
      celldz[izone] = 1.0f;
    }
  }

}

//
// Sorts atoms into tiles
//
template <int tilesize>
void NeighborList<tilesize>::sort(const int *zonelist,
				  const float3 *max_xyz, const float3 *min_xyz,
				  float4 *xyzq,
				  float4 *xyzq_sorted,
				  cudaStream_t stream) {

  int ncellx[8], ncelly[8], ncellz[8];
  float celldx[8], celldy[8], celldz[8];
  float inv_dx[8], inv_dy[8], inv_dz[8];

  int ncoord = zonelist[7];

  int nthread = 512;
  int nblock = (ncoord-1)/nthread+1;

  set_cell_sizes(zonelist, max_xyz, min_xyz, ncellx, ncelly, ncellz, celldx, celldy, celldz);

  int ncol_tot = 0;
  for (int i=0;i < 8;i++) ncol_tot += ncellx[i]*ncelly[i];

  reallocate<int>(&col_n, &col_n_len, ncol_tot, 1.2f);
  reallocate<int>(&col_pos, &col_pos_len, ncol_tot+1, 1.2f);
  reallocate<int>(&col_ind, &col_ind_len, ncoord, 1.2f);

  clear_gpu_array<int>(col_n, ncol_tot, stream);

  for (int izone=0;izone < 8;izone++) {
    inv_dx[izone] = 1.0f/celldx[izone];
    inv_dy[izone] = 1.0f/celldy[izone];
    inv_dz[izone] = 1.0f/celldz[izone];
  }

  //
  // Calculate number of atoms in each z-column
  //
  int ind0 = 0;
  for (int izone=0;izone < 8;izone++) {
    int istart, iend;
    if (izone > 0) {
      istart = zonelist[izone-1];
    } else {
      istart = 0;
    }
    iend = zonelist[izone] - 1;
    if (iend >= istart) {

      calc_z_column_index_kernel<<< nblock, nthread, 0, stream >>>
	(istart, iend, xyzq, ind0, ncellx[izone], ncelly[izone], 
	 min_xyz[izone].x, min_xyz[izone].y,
	 inv_dx[izone], inv_dy[izone], col_n, col_ind);
      cudaCheck(cudaGetLastError());

      ind0 += ncellx[izone]*ncelly[izone];
    }
  }

  /*
  thrust::device_ptr<int> col_n_ptr(col_n);
  thrust::device_ptr<int> col_pos_ptr(col_pos);
  thrust::exclusive_scan(col_n_ptr, col_n_ptr + ncol_tot, col_pos_ptr);
  clear_gpu_array<int>(col_n, ncol_tot, stream);
  */

  /*
  {
    int *h_tmp = new int[ncol_tot];
    copy_DtoH<int>(col_n, h_tmp, ncol_tot);
    for (int i=0;i < ncol_tot;i++)
      std::cout << h_tmp[i] << " ";
    std::cout << std::endl;
    delete [] h_tmp;
  }
  */

  //
  // Calculate positions
  //
  nthread = min(((ncol_tot-1)/32+1)*32, get_max_nthread());
  //std::cout << "nthread = " << nthread << std::endl;
  int shmem_size = nthread*sizeof(int);
  calc_z_column_pos_kernel<<< 1, nthread, shmem_size, stream >>>(ncol_tot, col_n, col_pos);

  /*
  std::cout << "--------------------------------------------------------" << std::endl;
  {
    int *h_tmp = new int[ncol_tot];
    copy_DtoH<int>(col_pos, h_tmp, ncol_tot);
    for (int i=0;i < ncol_tot;i++)
      std::cout << h_tmp[i] << " ";
    std::cout << std::endl;
    delete [] h_tmp;
  }
  */

  nthread = 512;
  nblock = (ncoord-1)/nthread+1;
  reorder_atoms_z_column_kernel<<< nblock, nthread, 0, stream >>>
    (ncoord, col_ind, col_n, col_pos, xyzq, xyzq_sorted);
  cudaCheck(cudaGetLastError());

  // Now sort according to z coordinate
  nthread = 11*tilesize;
  nblock = ncellx[0]*ncelly[0];
  if (nthread < get_max_nthread()) {
    shmem_size = nthread*sizeof(keyval_t);
    sort_z_column_kernel<<< nblock, nthread, shmem_size, stream >>>
      (col_pos, xyzq_sorted);
    cudaCheck(cudaGetLastError());
  } else {
    std::cerr << "Neighborlist::sort, this version of sort_z_column_kernel not implemented yet"
	      << std::endl;
  }

  //  reorder_atoms_kernel<<< nblock, nthread, 0, stream >>>
  //    (ncoord, tilex_val, xyzq, xyzq_sorted);
  //cudaCheck(cudaGetLastError());

}

//
// Calculates bounding box
//
template <int tilesize>
__global__ void calc_bounding_box_kernel(const int ncell,
					 const int* __restrict__ cell_start,
					 const float4* __restrict__ xyzq,
					 bb_t* __restrict__ bb) {

  const int icell = threadIdx.x + blockIdx.x*blockDim.x;

  if (icell < ncell) {
    int base = cell_start[icell];
    float4 xyzq_val = xyzq[base];
    float x0 = xyzq_val.x;
    float y0 = xyzq_val.y;
    float z0 = xyzq_val.z;
    float x1 = xyzq_val.x;
    float y1 = xyzq_val.y;
    float z1 = xyzq_val.z;
    for (int i=1;i < tilesize;i++) {
      xyzq_val = xyzq[base + i];
      x0 = min(x0, xyzq_val.x);
      y0 = min(y0, xyzq_val.y);
      z0 = min(z0, xyzq_val.z);
      x1 = max(x1, xyzq_val.x);
      y1 = max(y1, xyzq_val.y);
      z1 = max(z1, xyzq_val.z);
    }
    bb[icell].x = 0.5f*(x0 + x1);
    bb[icell].y = 0.5f*(y0 + y1);
    bb[icell].z = 0.5f*(z0 + z1);
    bb[icell].wx = 0.5f*(x1 - x0);
    bb[icell].wy = 0.5f*(y1 - y0);
    bb[icell].wz = 0.5f*(z1 - z0);
  }

}

//
// Calculates bounding boxes for tiles
//
template <int tilesize>
void NeighborList<tilesize>::calc_bounding_box(const int ncell,
					       const int *cell_start,
					       const float4 *xyzq,
					       cudaStream_t stream) {
  int nthread = 512;
  int nblock = (ncell-1)/nthread+1;

  calc_bounding_box_kernel<tilesize> <<< nblock, nthread >>>
    (ncell, cell_start, xyzq, bb);

  cudaCheck(cudaGetLastError());
}

//#######################################################################

//
// Class creator
//
template <int tilesize>
NeighborList<tilesize>::NeighborList() {
  ni = 0;
  ntot = 0;

  tile_excl = NULL;
  tile_excl_len = 0;

  ientry = NULL;
  ientry_len = 0;

  tile_indj = NULL;
  tile_indj_len = 0;

  // Sparse
  ni_sparse = 0;
  ntot_sparse = 0;

  pairs_len = 0;
  pairs = NULL;
  
  ientry_sparse_len = 0;
  ientry_sparse = NULL;

  tile_indj_sparse_len = NULL;
  tile_indj_sparse = NULL;

  // Neighbor list building
  col_n_len = 0;
  col_n = NULL;

  col_pos_len = 0;
  col_pos = NULL;

  col_ind_len = 0;
  col_ind = NULL;
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
  if (col_n != NULL) deallocate<int>(&col_n);
  if (col_pos != NULL) deallocate<int>(&col_pos);
  if (col_ind != NULL) deallocate<int>(&col_ind);
}

//
// Sets ientry from host memory array
//
template <int tilesize>
void NeighborList<tilesize>::set_ientry(int ni, ientry_t *h_ientry, cudaStream_t stream) {

  this->ni = ni;

  // Allocate & reallocate d_ientry
  reallocate<ientry_t>(&ientry, &ientry_len, ni, 1.4f);

  // Copy to device
  copy_HtoD<ientry_t>(h_ientry, ientry, ni, stream);
}

//----------------------------------------------------------------------------------------
//
// Builds neighborlist
//

struct cell_t {
  int izone;
  int icellx;
  int icelly;
  int icellz;
};

#ifdef NOTREADY

//
// The entire warp enters here
// If IvsI = true, search within I zone
//
template <bool IvsI>
__device__ void get_cell_bounds(const int izone, const int jzone, const int icell, const int ncell,
				const float x0, const float x1, const float* bx, const float rcut,
				int& jcell0, int& jcell1, float *dist) {

  int jcell_start_left, jcell_start_right;

  if (izone == jzone) {
    // Search within a single zone (I)
    if (icell < 0) {
      // This is one of the image cells on the left =>
      // set the left cell boundary (jcell0) to 1 and start looking for the right
      // boundary from 1
      jcell_start_left = 0;       // with this value, we don't look for cells on the left
      jcell_start_right = 1;      // start looking for cells at right from 1
      jcell0 = 1;                  // left boundary set to minimum value
      jcell1 = 0;                    // set to "no cells" value
      dist[1] = 0.0f;
    } else if (icell >= ncell) {
      // This is one of the image cells on the right =>
      // set the right cell boundary (icell1) to ncell and start looking for the left
      // boundary from ncell
      jcell_start_left = ncell;      // start looking for cells at left from ncell
      jcell_start_right = ncell + 1; // with this value, we don't look for cells on the right
      jcell0 = ncell + 1;            // set to "no cells" value
      jcell1 = ncell;                // right boundary set to maximum value
      dist[ncell] = 0.0f;
    } else {
      jcell_start_left = icell - 1;
      jcell_start_right = icell + 1;
      jcell0 = icell;
      jcell1 = icell;
      dist[icell] = 0.0f;
    }
  } else {
    if (bx(0) >= x1 || (bx(0) < x1 && bx(0) > x0)) {
      // j-zone is to the right of i-zone
      // => no left search, start right search from 1
      jcell_start_left = 0;
      jcell_start_right = 1;
      jcell0 = 1;
      jcell1 = 0;
    } else if (bx[ncell] <= x0 || (bx[ncell] > x0 && bx[ncell] < x1)) {
      // j-zone is to the left of i-zone
      // => no right search, start left search from ncell
      jcell_start_left = ncell;
      jcell_start_right = ncell + 1;
      jcell0 = ncell + 1;
      jcell1 = ncell;
    } else {
      // i-zone is between j-zones
      // => safe choice is to search the entire range
      jcell_start_left = ncell;
      jcell_start_right = 1;
      jcell0 = ncell;
      jcell1 = 1;
    }
  }

  // Check cells at left, stop once the distance to the cell right boundary 
  // is greater than the cutoff.
  //
  // Cell right boundary is at bx(i)
  for (int j=jcell_start_left;j >= 1;j--) {
    float d = x0 - bx[j];
    if (d > cut) break;
    dist[j] = max(0.0f, d);
    jcell0 = j;
  }

  // Check cells at right, stop once the distance to the cell left boundary
  // is greater than the cutoff.
  //
  // Cell left boundary is at bx(i-1)
  for (int j=jcell_start_right;j <= ncell;j++) {
    float d = bx[j-1] - x1;
    if (d > cut) break;
    dist[j] = max(0.0f, d);
    jcell1 = j;
  }

  // Cell bounds are jcell0:jcell1
      
}

//
// Build neighborlist for one zone at the time
// One warp takes care of one cell
//
template < int tilesize, bool IvsI >
__global__
void build_nlist_kernel(const int ncell, const int izone, const int n_jzone,
			const int *cellx, const int *celly, const int *cellz,
			const bb_t * bb,
			const float *cellbx, const float *cellby, const float *cellbz) {

  // Shared memory
  extern __shared__ char shbuf[];
  volatile int *jcellx0;
  volatile int *jcelly0;
  volatile int *jcellz0;
  volatile int *jcellx1;
  volatile int *jcelly1;
  volatile int *jcellz1;

  // Index of the i-cell
  const int icell = (threadId.x + blockIdx.x*blockDim.x)/WARPSIZE;

  if (icell >= ncell) return;

  int icellx = cellx[icell];
  int icelly = celly[icell];
  int icellz = cellz[icell];

  bb_t ibb = bb[icell];

  for (int imx=imx_lo;imx <= imx_hi;imx++) {
    float imbbx0 = ibb.x + imx*boxx;
    int n_jcellx = 0;
    for (int jjzone=0;jjzone < n_jzone;jjzone++) {
      int jzone = int_zone[izone][jjzone];
      int jcellx0_t, jcellx1_t;
      get_cell_bounds<IvsI>(izone, jzone, icellx + imx*ncellx[izone], ncellx[jzone],
			    imbbx0-ibb.wx, imbbx0+ibb.wx, cellbx[jzone], rcut,
			    jcellx0_t, jcellx1_t);
      n_jcellx += max(0, jcellx1_t-jcellx0_t+1);
      jcellx0[jzone] = jcellx0_t;
      jcellx1[jzone] = jcellx1_t;
    }

    for (int imy=imy_lo;imy <= imy_hi;imy++) {
      float imbby0 = ibb.y + imy*boxy;
      int n_jcelly = 0;
      for (int jjzone=0;jjzone < n_jzone;jjzone++) {
	int jzone = int_zone[izone][jjzone];
	int jcelly0_t, jcelly1_t;
	get_cell_bounds<IvsI>(izone, jzone, icelly + imy*ncelly[izone], ncelly[jzone],
			      imbby0-ibb.wy, imbby0+ibb.wy, cellby[jzone], rcut,
			      jcelly0_t, jcelly1_t);
	n_jcelly += max(0, jcelly1_t-jcelly0_t+1);
	jcelly0[jzone] = jcelly0_t;
	jcelly1[jzone] = jcelly1_t;
      }
    } // for (int imy=imy_lo;imy <= imy_hi;imy++)

    for (int imz=imz_lo;imz <= imz_hi;imz++) {
	float imbbz0 = ibb.z + imz*boxz;
	
	int ish = imx+1 + 3*(imy+1 + 3*(imz+1));
	
	for (int jjzone=0;jjzone < n_jzone;jjzone++) {
	  int jzone = int_zone[izone][jjzone];


	  if (jcelly1[jzone] >= jcelly0[jzone] && jcellx1[jzone] >= jcellx0[jzone]) {
	    // Loop over j-cells
	    // NOTE: we do this in order y, x, z so that the resulting tile list
	    //       is ordered
	    for (int jcelly=jcelly0[jzone]; jcelly <= jcelly1(jzone);jcelly++) {
	      float celldist1 = ydist[ydist_pos + jcelly];
	      celldist1 *= celldist1;
	      jcellx0_t = jcellx0[jzone];
	      for (int jcellx=jcellx0_t; jcellx <= jcellx1[jzone]; jcellx++) {
		float celldist2 = celldist1 + xdist[xdist_pos + jcellx];
		celldist2 *= celldist2;
		if (celldist2 > cutsq) continue;
		// Get jcellz limits (jcellz0, jcellz1)
		pos_xy = jcellx + (jcelly-1)*ncellx[jzone];
		pos_cellbz = (max_ncellz(jzone)+1)*(pos_xy - 1);
		pos_ncellz = pos_xy + startcol_zone[jzone];
		get_cell_bounds<IvsI>(izone, jzone, icellz_im,
				      ncellz[pos_ncellz], imbbz0-ibb.wz, imbbz0+ibb.wz,
				      cellbz[jzone]%array(pos_cellbz:), cut, jcellz0, jcellz1, zdist);
		for (int jcellz=jcellz0; jcellz <= jcellz1; jcellz++) {
		  if (celldist2 + zdist(jcellz)**2 > cutsq) continue;
		  // j-cell index is calculated as jcellz + start of the column cells
		  jcell = jcellz + startcell_col[pos_ncellz];

		  // Read bounding box for j-cell
		  bb_t jbb = bb[jcell];
                               
		  // Calculate distance between i- and j-cell bounding boxes
		  float bbxdist = max(0.0f, fabs(imbbx0 - jbb.x) - ibb.wx - jbb.wx);
		  float bbydist = max(0.0f, fabs(imbby0 - jbb.y) - ibb.wy - jbb.wy);
		  float bbzdist = max(0.0f, fabs(imbbz0 - jbb.z) - ibb.wz - jbb.wz);

		  if (bbxdist**2 + bbydist**2 + bbzdist**2 < cutsq) {
		  }


	}
	
    } // for (int imz=imz_lo;imz <= imz_hi;imz++)


  } // for (int imx=imx_lo;imx <= imx_hi;imx++)

}

template <int tilesize>
void NeighborList<tilesize>::build_nlist(const float boxx, const float boxy, const float boxz,
					 const float roff,
					 const int n_ijlist, const int3 *ijlist,
					 const int *cell_start,
					 const float4 *xyzq,
					 cudaStream_t stream) {

  build_nlist_kernel<tilesize, true>
    <<< nblock, nthread, shmem_size, stream >>>
    ();

  build_nlist_kernel<tilesize, false>
    <<< nblock, nthread, shmem_size, stream >>>
    ();

}
#endif // NOTREADY

//----------------------------------------------------------------------------------------
//
// Builds tilex exclusion mask from ijlist[] based on distance and index
// Builds exclusion mask based on atom-atom distance and index (i >= j excluded)
//
// Uses 32 threads to calculate the distances for a single ijlist -entry.
//
const int nwarp_build_excl_dist = 8;

template < int tilesize >
__global__ void build_excl_kernel(const unsigned int base_tid, const int n_ijlist, const int3 *ijlist,
				  const int *cell_start, const float4 *xyzq,
				  int *tile_indj,
				  tile_excl_t<tilesize> *tile_excl,
				  const float boxx, const float boxy, const float boxz,
				  const float roff2) {
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
  float3 *sh_xyzi = (float3 *)&shmem[0];    // nwarp_build_excl_dist*tilesize
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

  int istart = cell_start[icell] - 1;
  int iend   = cell_start[icell+1] - 2;

  int jstart = cell_start[jcell] - 1;
  int jend   = cell_start[jcell+1] - 2;

  const unsigned int load_ij = threadIdx.x % tilesize;
  const int sh_start = (threadIdx.x/warpsize)*tilesize;

  // Load atom i coordinates to shared memory
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
  __syncthreads();

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
      excl |= ((r2 >= roff2) | (q_samecell && (tid <= i)) ) << t;
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
      excl |= ((r2 >= roff2) | (q_samecell && (load_ij <= load_i)) ) << t;
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

/*
void test_excl_dist_index(const int n_ijlist, const int3 *d_ijlist,
			  const int *d_cell_start, const float4 *d_xyzq,
			  int *d_tile_indj,
			  tile_excl_t *d_tile_excl,
			  const float boxx, const float boxy, const float boxz,
			  const float roff2) {

  int3 *h_ijlist;
  int *h_cell_start;
  float4 *h_xyzq;
  int *h_tile_indj;
  tile_excl_t *h_tile_excl;

  h_ijlist = (int3 *)malloc(n_ijlist*sizeof(int3));
  h_cell_start = (int *)malloc(mdsim.ncell*sizeof(int));
  h_xyzq = (float4 *)malloc(mdsim.ncoord*sizeof(float4));
  h_tile_indj = (int *)malloc(n_ijlist*sizeof(int));
  h_tile_excl = (tile_excl_t *)malloc(n_ijlist*sizeof(tile_excl_t));

  cudaCheck(cudaMemcpy(h_ijlist, d_ijlist, sizeof(int3)*n_ijlist,
		       cudaMemcpyDeviceToHost));

  cudaCheck(cudaMemcpy(h_cell_start, d_cell_start, sizeof(int)*mdsim.ncell,
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

    int istart = h_cell_start[icell] - 1;
    int iend   = h_cell_start[icell+1] - 2;

    int jstart = h_cell_start[jcell] - 1;
    int jend   = h_cell_start[jcell+1] - 2;

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
	unsigned int excl = ((r2 >= roff2) | (q_samecell && (j <= i)) ) << shbit;
	unsigned int excl_gpu = h_tile_excl[wid].excl[iexcl];
	if ( ((excl_gpu & ibit) ^ excl) != 0 && fabsf(r2-roff2) > 7.0e-5) {
	  printf("Error found in test_excl_dist_index:\n");
	  printf("wid = %d i = %d j = %d iexcl = %d shbit = %d\n",wid,i,j,iexcl,shbit);
	  printf("ii = %d jj = %d %d %d %d %d\n",ii,jj,r2 >= roff2,
		 (q_samecell && (j <= i)),icell,jcell);
	  printf("%x ^ %x = %x \n",excl_gpu & ibit, excl, (excl_gpu & ibit) ^ excl);
	  printf("i:  %f %f %f\n",xi,yi,zi);
	  printf("j:  %f %f %f\n",xj,yj,zj);
	  printf("jo: %f %f %f\n",xyzq_j.x,xyzq_j.y,xyzq_j.z);
	  printf("sh: %f %f %f\n",xsh,ysh,zsh);
	  printf("dx: %1.8f %1.8f %1.8f\n",dx,dy,dz);
	  printf("r2: %f %e\n",r2,fabsf(r2-roff2));
	  exit(1);
	}
      }
    }

  }

  free(h_ijlist);
  free(h_cell_start);
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
					const float roff,
					const int n_ijlist, const int3 *ijlist,
					const int *cell_start,
					const float4 *xyzq,
					cudaStream_t stream) {

  if (n_ijlist == 0) return;

  // Allocate & re-allocate (d_tile_indj, d_tile_excl)
  reallocate<int>(&tile_indj, &tile_indj_len, n_ijlist, 1.2f);
  reallocate<tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, n_ijlist, 1.2f);

  float roff2 = roff*roff;

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
      (base_tid, n_ijlist, ijlist, cell_start,
       xyzq, tile_indj, tile_excl,
       boxx, boxy, boxz,
       roff2);

    base_tid += nblock*nthread;

    cudaCheck(cudaGetLastError());
  }

  /*
  if (mdsim.q_test != 0) {
    test_excl_dist_index(mdsim.n_ijlist, mdsim.ijlist, mdsim.cell_start,
			 mdsim.xyzq.xyzq, mdsim.tile_indj, mdsim.tile_excl,
			 boxx, boxy, boxz,
			 roff2);
  }
  */

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
// Splits neighbor list into dense and sparse parts
//
template <int tilesize>
void NeighborList<tilesize>::split_dense_sparse(int npair_cutoff) {

  ientry_t *h_ientry = new ientry_t[ni];
  int *h_tile_indj = new int[ntot];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[ntot];

  ientry_t *h_ientry_dense = new ientry_t[ni];
  int *h_tile_indj_dense = new int[ntot];
  tile_excl_t<tilesize> *h_tile_excl_dense = new tile_excl_t<tilesize>[ntot];

  ientry_t *h_ientry_sparse = new ientry_t[ni];
  int *h_tile_indj_sparse = new int[ntot];
  pairs_t<tilesize> *h_pairs = new pairs_t<tilesize>[ntot];

  copy_DtoH<ientry_t>(ientry, h_ientry, ni);
  copy_DtoH<int>(tile_indj, h_tile_indj, ntot);
  copy_DtoH< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, ntot);

  int ni_dense = 0;
  int ntot_dense = 0;
  ni_sparse = 0;
  ntot_sparse = 0;
  for (int i=0;i < ni;i++) {
    bool sparse_i_tiles = true;
    int startj_dense = ntot_dense;
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
	h_tile_indj_sparse[ntot_sparse] = h_tile_indj[j];
	ntot_sparse++;
      } else {
	// Dense
	for (int k=0;k < (num_excl<tilesize>::val);k++) {
	  h_tile_excl_dense[ntot_dense].excl[k] = h_tile_excl[j].excl[k];
	}
	h_tile_indj_dense[ntot_dense] = h_tile_indj[j];
	ntot_dense++;
	sparse_i_tiles = false;
      }

    }

    if (sparse_i_tiles) {
      // Sparse
    } else {
      h_ientry_dense[ni_dense] = h_ientry[i];
      h_ientry_dense[ni_dense].startj = startj_dense;
      h_ientry_dense[ni_dense].endj = ntot_dense - 1;
      ni_dense++;
    }
  }

  ni = ni_dense;
  ntot = ntot_dense;

  copy_HtoD<ientry_t>(h_ientry_dense, ientry, ni);
  copy_HtoD<int>(h_tile_indj_dense, tile_indj, ntot);
  copy_HtoD< tile_excl_t<tilesize> >(h_tile_excl_dense, tile_excl, ntot);

  allocate<ientry_t>(&ientry_sparse, ni_sparse);
  allocate<int>(&tile_indj_sparse, ntot_sparse);
  allocate< pairs_t<tilesize> >(&pairs, ntot_sparse);
  ientry_sparse_len = ni_sparse;
  tile_indj_sparse_len = ntot_sparse;
  pairs_len = ntot_sparse;

  copy_HtoD<ientry_t>(h_ientry_sparse, ientry_sparse, ni_sparse);
  copy_HtoD<int>(h_tile_indj_sparse, tile_indj_sparse, ntot_sparse);
  copy_HtoD< pairs_t<tilesize> >(h_pairs, pairs, ntot_sparse);

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

  ientry_t *h_ientry = new ientry_t[ni];
  int *h_tile_indj = new int[ntot];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[ntot];

  ientry_t *h_ientry_noempty = new ientry_t[ni];
  int *h_tile_indj_noempty = new int[ntot];
  tile_excl_t<tilesize> *h_tile_excl_noempty = new tile_excl_t<tilesize>[ntot];

  copy_DtoH<ientry_t>(ientry, h_ientry, ni);
  copy_DtoH<int>(tile_indj, h_tile_indj, ntot);
  copy_DtoH< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, ntot);

  int ni_noempty = 0;
  int ntot_noempty = 0;
  for (int i=0;i < ni;i++) {
    bool empty_i_tiles = true;
    int startj_noempty = ntot_noempty;
    for (int j=h_ientry[i].startj;j <= h_ientry[i].endj;j++) {
      bool empty_tile = true;
      for (int k=0;k < (num_excl<tilesize>::val);k++) {
	unsigned int n1bit = BitCount(h_tile_excl[j].excl[k]);
	if (n1bit != 32) empty_tile = false;
      }

      if (!empty_tile) {
	for (int k=0;k < (num_excl<tilesize>::val);k++) {
	  h_tile_excl_noempty[ntot_noempty].excl[k] = h_tile_excl[j].excl[k];
	}
	h_tile_indj_noempty[ntot_noempty] = h_tile_indj[j];
	ntot_noempty++;
	empty_i_tiles = false;
      }
    }

    if (!empty_i_tiles) {
      h_ientry_noempty[ni_noempty] = h_ientry[i];
      h_ientry_noempty[ni_noempty].startj = startj_noempty;
      h_ientry_noempty[ni_noempty].endj = ntot_noempty - 1;
      ni_noempty++;
    }
  }

  ni = ni_noempty;
  ntot = ntot_noempty;

  copy_HtoD<ientry_t>(h_ientry_noempty, ientry, ni);
  copy_HtoD<int>(h_tile_indj_noempty, tile_indj, ntot);
  copy_HtoD< tile_excl_t<tilesize> >(h_tile_excl_noempty, tile_excl, ntot);

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

  ientry_t *h_ientry = new ientry_t[ni];
  int *h_tile_indj = new int[ntot];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[ntot];

  copy_DtoH<ientry_t>(ientry, h_ientry, ni);
  copy_DtoH<int>(tile_indj, h_tile_indj, ntot);
  copy_DtoH< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, ntot);

  std::cout << "Number of i-tiles = " << ni << ", total number of tiles = " << ntot << std::endl;

  std::ofstream file_npair("npair.txt", std::ofstream::out);
  std::ofstream file_nj("nj.txt", std::ofstream::out);

  unsigned int nexcl_bit = 0;
  unsigned int nexcl_bit_self = 0;
  unsigned int nempty_tile = 0;
  unsigned int nempty_line = 0;
  for (int i=0;i < ni;i++) {
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
    }
  }

  file_npair.close();
  file_nj.close();

  unsigned int ntot_pairs = ntot*tilesize*tilesize;
  std::cout << "Total number of pairs = " << ntot_pairs << std::endl;
  std::cout << "Number of excluded pairs = " << nexcl_bit << " (" << 
    ((double)nexcl_bit*100)/(double)ntot_pairs << "%)" << std::endl;
  std::cout << "Number of excluded pairs in self (i==j) tiles = " << nexcl_bit_self << " (" << 
    ((double)nexcl_bit_self*100)/(double)ntot_pairs << "%)" << std::endl;
  std::cout << "Number of empty lines = " << nempty_line << " (" <<
    ((double)nempty_line*100)/((double)(ntot*tilesize)) << "%)" << std::endl;
  std::cout << "Number of empty tiles = " << nempty_tile << " (" <<
    ((double)nempty_tile*100)/(double)ntot << "%)" << std::endl;

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

    file >> ni >> ntot;

    h_ientry = new ientry_t[ni];
    h_tile_indj = new int[ntot];
    h_tile_excl = new tile_excl_t<tilesize>[ntot];

    for (int i=0;i < ni;i++) {
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

  reallocate<ientry_t>(&ientry, &ientry_len, ni, 1.2f);
  reallocate<int>(&tile_indj, &tile_indj_len, ntot, 1.2f);
  reallocate< tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, ntot, 1.2f);

  copy_HtoD<ientry_t>(h_ientry, ientry, ni);
  copy_HtoD<int>(h_tile_indj, tile_indj, ntot);
  copy_HtoD< tile_excl_t<tilesize> >(h_tile_excl, tile_excl, ntot);

  delete [] h_ientry;
  delete [] h_tile_indj;
  delete [] h_tile_excl;
}

//
// Explicit instances of DirectForce
//
template class NeighborList<16>;
template class NeighborList<32>;
