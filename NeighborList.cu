#include <iostream>
#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "NeighborList.h"

static __device__ NeighborListParam_t d_nlist_param;

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
    if (i < d_nlist_param.zone_patom[izone]) {
      float4 xyzq_val = xyzq[i];
      float x = xyzq_val.x;
      float y = xyzq_val.y;
      float3 minxyz = d_nlist_param.minxyz[izone];
      int ix = (int)((x - minxyz.x)*d_nlist_param.inv_celldx[izone]);
      int iy = (int)((y - minxyz.y)*d_nlist_param.inv_celldy[izone]);
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

/*
//
// Computes z column position using parallel exclusive prefix sum
//
// NOTE: Must have nblock = 1, we loop over buckets to avoid multiple kernel calls
//
__global__ void calc_z_column_pos_kernel(const int ncol_tot,
					 int* __restrict__ col_natom,
					 int* __restrict__ col_patom) {
  // Shared memory
  // Requires: blockDim.x*sizeof(int)
  extern __shared__ int shpos[];

  if (threadIdx.x == 0) col_patom[0] = 0;

  int offset = 0;
  for (int base=0;base < ncol_tot;base += blockDim.x) {
    int i = base + threadIdx.x;
    shpos[threadIdx.x] = (i < ncol_tot) ? col_natom[i] : 0;
    if (i < ncol_tot) col_natom[i] = 0;
    __syncthreads();

    for (int d=1;d < blockDim.x; d *= 2) {
      int tmp = (threadIdx.x >= d) ? shpos[threadIdx.x-d] : 0;
      __syncthreads();
      shpos[threadIdx.x] += tmp;
      __syncthreads();
    }

    // Write result into global memory
    if (i < ncol_tot) col_patom[i+1] = shpos[threadIdx.x] + offset;

    offset += shpos[blockDim.x-1];
  }

}
*/

//
// Computes z column position using parallel exclusive prefix sum
// Also computes the cell_patom, col_ncellz, and ncell
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
    col_ncellz[i] = tmpval.y;                      // Set col_ncellz[icol]
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
      cell_xyz_zone_val.x = col_xy_zone_val.x;
      cell_xyz_zone_val.y = col_xy_zone_val.y;
      cell_xyz_zone_val.z = 0;
      cell_xyz_zone_val.w = col_xy_zone_val.z;
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

  // Write ncell into global GPU buffer
  if (threadIdx.x == 0) {
    d_nlist_param.ncell = offset.y;
  }

}

//
// Calculates ncellz_max[izone].
//
// blockDim.x = max number of columns over all zones
// Each thread block calculates one zone (blockIdx.x = izone)
//
__global__ void calc_ncellz_max(const int* __restrict__ col_ncellz) {

  // Shared memory
  // Requires: blockDim.x*sizeof(int)
  extern __shared__ int sh_col_ncellz[];

  // ncol[izone] gives the cumulative sum of ncellx[izone]*ncelly[izone]
  int start = d_nlist_param.ncol[blockIdx.x];
  int end   = d_nlist_param.ncol[blockIdx.x+1];
  int n = end - start - 1;

  if (n > 0) {
    // Load col_ncellz into shared memory
    int col_ncellz_val = 0;
    if (threadIdx.x < end) col_ncellz_val = col_ncellz[start + threadIdx.x];
    sh_col_ncellz[threadIdx.x] = col_ncellz_val;
    __syncthreads();
    
    // Reduce
    for (int d=1;d < n;d *= 2) {
      int t = threadIdx.x + d;
      int val = (t < n) ? sh_col_ncellz[t] : 0;
      __syncthreads();
      sh_col_ncellz[threadIdx.x] = max(sh_col_ncellz[threadIdx.x], val);
      __syncthreads();
    }
    
    // Write into global memory
    if (threadIdx.x == 0) {
      d_nlist_param.ncellz_max[blockIdx.x] = sh_col_ncellz[0];
    }
  } else {
    if (threadIdx.x == 0) {
      d_nlist_param.ncellz_max[blockIdx.x] = 0;
    }
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
					      int* __restrict__ loc2glo_ind) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  
  if (i < ncoord) {
    int icol = atom_icol[i];
    int pos = col_patom[icol];
    int n = atomicAdd(&col_natom[icol], 1);
    // new position = pos + n
    int newpos = pos + n;
    loc2glo_ind[newpos] = i;
    float4 xyzq_val = xyzq_in[i];
    xyzq_out[newpos] = xyzq_val;
  }

}

/*
//
// Re-order atoms according to pos. Deterministic version. Single thread block only.
//
__global__ void reorder_atoms_z_column2_kernel(const int ncoord,
					       const int* atom_icol,
					       int* col_natom,
					       const int* col_patom,
					       const float4* __restrict__ xyzq_in,
					       float4* __restrict__ xyzq_out,
					       int* __restrict__ loc2glo_ind) {
  // Shared memory
  // Requires: blockDim.x*sizeof(int)
  extern __shared__ int sh_n[];
  
  for (int ibase=0;ibase < ncoord;ibase+=blockDim.x) {
    int i = ibase + threadIdx.x;
    if (i < ncoord) {
      int icol = atom_icol[i];
      int pos = col_patom[icol];
    }

    sh_

    int n = atomicAdd(&col_natom[icol], 1);

    // new position = pos + n
    int newpos = pos + n;
    loc2glo_ind[newpos] = i;
    float4 xyzq_val = xyzq_in[i];
    xyzq_out[newpos] = xyzq_val;
  }

}
*/

//
// Sorts atoms according to z coordinate
//
// Uses bitonic sort, see:
// http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
//
// Each thread block sorts a single z column
//
struct keyval_t {
  float key;
  int val;
};
__global__ void sort_z_column_kernel(const int* __restrict__ col_patom,
				     float4* __restrict__ xyzq,
				     int* __restrict__ loc2glo_ind) {

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
  // loc2glo_ind_new[threadIdx.x + col_patom0] = loc2glo_ind[sh_keyval[threadIdx.x].val]
  //

  float4 xyzq_val;
  int ind_val;
  if (threadIdx.x < n) {
    int i = sh_keyval[threadIdx.x].val;
    ind_val = loc2glo_ind[i];
    xyzq_val = xyzq[i];
  }
  __syncthreads();
  if (threadIdx.x < n) {
    int newpos = threadIdx.x + col_patom0;
    xyzq[newpos] = xyzq_val;
    loc2glo_ind[newpos] = ind_val;
  }

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
      celldz_min[izone] = ysize/(float)(ncellz_max[izone]);
    } else {
      ncellx[izone] = 0;
      ncelly[izone] = 0;
      celldx[izone] = 1.0f;
      celldy[izone] = 1.0f;
      celldz_min[izone] = 1.0f;
    }
  }

  std::cout << "celldx = " << celldx[0] << " ncellx[0] = " << ncellx[0] 
	    << " xsize = " << max_xyz[0].x - min_xyz[0].x + 0.001f << std::endl;

  std::cout << "celldy = " << celldy[0] << " ncelly[0] = " << ncelly[0] 
	    << " ysize = " << max_xyz[0].y - min_xyz[0].y + 0.001f << std::endl;

}

//
// Tests for z columns
//
template <int tilesize>
bool NeighborList<tilesize>::test_z_columns(const int* zone_patom,
					    const int* ncellx, const int* ncelly,
					    const int ncol_tot,
					    const float3* min_xyz,
					    const float* celldx, const float* celldy,
					    float4* xyzq, float4* xyzq_sorted,
					    int* col_patom, int* loc2glo_ind) {

  int ncoord = zone_patom[7];
  float4 *h_xyzq = new float4[ncoord];
  copy_DtoH<float4>(xyzq, h_xyzq, ncoord);
  float4 *h_xyzq_sorted = new float4[ncoord];
  copy_DtoH<float4>(xyzq_sorted, h_xyzq_sorted, ncoord);

  int *h_col_patom = new int[ncol_tot+1];
  copy_DtoH<int>(col_patom, h_col_patom, ncol_tot+1);
  int *h_loc2glo_ind = new int[ncoord];
  copy_DtoH<int>(loc2glo_ind, h_loc2glo_ind, ncoord);

  bool ok = true;

  int izone, i, j;
  float x, y, xj, yj;
  int ix, iy, ind, lo_ind, hi_ind;
  try {
    int ind0 = 0;
    for (izone=0;izone < 8;izone++) {
      int istart, iend;
      if (izone > 0) {
	istart = zone_patom[izone-1];
      } else {
	istart = 0;
      }
      iend = zone_patom[izone] - 1;
      if (iend >= istart) {
	float x0 = min_xyz[izone].x;
	float y0 = min_xyz[izone].y;
	for (i=istart;i <= iend;i++) {
	  x = h_xyzq_sorted[i].x;
	  y = h_xyzq_sorted[i].y;
	  ix = (int)((x - x0)/celldx[izone]);
	  iy = (int)((y - y0)/celldy[izone]);
	  ind = ind0 + ix + iy*ncellx[izone];
	  lo_ind = h_col_patom[ind];
	  hi_ind = h_col_patom[ind+1] - 1;
	  if (i < lo_ind || i > hi_ind) throw 1;
	}
	for (i=istart;i <= iend;i++) {
	  x = h_xyzq_sorted[i].x;
	  y = h_xyzq_sorted[i].y;
	  j = h_loc2glo_ind[i];
	  xj = h_xyzq[j].x;
	  yj = h_xyzq[j].y;
	  if (x != xj || y != yj) throw 2;
	}	
	ind0 += ncellx[izone]*ncelly[izone];
      }
    }
  }
  catch (int a) {
    std::cout << "test_z_columns FAILED at i=" << i << std::endl;
    if (a == 1) {
      std::cout << "ind, lo_ind, hi_ind = " << ind << " " << lo_ind << " " << hi_ind << std::endl;
    } else if (a == 2) {
      std::cout << "x,y   =" << x << " " << y << std::endl;
      std::cout << "xj,yj =" << xj << " " << yj << std::endl;
    }
    ok = false;
  }

  if (ok) std::cout << "test_z_columns OK" << std::endl;

  delete [] h_xyzq;
  delete [] h_xyzq_sorted;
  delete [] h_col_patom;
  delete [] h_loc2glo_ind;

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
				       const float* celldx, const float* celldy,
				       float4* xyzq, float4* xyzq_sorted,
				       int* col_patom, int* cell_patom,
				       int* loc2glo_ind) {

  int ncoord = zone_patom[7];
  float4 *h_xyzq = new float4[ncoord];
  copy_DtoH<float4>(xyzq, h_xyzq, ncoord);
  float4 *h_xyzq_sorted = new float4[ncoord];
  copy_DtoH<float4>(xyzq_sorted, h_xyzq_sorted, ncoord);
  int *h_col_patom = new int[ncol_tot+1];
  copy_DtoH<int>(col_patom, h_col_patom, ncol_tot+1);
  int *h_loc2glo_ind = new int[ncoord];
  copy_DtoH<int>(loc2glo_ind, h_loc2glo_ind, ncoord);
  int *h_cell_patom = new int[ncell_max];
  copy_DtoH<int>(cell_patom, h_cell_patom, ncell_max);

  bool ok = true;

  int izone, i, j, k, prev_ind;
  float x, y, z, prev_z;
  float xj, yj, zj;
  int ix, iy, ind, lo_ind, hi_ind;
  try {

    k = 0;
    for (i=1;i < ncol_tot+1;i++) {
      for (j=h_col_patom[i-1];j < h_col_patom[i];j+=32) {
	if (j != h_cell_patom[k]) throw 4;
	k++;
      }
    }

    int ind0 = 0;
    for (izone=0;izone < 8;izone++) {
      int istart, iend;
      if (izone > 0) {
	istart = zone_patom[izone-1];
      } else {
	istart = 0;
      }
      iend = zone_patom[izone] - 1;
      if (iend >= istart) {
	float x0 = min_xyz[izone].x;
	float y0 = min_xyz[izone].y;
	prev_z = min_xyz[izone].z;
	prev_ind = ind0;
	for (i=istart;i <= iend;i++) {
	  x = h_xyzq_sorted[i].x;
	  y = h_xyzq_sorted[i].y;
	  z = h_xyzq_sorted[i].z;
	  
	  ix = (int)((x - x0)/celldx[izone]);
	  iy = (int)((y - y0)/celldy[izone]);
	  ind = ind0 + ix + iy*ncellx[izone];

	  if (prev_ind != ind) {
	    prev_z = min_xyz[izone].z;
	  }

	  lo_ind = h_col_patom[ind];
	  hi_ind = h_col_patom[ind+1] - 1;
	  if (i < lo_ind || i > hi_ind) throw 1;
	  if (z < prev_z) throw 2;
	  prev_z = z;
	  prev_ind = ind;
	}

	for (i=istart;i <= iend;i++) {
	  x = h_xyzq_sorted[i].x;
	  y = h_xyzq_sorted[i].y;
	  z = h_xyzq_sorted[i].z;	  
	  j = h_loc2glo_ind[i];
	  xj = h_xyzq[j].x;
	  yj = h_xyzq[j].y;
	  zj = h_xyzq[j].z;
	  if (x != xj || y != yj || z != zj) throw 3;
	}	

	ind0 += ncellx[izone]*ncelly[izone];
      }
    }
  }
  catch (int a) {
    std::cout << "test_sort FAILED at i=" << i << std::endl;
    if (a == 1) {
      std::cout << "ind, lo_ind, hi_ind = " << ind << " " << lo_ind << " " << hi_ind << std::endl;
    } else if (a == 2) {
      std::cout << "prev_z, z = " << prev_z << " " << z << std::endl;
    } else if (a == 3) {
      std::cout << "x,y,z   =" << x << " " << y << " " << z << std::endl;
      std::cout << "xj,yj,zj=" << xj << " " << yj << " " << zj << std::endl;
    } else if (a == 4) {
      std::cout << "j,k=" << j << " " << k << "cell_patom[k]=" << h_cell_patom[k] << std::endl;
    }
    ok = false;
  }
  catch(...) {
    std::cout << "default catch" << std::endl;
  }

  if (ok) std::cout << "test_sort OK" << std::endl;

  delete [] h_xyzq;
  delete [] h_xyzq_sorted;
  delete [] h_col_patom;
  delete [] h_cell_patom;
  delete [] h_loc2glo_ind;

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
      int ncell_i = ncellx[izone]*ncelly[izone]*ncellz_max[izone];
      int ncell_j = ncellx[jzone]*ncelly[jzone]*ncellz_max[jzone];
      // Estimate the number of neighbors in each direction for the positive direction and multiply
      // by the number of cells
      int n_neigh_ij = ((int)ceilf(rcut/celldx[izone])+1)*((int)ceilf(rcut/celldy[izone])+1)
	*((int)ceilf(rcut/celldz_min[izone])+1)*ncell_j;
      int n_neigh_ji = ((int)ceilf(rcut/celldx[jzone])+1)*((int)ceilf(rcut/celldy[jzone])+1)
	*((int)ceilf(rcut/celldz_min[jzone])+1)*ncell_i;
      n_tile_est += max(n_neigh_ij, n_neigh_ji);
    }
  }

  // Assume every i-j tile is in a separate ientry (worst case)
  n_ientry_est = n_tile_est;
}

//
// Sorts atoms into tiles
//
template <int tilesize>
void NeighborList<tilesize>::sort(const int *zone_patom,
				  const float3 *max_xyz, const float3 *min_xyz,
				  float4 *xyzq,
				  float4 *xyzq_sorted,
				  cudaStream_t stream) {

  int ncoord = zone_patom[7];

  //
  // Calculate zone_natom
  //
  // zone_natom[izone] = number of atoms in zone "izone"
  for (int izone=0;izone < 8;izone++) {
    int nstart;
    if (izone > 0) {
      nstart = zone_patom[izone-1];
    } else {
      nstart = 0;
    }
    zone_natom[izone] = zone_patom[izone] - nstart;
  }

  set_int_zone(zone_natom, n_int_zone, int_zone);
  set_cell_sizes(zone_natom, max_xyz, min_xyz, ncellx, ncelly, ncellz_max,
		 celldx, celldy, celldz_min);

  // Setup nlist_param and copy it to GPU
  int ncol = 0;
  int max_ncellxy = 0;
  for (int izone=0;izone < 8;izone++) {
    h_nlist_param->zone_patom[izone] = zone_patom[izone];
    h_nlist_param->n_int_zone[izone] = n_int_zone[izone];
    for (int jzone=0;jzone < 8;jzone++) {
      h_nlist_param->int_zone[izone][jzone] = int_zone[izone][jzone];
    }
    h_nlist_param->ncol[izone] = ncol;
    h_nlist_param->ncellx[izone] = ncellx[izone];
    h_nlist_param->ncelly[izone] = ncelly[izone];
    max_ncellxy = max(max_ncellxy, ncellx[izone]*ncelly[izone]);
    ncol += ncellx[izone]*ncelly[izone];
    h_nlist_param->celldx[izone] = celldx[izone];
    h_nlist_param->celldy[izone] = celldy[izone];
    h_nlist_param->inv_celldx[izone] = 1.0f/celldx[izone];
    h_nlist_param->inv_celldy[izone] = 1.0f/celldy[izone];
    h_nlist_param->minxyz[izone].x = min_xyz[izone].x;
    h_nlist_param->minxyz[izone].y = min_xyz[izone].y;
    h_nlist_param->minxyz[izone].z = min_xyz[izone].z;
  }
  h_nlist_param->ncol[8] = ncol;

  set_nlist_param(stream);

  int ncol_tot = 0;
  ncell_max = 0;
  for (int izone=0;izone < 8;izone++) {
    ncol_tot += ncellx[izone]*ncelly[izone];
    ncell_max += ncellx[izone]*ncelly[izone]*ncellz_max[izone];
  }
  // NOTE: ncell_max is an approximate upper bound for the number of cells,
  //       it is possible to blow this bound, so we should check for it

  reallocate<int>(&col_natom, &col_natom_len, ncol_tot, 1.2f);
  reallocate<int>(&col_patom, &col_patom_len, ncol_tot+1, 1.2f);
  reallocate<int>(&atom_icol, &atom_icol_len, ncoord, 1.2f);
  reallocate<int>(&loc2glo_ind, &loc2glo_ind_len, ncoord, 1.2f);
  reallocate<int>(&cell_patom, &cell_patom_len, ncell_max, 1.2f);
  reallocate<int4>(&cell_xyz_zone, &cell_xyz_zone_len, ncell_max, 1.2f);
  reallocate<float>(&cell_bz, &cell_bz_len, ncell_max, 1.2f);
  reallocate<int>(&col_ncellz, &col_ncellz_len, ncol_tot, 1.2f);
  reallocate<int3>(&col_xy_zone, &col_xy_zone_len, ncol_tot, 1.2f);
  reallocate<int>(&col_cell, &col_cell_len, ncol_tot, 1.2f);

  clear_gpu_array<int>(col_natom, ncol_tot, stream);

  int nthread, nblock;

  //
  // Calculate number of atoms in each z-column (col_natom)
  // and the column index for each atom (atom_icol)
  //
  nthread = 512;
  nblock = (ncoord-1)/nthread+1;
  calc_z_column_index_kernel<<< nblock, nthread, 0, stream >>>
    (xyzq, col_natom, atom_icol, col_xy_zone);
  cudaCheck(cudaGetLastError());

  /*
  int ind0 = 0;
  for (int izone=0;izone < 8;izone++) {
    int istart, iend;
    if (izone > 0) {
      istart = zone_patom[izone-1];
    } else {
      istart = 0;
    }
    iend = zone_patom[izone] - 1;
    if (iend >= istart) {

      nthread = 512;
      nblock = (ncoord-1)/nthread+1;

      calc_z_column_index_kernel<<< nblock, nthread, 0, stream >>>
	(istart, iend, xyzq, ind0, izone, col_natom, atom_icol);
      cudaCheck(cudaGetLastError());

      ind0 += ncellx[izone]*ncelly[izone];
    }
  }
  */

  /*
  thrust::device_ptr<int> col_natom_ptr(col_natom);
  thrust::device_ptr<int> col_patom_ptr(col_patom);
  thrust::exclusive_scan(col_natom_ptr, col_natom_ptr + ncol_tot, col_patom_ptr);
  clear_gpu_array<int>(col_natom, ncol_tot, stream);
  */

  //
  // Calculate positions in z columns
  // NOTE: Clears col_natom and sets (col_patom, cell_patom, col_ncellz, d_nlist_param.ncell)
  //
  std::cout << "ncol_tot = " << ncol_tot << std::endl;
  nthread = min(((ncol_tot-1)/tilesize+1)*tilesize, get_max_nthread());
  std::cout << "nthread = " << nthread << std::endl;
  //int shmem_size = nthread*sizeof(int);
  //  calc_z_column_pos_kernel<<< 1, nthread, shmem_size, stream >>>
  //    (ncol_tot, col_natom, col_patom);
  int shmem_size = nthread*sizeof(int2);
  calc_z_column_pos_kernel<tilesize> <<< 1, nthread, shmem_size, stream >>>
    (ncol_tot, col_xy_zone, col_natom, col_patom, cell_patom, col_ncellz, cell_xyz_zone,
     col_cell);

  //
  // Calculate ncellz_max[izone]
  //
  nthread = ((max_ncellxy-1)/warpsize+1)*warpsize;
  nblock = 8;
  shmem_size = nthread*sizeof(int);
  calc_ncellz_max<<< nblock, nthread, shmem_size, stream >>>(col_ncellz);

  //
  // Reorder atoms into z-columns
  // NOTE: also sets up startcell_zone[izone]
  //
  nthread = 512;
  nblock = (ncoord-1)/nthread+1;
  reorder_atoms_z_column_kernel<<< nblock, nthread, 0, stream >>>
    (ncoord, atom_icol, col_natom, col_patom, xyzq, xyzq_sorted, loc2glo_ind);
  cudaCheck(cudaGetLastError());

  // Test z columns
  cudaCheck(cudaDeviceSynchronize());
  test_z_columns(zone_patom, ncellx, ncelly, ncol_tot, min_xyz, celldx, celldy, xyzq, xyzq_sorted,
		 col_patom, loc2glo_ind);

  // Now sort according to z coordinate
  nthread = 512; //11*tilesize;
  nblock = ncellx[0]*ncelly[0];
  if (nthread < get_max_nthread()) {
    shmem_size = nthread*sizeof(keyval_t);
    sort_z_column_kernel<<< nblock, nthread, shmem_size, stream >>>
      (col_patom, xyzq_sorted, loc2glo_ind);
    cudaCheck(cudaGetLastError());
  } else {
    std::cerr << "Neighborlist::sort, this version of sort_z_column_kernel not implemented yet"
	      << std::endl;
  }

  // Test sort
  cudaCheck(cudaDeviceSynchronize());
  test_sort(zone_patom, ncellx, ncelly, ncol_tot, ncell_max,
	    min_xyz, celldx, celldy, xyzq, xyzq_sorted,
	    col_patom, cell_patom, loc2glo_ind);

  //  reorder_atoms_kernel<<< nblock, nthread, 0, stream >>>
  //    (ncoord, tilex_val, xyzq, xyzq_sorted);
  //cudaCheck(cudaGetLastError());
}

//
// Calculates bounding box (bb) and cell z-boundaries (cell_bz)
// NOTE: Each thread calculates one bounding box
//
template <int tilesize>
__global__ void calc_bounding_box_kernel(const int* __restrict__ cell_patom,
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

    //int ix = (int)((minx - x0)*inv_dx);
    //int iy = (int)((miny - y0)*inv_dy);

    for (int i=istart+1;i < iend;i++) {
      /*
      if (i < 0 || i >= 23558) {
	printf("ERROR i = %d\n",i);
	return;
      }
      */
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
  }

}

//#######################################################################

//
// Class creator
//
template <int tilesize>
NeighborList<tilesize>::NeighborList(int nx, int ny, int nz) {
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

  tile_indj_sparse_len = NULL;
  tile_indj_sparse = NULL;

  // Neighbor list building
  col_natom_len = 0;
  col_natom = NULL;

  col_patom_len = 0;
  col_patom = NULL;

  atom_icol_len = 0;
  atom_icol = NULL;

  col_ncellz_len = 0;
  col_ncellz = NULL;

  col_xy_zone_len = 0;
  col_xy_zone = NULL;

  col_cell_len = 0;
  col_cell = NULL;

  loc2glo_ind_len = 0;
  loc2glo_ind = NULL;
  
  cell_patom_len = 0;
  cell_patom = NULL;

  cell_xyz_zone_len = 0;
  cell_xyz_zone = NULL;

  cell_bz_len = 0;
  cell_bz = NULL;

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
  if (loc2glo_ind != NULL) deallocate<int>(&loc2glo_ind);
  if (cell_patom != NULL) deallocate<int>(&cell_patom);
  if (col_ncellz != NULL) deallocate<int>(&col_ncellz);
  if (col_xy_zone != NULL) deallocate<int3>(&col_xy_zone);
  if (col_cell != NULL) deallocate<int>(&col_cell);
  if (cell_xyz_zone != NULL) deallocate<int4>(&cell_xyz_zone);
  if (cell_bz != NULL) deallocate<float>(&cell_bz);
  if (bb != NULL) deallocate<bb_t>(&bb);
  deallocate_host<NeighborListParam_t>(&h_nlist_param);
}

//
// Sets ientry from host memory array
//
template <int tilesize>
void NeighborList<tilesize>::set_ientry(int n_ientry, ientry_t *h_ientry, cudaStream_t stream) {

  this->n_ientry = n_ientry;

  // Allocate & reallocate d_ientry
  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.4f);

  // Copy to device
  copy_HtoD<ientry_t>(h_ientry, ientry, n_ientry, stream);
}

//----------------------------------------------------------------------------------------
//
// Builds neighborlist
//


//
// The entire warp enters here
// If IvsI = true, search within I zone
//
template <bool IvsI>
__device__
void get_cell_bounds_z(const int izone, const int jzone, const int icell, const int ncell,
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
      jcell_start_right = 0;         // start looking for cells at right from 1
      jcell0 = 0;                    // left boundary set to minimum value
      jcell1 = -1;                   // set to "no cells" value
      //      dist[1] = 0.0f;
    } else if (icell >= ncell) {
      // This is one of the image cells on the right =>
      // set the right cell boundary (icell1) to ncell and start looking for the left
      // boundary from ncell
      jcell_start_left = ncell-1;    // start looking for cells at left from ncell
      jcell_start_right = ncell;     // with this value, we don't look for cells on the right
      jcell0 = ncell;                // set to "no cells" value
      jcell1 = ncell-1;              // right boundary set to maximum value
      //      dist[ncell] = 0.0f;
    } else {
      jcell_start_left = icell - 1;
      jcell_start_right = icell + 1;
      jcell0 = icell;
      jcell1 = icell;
      //      dist[icell] = 0.0f;
    }
  } else {
    /*
    // Search between two different zones
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
    */
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
    //dist[j] = max(0.0f, d);
    jcell0 = j;
  }

  //
  // Check cells at right, stop once the distance to the cell left boundary
  // is greater than the cutoff.
  //
  // Cell left boundary is at bx[i-1]
  //
  for (int j=jcell_start_right;j < ncell;j++) {
    float bx_j = (j > 0) ? bx[j-1] : d_nlist_param.minxyz[jzone].z;
    float d = bx_j - x1;
    if (d > rcut) break;
    //dist[j] = max(0.0f, d);
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
void get_cell_bounds_xy(const int izone, const int jzone, const int icell,
			const int ncell,
			const float x0, const float x1,
			const float dx, const float rcut,
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
    float bx = d_nlist_param.minxyz[0].x + icell*dx;
    jcell0 = max(0, icell - (int)ceilf((rcut - (x0 - bx))/dx));

    //
    // Check cells at right, stop once the distance to the cell left boundary
    // is greater than the cutoff.
    //
    // Cell left boundary is at bx
    // portion inside i-cell is (bx-x1)
    // => what is left of rcut on the right of i-cell is rcut-(bx-x1)
    //
    bx = d_nlist_param.minxyz[0].x + (icell+1)*dx;
    jcell1 = min(ncell-1, icell + (int)ceilf((rcut - (bx - x1))/dx));

    //
    // Take care of the boundaries:
    //
    if (icell < 0) jcell0 = 0;
    if (icell >= ncell) jcell1 = ncell - 1;

    /*
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
    */

  } else {
    /*
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
    */
  }


  // Cell bounds are jcell0:jcell1
      
}

//
// Finds minimum of z0 and maximum of z1 across warp using __shfl -command
//
__forceinline__ __device__ void minmax_shfl(int z0, int z1, int &z0_min, int &z1_max) {
#if __CUDA_ARCH__ >= 300
  for (int i=16;i >= 1;i/=2) {
    z0_min = min(z0_min, __shfl_xor(z0, i));
    z1_max = max(z1_max, __shfl_xor(z1, i));
  }
#endif
}

//
// Broadcasts value from lane 0 to all lanes
//
__forceinline__ __device__ int bcast_shfl(int val) {
#if __CUDA_ARCH__ >= 300
  return __shfl(val, 0);
#else
  return 0;
#endif
}

//
// Calculates exclusive plus-scan across warp for binary (0 or 1) values
//
// wid = warp ID = threadIdx.x % warpsize
//
__forceinline__ __device__ int binary_scan(int val, int wid) {
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

/*
//
// Builds top_excl_pos[] and top_excl[] for local coordinate indexing
//
__global__ void build_local_top_excl_kernel(const int ncoord,
					    const int* __restrict__ loc2glo_ind,
					    const int* __restrict__ glo_top_excl_pos,
					    const int* __restrict__ glo_top_excl,
					    int* __restrict__ top_excl_pos,
					    int* __restrict__ top_excl) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  
  if (i < ncoord) {
    int j = loc2glo_ind[i];
    int jstart = glo_top_excl_pos[j];
    int jend   = glo_top_excl_pos[j+1] - 1;
    for (int j=jstart;j <= jend;j++) {
      int k = glo_top_excl[j];
    }
  }
}

//
// Apply topological exclusions to the neighborlist
//
__global__ void apply_top_excl_kernel(const int n_top_excl_pair,
				      const int2* __restrict__ top_excl_pair,
				      const int* __restrict__ atom_cell) {
  const int itop = threadIdx.x + blockIdx.x*blockDim.x;
  
  if (itop < n_top_excl_pair) {
    int2 val = top_excl_pair[itop];
    int i = val.x;
    int j = val.y;
    int icell = atom_cell[i];
    int jcell = atom_cell[j];
    //atomicOr(&excl[iexcl], top_mask);
  }
}
*/

/*
//
// Returns topological exclusion mask
//
// NOTE: top_excl_pos[] and top_excl[] are in local coordinates
//
template <int tilesize>
__device__ int get_top_excl_mask(const int wid, const int icell, const int jcell,
				 const int* __restrict__ cell_patom,
				 const int* __restrict__ loc2glo_ind,
				 const int* __restrict__ top_excl_pos) {

  int istart = cell_patom[icell] - 1;
  int iend   = cell_patom[icell+1] - 2;

  int jstart = cell_patom[jcell] - 1;
  int jend   = cell_patom[jcell+1] - 2;

  int i = loc2glo_ind[istart + wid];
  int excl_start = top_excl_pos[i];
  int excl_end   = top_excl_pos[i+1] - 1;

  for (int excl_i = excl_start;excl_i <= excl_end;excl_i++) {
    int j = top_excl[excl_i];
  }

  return excl;
}
*/

//
// Flush jlist into global memory
//
template <int tilesize>
__device__ void flush_jlist(const int wid, const int icell, const int n_jlist,
			    volatile int* __restrict__ sh_jlist, const int ish,
			    const float boxx, const float boxy, const float boxz,
			    const float rcut2, const float4* __restrict__ xyzq,
			    const int* __restrict__ cell_patom,
			    volatile float3* __restrict__ sh_xyzi,
			    int* __restrict__ tile_indj,
			    tile_excl_t<tilesize>* __restrict__ tile_excl,
			    ientry_t* __restrict__ ientry) {

  // Allocate space on the global tile_excl and tile_indj -lists
  // NOTE: we are allocating space for n_jlist entries. However, not all of these are used
  //       because some of the i-j tiles will be empty. If we don't want to keep these
  //       "ghost" tiles in the list, we need to setup another shared memory buffer for
  //       exclusion masks and then only add the tiles that are non-empty.
  int jtile_start;
  if (wid == 0) {
    jtile_start = atomicAdd(&d_nlist_param.n_tile, n_jlist);
  }
#if __CUDA_ARCH__ < 300
  printf("flush_jlist, bcast_shfl not implemented for __CUDA_ARCH__ < 300\n");
#else
  jtile_start = bcast_shfl(jtile_start);
#endif

  // Calculate exclusion mask (=check for distance exclusions)
  int n_jlist_new = 0;
  int istart = cell_patom[icell] - 1;
  int iend   = cell_patom[icell+1] - 2;
  for (int i=0;i < n_jlist;i++) {
    int jcell = sh_jlist[i];
    int jstart = cell_patom[jcell] - 1;
    int jend   = cell_patom[jcell+1] - 2;
    int excl = get_dist_excl_mask<tilesize>(wid, istart, iend, jstart, jend, ish, 
					    boxx, boxy, boxz, rcut2, xyzq, sh_xyzi);
    // We keep this j-cell in the list if any of the exclusion masks are
    // not all-exclusive
    // NOTE: writing out exclusion mask here voids the need for storing these in
    //       shared memory
    if (__any(excl != -1)) {
      // Keep in list
      int jtile = jtile_start + n_jlist_new;
      tile_excl[jtile].excl[wid] = excl;
      if (wid == 0) tile_indj[jtile] = jstart;
      n_jlist_new++;
    }
  }
  
  // Add to ientry list
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

const int n_jlist_max = 100;

//
// Build neighborlist for one zone at the time
//
// NOTE: One warp takes care of one cell
//
template < int tilesize, bool IvsI >
__global__
void build_kernel(const int4* __restrict__ cell_xyz_zone,
		  const int* __restrict__ col_ncellz,
		  const int* __restrict__ col_cell,
		  const float* __restrict__ cell_bz,
		  const int* __restrict__ cell_patom,
		  const float4* __restrict__ xyzq,
		  const float boxx, const float boxy, const float boxz,
		  const float rcut, const float rcut2,
		  const bb_t* __restrict__ bb,
		  int* __restrict__ tile_indj,
		  tile_excl_t<tilesize>* __restrict__ tile_excl,
		  ientry_t* __restrict__ ientry) {

  // Shared memory
  extern __shared__ char shbuf[];

  // Index of the i-cell
  const int icell = (threadIdx.x + blockIdx.x*blockDim.x)/warpsize;

  if (icell >= d_nlist_param.ncell) return;

  // Warp index
  const int wid = threadIdx.x % warpsize;

  // Get (icellx, icelly, icellz, izone):
  int4 icell_xyz_zone = cell_xyz_zone[icell];
  int icellx = icell_xyz_zone.x;
  int icelly = icell_xyz_zone.y;
  int icellz = icell_xyz_zone.z;
  int izone  = IvsI ? 0 : icell_xyz_zone.w;

  int n_jzone = IvsI ? 1 : d_nlist_param.n_int_zone[izone];
  
  if (n_jzone == 0) return;

  // Load bounding box
  bb_t ibb = bb[icell];

  // ----------------------------------------------------------------
  // Calculate shared memory pointers:
  //
  // Total memory requirement:
  // (blockDim.x/warpsize)*( (~IvsI)*n_jzone*sizeof(int4) + n_jlist_max*sizeof(int) 
  //                         + tilesize*sizeof(float3))
  //
  // Required space:
  // jcellxy: (blockDim.x/warpsize)*n_jzone*sizeof(int4)
  // NOTE: Each warp has its own jcellxy[]
  volatile int4 *sh_jcellxy = (int4 *)&shbuf[(threadIdx.x/warpsize)*n_jzone*sizeof(int4)];
  int shbuf_pos;
  if (IvsI) {
    shbuf_pos = 0;
  } else {
    shbuf_pos = (blockDim.x/warpsize)*n_jzone*sizeof(int4);
  }

  // Temporary j-cell list. Each warp has its own jlist
  // sh_jlist: (blockDim.x/warpsize)*n_jlist_max*sizeof(int)
  volatile int *sh_jlist = (int *)&shbuf[shbuf_pos + (threadIdx.x/warpsize)*n_jlist_max*sizeof(int)];
  shbuf_pos += (blockDim.x/warpsize)*n_jlist_max*sizeof(int);

  // i-cell coordinates (x, y, z)
  // sh_xyzi: (blockDim.x/warpsize)*tilesize*sizeof(float3)
  volatile float3* sh_xyzi = (float3 *)&shbuf[shbuf_pos + 
					      (threadIdx.x/warpsize)*tilesize*sizeof(float3)];
  // ----------------------------------------------------------------

  for (int imx=d_nlist_param.imx_lo;imx <= d_nlist_param.imx_hi;imx++) {
    float imbbx0 = ibb.x + imx*boxx;
    int n_jcellx = 0;
    int jcellx_min, jcellx_max;
    if (IvsI) {
      get_cell_bounds_xy<IvsI>(0, 0, icellx + imx*d_nlist_param.ncellx[0],
			       d_nlist_param.ncellx[0], imbbx0-ibb.wx, imbbx0+ibb.wx,
			       d_nlist_param.celldx[0], rcut, jcellx_min, jcellx_max);
      n_jcellx = max(0, jcellx_max - jcellx_min + 1);
      if (n_jcellx == 0) continue;
    } else {
      if (wid < n_jzone) {
	int jzone = d_nlist_param.int_zone[izone][wid];
	int jcellx0_t, jcellx1_t;
	get_cell_bounds_xy<IvsI>(izone, jzone, icellx + imx*d_nlist_param.ncellx[izone],
				 d_nlist_param.ncellx[jzone], imbbx0-ibb.wx, imbbx0+ibb.wx,
				 d_nlist_param.celldx[jzone], rcut, jcellx0_t, jcellx1_t);
	n_jcellx = max(0, jcellx1_t-jcellx0_t+1);
	sh_jcellxy[wid].x = jcellx0_t;
	sh_jcellxy[wid].y = jcellx1_t;
      }
      if (__all(n_jcellx == 0)) continue;
    }
    
    for (int imy=d_nlist_param.imy_lo;imy <= d_nlist_param.imy_hi;imy++) {
      float imbby0 = ibb.y + imy*boxy;
      int n_jcelly = 0;
      int jcelly_min, jcelly_max;
      if (IvsI) {
	get_cell_bounds_xy<IvsI>(0, 0, icelly + imy*d_nlist_param.ncelly[0],
				 d_nlist_param.ncelly[0], imbby0-ibb.wy, imbby0+ibb.wy,
				 d_nlist_param.celldy[0], rcut, jcelly_min, jcelly_max);
	n_jcelly = max(0, jcelly_max - jcelly_min + 1);
	if (n_jcelly == 0) continue;
      } else {
	if (wid < n_jzone) {
	  int jzone = IvsI ? 0 : d_nlist_param.int_zone[izone][wid];
	  int jcelly0_t, jcelly1_t;
	  get_cell_bounds_xy<IvsI>(izone, jzone, icelly + imy*d_nlist_param.ncelly[izone],
				   d_nlist_param.ncelly[jzone], imbby0-ibb.wy, imbby0+ibb.wy,
				   d_nlist_param.celldy[jzone], rcut, jcelly0_t, jcelly1_t);
	  n_jcelly = max(0, jcelly1_t-jcelly0_t+1);
	  sh_jcellxy[wid].z = jcelly0_t;
	  sh_jcellxy[wid].w = jcelly1_t;
	}
	if (__all(n_jcelly == 0)) continue;
      }

      for (int imz=d_nlist_param.imz_lo;imz <= d_nlist_param.imz_hi;imz++) {
	float imbbz0 = ibb.z + imz*boxz;
	int ish = imx+1 + 3*(imy+1 + 3*(imz+1));

	int n_jlist = 0;

	if (IvsI) {
	  int total_xy = n_jcellx*n_jcelly;
	  /*
	  if (threadIdx.x == 0 && icell == 0 && imz == 0) {
	    printf("%d %d %d icell = %d total_xy = %d\n",imx,imy,imz,icell,total_xy);
	  }
	  */
	  int jcellz_min=1000000, jcellz_max=0;
	  for (int ibase=0;ibase < total_xy;ibase+=warpsize) {
	    int i = ibase + wid;
	    int jcellz0_t=1000000, jcellz1_t=0;
	    if (i < total_xy) {
	      int jcelly = i/n_jcellx;
	      int jcellx = i - jcelly*n_jcellx;
	      jcellx += jcellx_min;
	      jcelly += jcelly_min;
	      int jcol = jcellx + jcelly*d_nlist_param.ncellx[0];
	      int cell0 = col_cell[jcol];
	      get_cell_bounds_z<IvsI>(0, 0, icellz + imz*col_ncellz[jcol],
				      col_ncellz[jcol], imbbz0-ibb.wz, imbbz0+ibb.wz,
				      &cell_bz[cell0], rcut, jcellz0_t, jcellz1_t);
	      /*
	      if (icell == 0 && imx == 0 && imy == 0 && imz == 0) {
		printf("jcell %d %d %d %d, %f %f cell_bz = %f %f %f\n",
		       jcellx,jcelly,jcellz0_t,jcellz1_t,imbbz0-ibb.wz, imbbz0+ibb.wz,
		       d_nlist_param.minxyz[0].z,
		       cell_bz[cell0],cell_bz[cell0+1]);
	      }
	      */
	    }
#if __CUDA_ARCH__ < 300
	    printf("build_kernel: minmax_shfl not implemented for __CUDA_ARCH__ < 300\n");
#else
	    minmax_shfl(jcellz0_t, jcellz1_t, jcellz_min, jcellz_max);
#endif
	  }

	  int n_jcellz = jcellz_max - jcellz_min + 1;
	  int total_xyz = n_jcellx*n_jcelly*n_jcellz;
	  /*
	  if (threadIdx.x == 0 && icell == 0 && imz == 0) {
	    printf("%d %d %d icell = %d total_xyz = %d jcellz = %d %d\n",imx,imy,imz,icell,
		   total_xyz,jcellz_min,jcellz_max);
	  }
	  */
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
	      int jcelly = it/(n_jcellx*n_jcellz);
	      it -= jcelly*(n_jcellx*n_jcellz);
	      int jcellx = it/n_jcellz;
	      int jcellz = it - jcellx*n_jcellz;
	      jcellx += jcellx_min;
	      jcelly += jcelly_min;
	      jcellz += jcellz_min;
	      // Calculate column index "jcol" and final cell index "jcell"
	      int jcol = jcellx + jcelly*d_nlist_param.ncellx[0];
	      jcell = col_cell[jcol] + jcellz;
	      if (icell <= jcell) {
		// Read bounding box for j-cell
		bb_t jbb = bb[jcell];
		// Calculate distance between i-cell and j-cell bounding boxes
		float dx = max(0.0f, fabsf(imbbx0 - jbb.x) - ibb.wx - jbb.wx);
		float dy = max(0.0f, fabsf(imbby0 - jbb.y) - ibb.wy - jbb.wy);
		float dz = max(0.0f, fabsf(imbbz0 - jbb.z) - ibb.wz - jbb.wz);
		float r2 = dx*dx + dy*dy + dz*dz;
		if (r2 < rcut2) {
		  /*
		  if (icell == 0 && imx == 0 && imy == 0 && imz == 0) {
		    printf("jcell = %d, jcellxyz = %d %d %d r2 = %f %f %f %f\n",
			   jcell,jcellx,jcelly,jcellz,r2,dx,dy,dz);
		  }
		  */
		  ok = 1;
		}
	      }
	    } // if (i < total_xyz)
	    //
	    // Add j-cells into temporary list (in shared memory)
	    //
	    // First reduce to calculate position for each thread in warp
	    int pos = binary_scan(ok, wid);
	    int n_jlist_add = binary_reduce(ok);

	    /*
	    //
	    // Compute flags:
	    // buffer_full    = buffer will overflow after adding current j-cells
	    // last_iteration = we are on the last iteration of the loop
	    //
	    // NOTE: This rather complicated programming structure allows us to have only
	    //       one call to flush_jlist
	    bool buffer_full = ((n_jlist + n_jlist_add) > n_jlist_max);
	    bool last_iteration = (ibase + warpsize >= total_xyz);

	    //
	    // There are two possibilities why we have to flush:
	    // 1) we are on the last iteration
	    // 2) buffer is full
	    //
	    if (last_iteration || buffer_full) {
	      // Buffer is not full => add items to it
	      if (!buffer_full && ok) sh_jlist[n_jlist + pos] = jcell;
	      if (!buffer_full) n_jlist += n_jlist_add;
	      // Flush jlist
	      flush_jlist();
	      // Restart jlist
	      n_jlist = 0;
	      // If buffer was full, we need to add the items to it now
	      if (buffer_full && ok) sh_jlist[pos] = jcell;
	      if (buffer_full) n_jlist = n_jlist_add;
	    }
	    */

	    // Flush if the buffer is full
	    if ((n_jlist + n_jlist_add) > n_jlist_max) {
	      flush_jlist<tilesize>(wid, icell, n_jlist, sh_jlist, ish,
				    boxx, boxy, boxz, rcut2, xyzq, cell_patom,
				    sh_xyzi, tile_indj, tile_excl, ientry);
	      n_jlist = 0;
	    }
	    n_jlist += n_jlist_add;

	  }

	  if (n_jlist > 0) flush_jlist<tilesize>(wid, icell, n_jlist, sh_jlist, ish,
						 boxx, boxy, boxz, rcut2, xyzq, cell_patom,
						 sh_xyzi, tile_indj, tile_excl, ientry);



	} else {
	  int n_jcellx_tot = n_jcellx;
	  int n_jcelly_tot = n_jcelly;
#if __CUDA_ARCH__ < 300
	  printf("build_kernel: this part not implemented (2)\n");
#else
	  for (int i=16;i >= 1;i /= 2) {
	    n_jcellx_tot += __shfl_xor(n_jcellx_tot, i);
	    n_jcelly_tot += __shfl_xor(n_jcelly_tot, i);
	  }
#endif
	  // Total amount of work
	  int total = n_jcellx_tot*n_jcelly_tot*n_jzone;
	}


      } // for (int imz=imz_lo;imz <= imz_hi;imz++)
    } // for (int imy=imy_lo;imy <= imy_hi;imy++)
  } // for (int imx=imx_lo;imx <= imx_hi;imx++)

}

//
// Builds neighborlist
//
template <int tilesize>
void NeighborList<tilesize>::build(const float boxx, const float boxy, const float boxz,
				   const float rcut,
				   const float4 *xyzq,
				   cudaStream_t stream) {

  get_nlist_param();
  std::cout << "ncell = " << h_nlist_param->ncell << " ncell_max = " << ncell_max << std::endl;
  reallocate<bb_t>(&bb, &bb_len, ncell_max, 1.2f);

  int n_tile_est, n_ientry_est;
  get_tile_ientry_est(n_int_zone, int_zone, ncellx, ncelly, ncellz_max,
		      celldx, celldy, celldz_min, rcut,
		      n_tile_est, n_ientry_est);
  std::cout << "n_ientry_est = " << n_ientry_est << " n_tile_est = " << n_tile_est << std::endl;

  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry_est, 1.0f);
  reallocate<tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, n_tile_est, 1.0f);
  reallocate<int>(&tile_indj, &tile_indj_len, n_tile_est, 1.0f);


  int nthread = 512;
  int nblock = (ncell_max-1)/nthread + 1;

  calc_bounding_box_kernel<tilesize> <<< nblock, nthread, 0, stream >>>
    (cell_patom, xyzq, bb, cell_bz);
  cudaCheck(cudaGetLastError());

  // Shared memory requirements:
  // (blockDim.x/warpsize)*( (~IvsI)*n_jzone*sizeof(int4) + n_jlist_max*sizeof(int) 
  //                         + tilesize*sizeof(float3))
  int shmem_size = (nthread/warpsize)*( n_jlist_max*sizeof(int) + tilesize*sizeof(float3));
  // For !IvsI, shmem_size += (nthread/warpsize)*n_int_zone_max*sizeof(int4)
  std::cout << "NeighborList::build, shmem_size = " << shmem_size << std::endl;

  nthread = 512;
  nblock = (ncell_max-1)/(nthread/warpsize) + 1;
  build_kernel<tilesize, true>
    <<< nblock, nthread, shmem_size, stream >>>
    (cell_xyz_zone, col_ncellz, col_cell, cell_bz, cell_patom, xyzq,
     boxx, boxy, boxz, rcut, rcut*rcut, bb, tile_indj, tile_excl, ientry);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDeviceSynchronize());
  get_nlist_param();
  std::cout << "n_ientry = " << h_nlist_param->n_ientry
	    << " n_tile = " << h_nlist_param->n_tile << std::endl;

}

//
// Test neighbor list building with a simple N^2 algorithm
//
template <int tilesize>
void NeighborList<tilesize>::test_build(const int *zone_patom,
					const float boxx, const float boxy, const float boxz,
					const float rcut, const float4 *xyzq) {

  cudaCheck(cudaDeviceSynchronize());
  get_nlist_param();

  int n_ientry = h_nlist_param->n_ientry;
  int n_tile = h_nlist_param->n_tile;

  int ncoord = zone_patom[7];

  float4* h_xyzq = new float4[ncoord];
  copy_DtoH<float4>(xyzq, h_xyzq, ncoord);

  float rcut2 = rcut*rcut;

  float hboxx = 0.5f*boxx;
  float hboxy = 0.5f*boxy;
  float hboxz = 0.5f*boxz;

  int npair_cpu = 0;
  for (int izone=0;izone < 8;izone++) {
    for (int jzone=0;jzone < 8;jzone++) {
      //if (izone == 0 && izone == jzone && icell > jcell) cycle;
      if (izone == 1 && jzone != 5) continue;
      if (izone == 2 && jzone != 1 && jzone != 6) continue;
      if (izone == 4 && jzone != 1 && jzone != 2 && jzone != 3) continue;

      int istart = (izone > 0) ? istart = zone_patom[izone-1] : 0;
      int jstart = (jzone > 0) ? jstart = zone_patom[jzone-1] : 0;
      int iend = zone_patom[izone];
      int jend = zone_patom[jzone];

      for (int i=istart;i < iend;i++) {
	float xi = h_xyzq[i].x;
	float yi = h_xyzq[i].y;
	float zi = h_xyzq[i].z;
	int npair_i = 0;
	if (izone == 0 && jzone == 0) jstart = i + 1;
	for (int j=jstart;j < jend;j++) {
	  float xj = h_xyzq[j].x;
	  float yj = h_xyzq[j].y;
	  float zj = h_xyzq[j].z;
	  float dx = xi - xj;
	  float dy = yi - yj;
	  float dz = zi - zj;
	  int imx=0, imy=0, imz=0;
	  if (dx > hboxx) {
	    imx = -1;
	  } else if (dx < -hboxx) {
	    imx = 1;
	  }
	  if (dy > hboxy) {
	    imy = -1;
	  } else if (dy < -hboxy) {
	    imy = 1;
	  }
	  if (dz > hboxz) {
	    imz = -1;
	  } else if (dz < -hboxz) {
	    imz = 1;
	  }
	  dx += imx*boxx;
	  dy += imy*boxy;
	  dz += imz*boxz;
	  float r2 = dx*dx + dy*dy + dz*dz;
	  if (r2 < rcut2) {
	    npair_i++;
	    npair_cpu++;
	  }
	}
	//
      }
    }
  }

  ientry_t* h_ientry = new ientry_t[n_ientry];
  tile_excl_t<tilesize>* h_tile_excl = new tile_excl_t<tilesize>[n_tile];
  int* h_tile_indj = new int[n_tile];
  copy_DtoH<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH<tile_excl_t<tilesize> >(tile_excl, h_tile_excl, n_tile);
  copy_DtoH<int>(tile_indj, h_tile_indj, n_tile);

  int npair_gpu = 0;
  int n = 0;
  for (int ind=0;ind < n_ientry;ind++) {
    int istart = h_ientry[ind].indi;
    int ish    = h_ientry[ind].ish;
    int startj = h_ientry[ind].startj;
    int endj   = h_ientry[ind].endj;
    n += endj - startj + 1;
    for (int jtile=startj;jtile <= endj;jtile++) {
      for (int wid=0;wid < (num_excl<tilesize>::val);wid++) {
	npair_gpu += 32 - BitCount_ref(h_tile_excl[jtile].excl[wid]);
      }
    }
  }

  delete [] h_xyzq;
  delete [] h_ientry;
  delete [] h_tile_excl;
  delete [] h_tile_indj;

  if (npair_cpu != npair_gpu) {
    std::cout << "##################################################" << std::endl;
    std::cout << "test_build FAILED" << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "n_ientry = " << n_ientry << " n_tile = " << n_tile << std::endl;
    std::cout << "npair_cpu = " << npair_cpu << " npair_gpu = " << npair_gpu << std::endl;
    std::cout << "##################################################" << std::endl;
  } else {
    std::cout << "test_build OK" << std::endl;
  }
}

//
// Setups topological exclusions from data structure used in CHARMM
//
template <int tilesize>
void NeighborList<tilesize>::setup_top_excl(int ncoord, int *iblo14, int *inb14) {

  int *nexcl = new int[ncoord];

  // Count the number of exclusions to nexcl[0 ... ncoord-1]
  for (int i=0;i < ncoord;i++) nexcl[i] = 0;

  for (int i=0;i < ncoord;i++) {
    int excl_start;
    if (i > 0) {
      excl_start = iblo14[i-1];
    } else {
      excl_start = 0;
    }
    int excl_end = iblo14[i] - 1;
    nexcl[i] += excl_end - excl_start + 1;
    for (int excl_i=excl_start; excl_i <= excl_end;excl_i++) {
      int j = abs(inb14[excl_i]) - 1;
      // add i-j exclusion to atom j
      nexcl[j]++;
    }
  }

  

  delete [] nexcl;
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
  reallocate<int>(&tile_indj, &tile_indj_len, n_ijlist, 1.2f);
  reallocate<tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, n_ijlist, 1.2f);

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

static int BitCount_ref(unsigned int u) {
  unsigned int x = u;
  int res = 0;
  while (x != 0) {
    res += (x & 1);
    x >>= 1;
  }
  return res;
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

  copy_DtoH<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH<int>(tile_indj, h_tile_indj, n_tile);
  copy_DtoH< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, n_tile);

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

  copy_HtoD<ientry_t>(h_ientry_dense, ientry, n_ientry);
  copy_HtoD<int>(h_tile_indj_dense, tile_indj, n_tile);
  copy_HtoD< tile_excl_t<tilesize> >(h_tile_excl_dense, tile_excl, n_tile);

  allocate<ientry_t>(&ientry_sparse, n_ientry_sparse);
  allocate<int>(&tile_indj_sparse, n_tile_sparse);
  allocate< pairs_t<tilesize> >(&pairs, n_tile_sparse);
  ientry_sparse_len = n_ientry_sparse;
  tile_indj_sparse_len = n_tile_sparse;
  pairs_len = n_tile_sparse;

  copy_HtoD<ientry_t>(h_ientry_sparse, ientry_sparse, n_ientry_sparse);
  copy_HtoD<int>(h_tile_indj_sparse, tile_indj_sparse, n_tile_sparse);
  copy_HtoD< pairs_t<tilesize> >(h_pairs, pairs, n_tile_sparse);

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

  copy_DtoH<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH<int>(tile_indj, h_tile_indj, n_tile);
  copy_DtoH< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, n_tile);

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

  copy_HtoD<ientry_t>(h_ientry_noempty, ientry, n_ientry);
  copy_HtoD<int>(h_tile_indj_noempty, tile_indj, n_tile);
  copy_HtoD< tile_excl_t<tilesize> >(h_tile_excl_noempty, tile_excl, n_tile);

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

  copy_DtoH<ientry_t>(ientry, h_ientry, n_ientry);
  copy_DtoH<int>(tile_indj, h_tile_indj, n_tile);
  copy_DtoH< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, n_tile);

  std::cout << "Number of i-tiles = " << n_ientry << ", total number of tiles = " 
	    << n_tile << std::endl;

  std::ofstream file_npair("npair.txt", std::ofstream::out);
  std::ofstream file_nj("nj.txt", std::ofstream::out);

  unsigned int nexcl_bit = 0;
  unsigned int nexcl_bit_self = 0;
  unsigned int nempty_tile = 0;
  unsigned int nempty_line = 0;
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
    }
  }

  file_npair.close();
  file_nj.close();

  unsigned int n_tile_pairs = n_tile*tilesize*tilesize;
  std::cout << "Total number of pairs = " << n_tile_pairs << std::endl;
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

  reallocate<ientry_t>(&ientry, &ientry_len, n_ientry, 1.2f);
  reallocate<int>(&tile_indj, &tile_indj_len, n_tile, 1.2f);
  reallocate< tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, n_tile, 1.2f);

  copy_HtoD<ientry_t>(h_ientry, ientry, n_ientry);
  copy_HtoD<int>(h_tile_indj, tile_indj, n_tile);
  copy_HtoD< tile_excl_t<tilesize> >(h_tile_excl, tile_excl, n_tile);

  delete [] h_ientry;
  delete [] h_tile_indj;
  delete [] h_tile_excl;
}

//
// Explicit instances of DirectForce
//
template class NeighborList<16>;
template class NeighborList<32>;
