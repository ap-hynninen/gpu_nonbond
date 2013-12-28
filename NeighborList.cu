#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "NeighborList.h"

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
}

//
// Sets ientry from host memory array
//
template <int tilesize>
void NeighborList<tilesize>::set_ientry(int ni, ientry_t *h_ientry) {

  this->ni = ni;

  // Allocate & reallocate d_ientry
  reallocate<ientry_t>(&ientry, &ientry_len, ni, 1.4f);

  // Copy to device
  copy_HtoD<ientry_t>(h_ientry, ientry, ni, get_direct_nonbond_stream());
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
__global__ void build_excl_kernel(const int n_ijlist, const int3 *ijlist,
				  const int *cell_start, const float4 *xyzq,
				  int *tile_indj,
				  tile_excl_t<tilesize> *tile_excl,
				  const float boxx, const float boxy, const float boxz,
				  const float roff2) {
  const int num_thread_per_excl = (32/(num_excl<tilesize>::val));

  // Global thread index
  const unsigned int gtid = threadIdx.x + blockDim.x*blockIdx.x;
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
					const float4 *xyzq) {

  // Allocate & re-allocate (d_tile_indj, d_tile_excl)
  reallocate<int>(&tile_indj, &tile_indj_len, n_ijlist, 1.2f);
  reallocate<tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, n_ijlist, 1.2f);

  float roff2 = roff*roff;

  int nthread = nwarp_build_excl_dist*warpsize;
  int nblock = (n_ijlist-1)/(nthread/warpsize) + 1;
  size_t shmem_size = nwarp_build_excl_dist*tilesize*sizeof(float3); 

  if (tilesize == 16) {
    shmem_size += nwarp_build_excl_dist*(num_excl<tilesize>::val)*sizeof(unsigned int);
  }

  build_excl_kernel<tilesize>
    <<< nblock, nthread, shmem_size, get_direct_nonbond_stream() >>>
    (n_ijlist, ijlist, cell_start,
     xyzq, tile_indj, tile_excl,
     boxx, boxy, boxz,
     roff2);

  /*
  if (mdsim.q_test != 0) {
    test_excl_dist_index(mdsim.n_ijlist, mdsim.ijlist, mdsim.cell_start,
			 mdsim.xyzq.xyzq, mdsim.tile_indj, mdsim.tile_excl,
			 boxx, boxy, boxz,
			 roff2);
  }
  */

  cudaCheck(cudaGetLastError());
}

//----------------------------------------------------------------------------------------
//
// Combines tile_excl_final on GPU
// One thread takes care of one integer in the exclusion mask, therefore:
//
// 32x32 tile, 32 integers per tile
// 16x16 tile, 8 integers per tile
//
template <int tilesize>
__global__ void add_tile_final_kernel(const int ntile_final,
				      const int *tile_ind_final,
				      const tile_excl_t<tilesize> *tile_excl_final,
				      tile_excl_t<tilesize> *tile_excl) {
  // Global thread index
  const unsigned int gtid = threadIdx.x + blockDim.x*blockIdx.x;
  // Index to tile_ind_final[]
  const unsigned int i = gtid / (num_excl<tilesize>::val);
  // Index to exclusion mask
  const unsigned int ix = gtid % (num_excl<tilesize>::val);

  if (i < ntile_final) {
    int ind = tile_ind_final[i];
    tile_excl[ind].excl[ix] |= tile_excl_final[i].excl[ix];
  }

}

//
// Host wrapper for add_tile_final_kernel
//
template <int tilesize>
void NeighborList<tilesize>::add_tile_final(const int ntile_final, const int *tile_ind_final,
					    const tile_excl_t<tilesize> *tile_excl_final) {
  int nthread = 256;
  int nblock = (ntile_final*(num_excl<tilesize>::val) - 1)/nthread + 1;
  
  add_tile_final_kernel<tilesize>
    <<< nblock, nthread, 0, get_direct_nonbond_stream() >>>
    (ntile_final, tile_ind_final, tile_excl_final, tile_excl);
  
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
