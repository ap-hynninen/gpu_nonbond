#include "CudaPMEForcefield.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

__global__ void heuristic_check_kernel(const int ncoord, const int stride,
				       const double* __restrict__ coord,
				       const double* __restrict__ ref_coord,
				       const float rsq_limit,
				       int* global_flag) {
  extern __shared__ int sh_flag[];
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int stride2 = stride*2;
  const int sh_flag_size = blockDim.x/warpsize;

  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
  if (tid < ncoord) {
    dx = (float)(coord[tid]         - ref_coord[tid]);
    dy = (float)(coord[tid+stride]  - ref_coord[tid+stride]);
    dz = (float)(coord[tid+stride2] - ref_coord[tid+stride2]);
  }

  float rsq = dx*dx + dy*dy + dz*dz;
  // flag = 1 update is needed
  //      = 0 no update needed
  int flag = (rsq > rsq_limit);
  // Reduce flag, packed into bits.
  // NOTE: this assumes that warpsize <= 32
  sh_flag[threadIdx.x/warpsize] = (flag << (threadIdx.x % warpsize));
  __syncthreads();
  if (threadIdx.x < sh_flag_size) {
    for (int d=1;d < sh_flag_size;d *= 2) {
      int t = threadIdx.x + d;
      int flag_val = (t < sh_flag_size) ? sh_flag[t] : 0;
      __syncthreads();
      sh_flag[threadIdx.x] |= flag_val;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      int flag_val = sh_flag[0];
      atomicOr(global_flag, flag_val);
    }
  }

}

//############################################################################################
//############################################################################################
//############################################################################################

//
// Class creator
//
CudaPMEForcefield::CudaPMEForcefield() {
  //const FFTtype fft_type = BOX;
  //grid = Grid<int, float, float2>(nfftx, nffty, nfftz, order, fft_type, numnode, mynode);
  allocate<int>(&d_heuristic_flag, 1);
  allocate_host<int>(&h_heuristic_flag, 1);
  grid = NULL;
}

//
// Class destructor
//
CudaPMEForcefield::~CudaPMEForcefield() {
  deallocate<int>(&d_heuristic_flag);
  deallocate_host<int>(&h_heuristic_flag);
  if (grid != NULL) delete grid;
}

//
// Calculate forces
//
void CudaPMEForcefield::calc(const cudaXYZ<double> *coord,
			     const bool calc_energy, const bool calc_virial,
			     Force<long long int> *force) {

  // Check for neighborlist heuristic update
  if (heuristic_check(coord)) {
    // Update neighborlist
    nlist.sort(zone_patom, max_xyz, min_xyz, xyzq_unsorted.xyzq, xyzq_sorted.xyzq);
    nlist.build(boxx, boxy, boxz, rnl, xyzq_sorted.xyzq);
  } else {
    // Copy coordinates to xyzq -array
    xyzq.set_xyz(coord->data, coord->stride);
  }

  // Direct non-bonded force
  dir.calc_force(xyzq.xyzq, &nlist, calc_energy, calc_virial, force->xyz.stride, force->xyz.data);

  // 1-4 interactions
  dir.calc_14_force(xyzq.xyzq, calc_energy, calc_virial, force->xyz.stride, force->xyz.data);

  // Bonded forces
  float boxx = 1.0f;
  float boxy = 1.0f;
  float boxz = 1.0f;
  bonded.calc_force(xyzq.xyzq, boxx, boxy, boxz, calc_energy, calc_virial,
		    force->xyz.stride, force->xyz.data);

  // Reciprocal forces (Only reciprocal nodes calculate these)
  if (grid != NULL) {
    double recip[9];
    for (int i=0;i < 9;i++) recip[i] = 0;
    recip[0] = 1.0/boxx;
    recip[4] = 1.0/boxy;
    recip[8] = 1.0/boxz;
    double kappa = 0.32;
    grid->spread_charge(xyzq.xyzq, xyzq.ncoord, recip);
    grid->r2c_fft();
    grid->scalar_sum(recip, kappa, calc_energy, calc_virial);
    grid->c2r_fft();
    //grid->gather_force(xyzq.xyzq, xyzq.ncoord, recip, force->xyz.stride, force->xyz.data);
  }

}

//
// Checks if non-bonded list needs to be updated
// Returns true if update is needed
//
bool CudaPMEForcefield::heuristic_check(const cudaXYZ<double> *coord) {
  assert(ref_coord.match(coord));
  assert(warpsize <= 32);

  double rsq_limit_dbl = fabs(rnl - roff)/2.0;
  rsq_limit_dbl *= rsq_limit_dbl;
  float rsq_limit = (float)rsq_limit_dbl;

  int ncoord = ref_coord.n;
  int stride = ref_coord.stride;
  int nthread = 512;
  int nblock = (ncoord - 1)/nthread + 1;

  int shmem_size = nthread/warpsize;

  heuristic_check_kernel<<< nblock, nthread, shmem_size, 0 >>>
    (ncoord, stride, coord->data, ref_coord.data, rsq_limit, d_heuristic_flag);

  cudaCheck(cudaGetLastError());

  copy_DtoH_sync<int>(d_heuristic_flag, h_heuristic_flag, 1);
  
  return (*h_heuristic_flag != 0);
}
