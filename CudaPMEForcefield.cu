#include <cassert>
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
CudaPMEForcefield::CudaPMEForcefield(const int nbondlist, const bondlist_t* h_bondlist,
				     const int nureyblist, const bondlist_t* h_ureyblist,
				     const int nanglelist, const anglelist_t* h_anglelist,
				     const int ndihelist, const dihelist_t* h_dihelist,
				     const int nimdihelist, const dihelist_t* imdihelist,
				     const int ncmaplist, const cmaplist_t* cmaplist,
				     const int nbondcoef, const float2 *h_bondcoef,
				     const int nureybcoef, const float2 *h_ureybcoef,
				     const int nanglecoef, const float2 *h_anglecoef,
				     const int ndihecoef, const float4 *h_dihecoef,
				     const int nimdihecoef, const float4 *h_imdihecoef,
				     const int ncmapcoef, const float2 *h_cmapcoef,
				     const double rnl, const double roff, const double ron,
				     const double kappa, const double e14fac,
				     const int vdw_model, const int elec_model,
				     const int nvdwparam, const float *h_vdwparam,
				     const int *h_vdwtype,
				     const int nfftx, const int nffty, const int nttz,
				     const int order) {

  // Bonded interactions
  setup_bonded(nbondlist, bondlist, nureyblist, ureyblist, nanglelist, anglelist,
	       ndihelist, dihelist, nimdihelist, imdihelist, ncmaplist, cmaplist);
  Bonded.setup_coef(nbondcoef, h_bondcoef, nureybcoef, h_ureybcoef,
		    nanglecoef, h_anglecoef, ndihecoef, h_dihecoef,
		    nimdihecoef, h_imdihecoef, ncmapcoef, h_cmapcoef);
  
  // Direct non-bonded interactions
  setup_direct_nonbonded(rnl, roff, ron, kappa, e14fac, vdw_model, elec_model,
			 nvdwparam, vdwparam, vdwtype);

  // Recip non-bonded interactions
  setup_recip_nonbonded(nfftx, nffty, nfftz, order);

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
// Setup direct non-bonded interactions.
//
void CudaPMEForcefield::setup_direct_nonbonded(const double rnl, const double roff, const double ron,
					       const double kappa, const double e14fac,
					       const int vdw_model, const int elec_model,
					       const int nvdwparam, const float *h_vdwparam,
					       const int *h_vdwtype) {

  dir.setup(boxx, boxy, boxz, kappa, roff, ron, e14fac, vdw_model, elec_model, true, true);
  dir.setup_vdwparam(nvdwparam, h_vdwparam);

  allocate<int>(&vdwtype, ncoord_glo);
  copy_HtoD<anglelist_t>(h_vdwtype, vdwtype, ncoord_glo);
}

//
// Setup bonded interactions. Copies in the global lists
//
void CudaPMEForcefield::setup_bonded(const int nbondlist, const bondlist_t* h_bondlist,
				     const int nureyblist, const bondlist_t* h_ureyblist,
				     const int nanglelist, const anglelist_t* h_anglelist,
				     const int ndihelist, const dihelist_t* h_dihelist,
				     const int nimdihelist, const dihelist_t* imdihelist,
				     const int ncmaplist, const cmaplist_t* cmaplist) {  
  assert((nureyblist == 0) || (nureyblist > 0 && nureyblist == nanglelist));

  bondlist = NULL;
  ureyblist = NULL;
  anglelist = NULL;
  dihelist = NULL;
  imdihelist = NULL;
  cmaplist = NULL;

  this->nbondlist = nbondlist;
  if (nbondlist > 0) {
    allocate<bondlist_t>(&bondlist, nbondlist);
    copy_HtoD<bondlist_t>(h_bondlist, bondlist, nbondlist);
  }

  this->nureyblist = nureyblist;
  if (nureyblist > 0) {
    allocate<bondlist_t>(&ureyblist, nureyblist);
    copy_HtoD<bondlist_t>(h_ureyblist, ureyblist, nureyblist);
  }

  this->nanglelist = nanglelist;
  if (nanglelist > 0) {
    allocate<anglelist_t>(&anglelist, nanglelist);
    copy_HtoD<anglelist_t>(h_anglelist, anglelist, nanglelist);
  }

  this->ndihelist = ndihelist;
  if (ndihelist > 0) {
    allocate<dihelist_t>(&dihelist, ndihelist);
    copy_HtoD<dihelist_t>(h_dihelist, dihelist, ndihelist);
  }

  this->nimdihelist = nimdihelist;
  if (nimdihelist > 0) {
    allocate<dihelist_t>(&imdihelist, nimdihelist);
    copy_HtoD<dihelist_t>(h_imdihelist, imdihelist, nimdihelist);
  }

  this->ncmaplist = ncmaplist;
  if (ncmaplist > 0) {
    allocate<cmaplist_t>(&cmaplist, ncmaplist);
    copy_HtoD<cmaplist_t>(h_cmaplist, cmaplist, ncmaplist);
  }

}

//
// Calculate forces
//
void CudaPMEForcefield::calc(const cudaXYZ<double> *coord,
			     const bool calc_energy, const bool calc_virial,
			     Force<long long int> *force) {

  float boxx = 1.0f;
  float boxy = 1.0f;
  float boxz = 1.0f;
  int zone_patom[8] = {23558, 23558, 23558, 23558, 23558, 23558, 23558, 23558};

  // Check for neighborlist heuristic update
  if (heuristic_check(coord)) {
    // Copy coordinates to xyzq -array
    xyzq.set_xyz(coord->data, coord->stride);
    // Update neighborlist
    nlist.sort(zone_patom, xyzq.xyzq, xyzq_sorted.xyzq);
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
