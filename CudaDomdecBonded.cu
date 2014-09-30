#include <cassert>
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "CudaDomdecBonded.h"

__global__ void build_tbl_kernel_single(const int nbond_tbl, int* __restrict__ bond_tbl,
					const int nureyb_tbl, int* __restrict__ ureyb_tbl,
					const int nangle_tbl, int* __restrict__ angle_tbl,
					const int ndihe_tbl, int* __restrict__ dihe_tbl,
					const int nimdihe_tbl, int* __restrict__ imdihe_tbl,
					const int ncmap_tbl, int* __restrict__ cmap_tbl,
					const int nin14_tbl, int* __restrict__ in14_tbl,
					const int nex14_tbl, int* __restrict__ ex14_tbl) {
  const int pos = threadIdx.x + blockIdx.x*blockDim.x;
  if (pos < nbond_tbl) {
    bond_tbl[pos] = pos;
  } else if (pos < nbond_tbl + nureyb_tbl) {
    ureyb_tbl[pos-nbond_tbl] = pos-nbond_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl) {
    angle_tbl[pos-nbond_tbl-nureyb_tbl] = pos-nbond_tbl-nureyb_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl) {
    dihe_tbl[pos-nbond_tbl-nureyb_tbl-nangle_tbl] = pos-nbond_tbl-nureyb_tbl-nangle_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl) {
    imdihe_tbl[pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl] = 
      pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl + ncmap_tbl) {
    cmap_tbl[pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl] = 
      pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl + ncmap_tbl
	     + nin14_tbl) {
    in14_tbl[pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl-ncmap_tbl] = 
      pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl-ncmap_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl + ncmap_tbl
	     + nin14_tbl + nex14_tbl) {
    ex14_tbl[pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl-ncmap_tbl-nin14_tbl] = 
      pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl-ncmap_tbl-nin14_tbl;
  }
}

__global__ void build_tbl_kernel(
				 const int nbond_tbl, int* __restrict__ bond_tbl,
				 const int nureyb_tbl, int* __restrict__ ureyb_tbl,
				 const int nangle_tbl, int* __restrict__ angle_tbl,
				 const int ndihe_tbl, int* __restrict__ dihe_tbl,
				 const int nimdihe_tbl, int* __restrict__ imdihe_tbl,
				 const int ncmap_tbl, int* __restrict__ cmap_tbl,
				 const int nin14_tbl, int* __restrict__ in14_tbl,
				 const int nex14_tbl, int* __restrict__ ex14_tbl) {
  const int pos = threadIdx.x + blockIdx.x*blockDim.x;
  if (pos < nbond_tbl) {
    bond_tbl[pos] = pos;
  } else if (pos < nbond_tbl + nureyb_tbl) {
    ureyb_tbl[pos-nbond_tbl] = pos-nbond_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl) {
    angle_tbl[pos-nbond_tbl-nureyb_tbl] = pos-nbond_tbl-nureyb_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl) {
    dihe_tbl[pos-nbond_tbl-nureyb_tbl-nangle_tbl] = pos-nbond_tbl-nureyb_tbl-nangle_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl) {
    imdihe_tbl[pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl] = 
      pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl + ncmap_tbl) {
    cmap_tbl[pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl] = 
      pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl + ncmap_tbl
	     + nin14_tbl) {
    in14_tbl[pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl-ncmap_tbl] = 
      pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl-ncmap_tbl;
  } else if (pos < nbond_tbl + nureyb_tbl + nangle_tbl + ndihe_tbl + nimdihe_tbl + ncmap_tbl
	     + nin14_tbl + nex14_tbl) {
    ex14_tbl[pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl-ncmap_tbl-nin14_tbl] = 
      pos-nbond_tbl-nureyb_tbl-nangle_tbl-ndihe_tbl-nimdihe_tbl-ncmap_tbl-nin14_tbl;
  }
}

//############################################################################################
//############################################################################################
//############################################################################################

//
// Class creator
//
CudaDomdecBonded::CudaDomdecBonded(const CudaDomdec& domdec, const int nbond, const bond_t* h_bond,
				   const int nureyb, const bond_t* h_ureyb,
				   const int nangle, const angle_t* h_angle,
				   const int ndihe, const dihe_t* h_dihe,
				   const int nimdihe, const dihe_t* h_imdihe,
				   const int ncmap, const cmap_t* h_cmap,
				   const int nin14, const xx14_t* h_in14,
				   const int nex14, const xx14_t* h_ex14) : domdec(domdec) {
  assert((nureyb == 0) || (nureyb > 0 && nureyb == nangle));

  bond = NULL;
  ureyb = NULL;
  angle = NULL;
  dihe = NULL;
  imdihe = NULL;
  cmap = NULL;
  in14 = NULL;
  ex14 = NULL;

  this->nbond = nbond;
  if (nbond > 0) {
    allocate<bond_t>(&bond, nbond);
    copy_HtoD<bond_t>(h_bond, bond, nbond);
  }

  this->nureyb = nureyb;
  if (nureyb > 0) {
    allocate<bond_t>(&ureyb, nureyb);
    copy_HtoD<bond_t>(h_ureyb, ureyb, nureyb);
  }

  this->nangle = nangle;
  if (nangle > 0) {
    allocate<angle_t>(&angle, nangle);
    copy_HtoD<angle_t>(h_angle, angle, nangle);
  }

  this->ndihe = ndihe;
  if (ndihe > 0) {
    allocate<dihe_t>(&dihe, ndihe);
    copy_HtoD<dihe_t>(h_dihe, dihe, ndihe);
  }

  this->nimdihe = nimdihe;
  if (nimdihe > 0) {
    allocate<dihe_t>(&imdihe, nimdihe);
    copy_HtoD<dihe_t>(h_imdihe, imdihe, nimdihe);
  }

  this->ncmap = ncmap;
  if (ncmap > 0) {
    allocate<cmap_t>(&cmap, ncmap);
    copy_HtoD<cmap_t>(h_cmap, cmap, ncmap);
  }

  this->nin14 = nin14;
  if (nin14 > 0) {
    allocate<xx14_t>(&in14, nin14);
    copy_HtoD<xx14_t>(h_in14, in14, nin14);
  }

  this->nex14 = nex14;
  if (nex14 > 0) {
    allocate<xx14_t>(&ex14, nex14);
    copy_HtoD<xx14_t>(h_ex14, ex14, nex14);
  }

  nbond_tbl = 0;
  bond_tbl_len = 0;
  bond_tbl = NULL;

  nureyb_tbl = 0;
  ureyb_tbl_len = 0;
  ureyb_tbl = NULL;

  nangle_tbl = 0;
  angle_tbl_len = 0;
  angle_tbl = NULL;

  ndihe_tbl = 0;
  dihe_tbl_len = 0;
  dihe_tbl = NULL;

  nimdihe_tbl = 0;
  imdihe_tbl_len = 0;
  imdihe_tbl = NULL;

  ncmap_tbl = 0;
  cmap_tbl_len = 0;
  cmap_tbl = NULL;

  nin14_tbl = 0;
  in14_tbl_len = 0;
  in14_tbl = NULL;

  nex14_tbl = 0;
  ex14_tbl_len = 0;
  ex14_tbl = NULL;

  tbl_upto_date = false;
}

//
// Class destructor
//
CudaDomdecBonded::~CudaDomdecBonded() {
  if (bond != NULL) deallocate<bond_t>(&bond);
  if (ureyb != NULL) deallocate<bond_t>(&ureyb);
  if (angle != NULL) deallocate<angle_t>(&angle);
  if (dihe != NULL) deallocate<dihe_t>(&dihe);
  if (imdihe != NULL) deallocate<dihe_t>(&imdihe);
  if (cmap != NULL) deallocate<cmap_t>(&cmap);
  if (in14 != NULL) deallocate<xx14_t>(&in14);
  if (ex14 != NULL) deallocate<xx14_t>(&ex14);

  if (bond_tbl != NULL) deallocate<int>(&bond_tbl);
  if (ureyb_tbl != NULL) deallocate<int>(&ureyb_tbl);
  if (angle_tbl != NULL) deallocate<int>(&angle_tbl);
  if (dihe_tbl != NULL) deallocate<int>(&dihe_tbl);
  if (imdihe_tbl != NULL) deallocate<int>(&imdihe_tbl);
  if (cmap_tbl != NULL) deallocate<int>(&cmap_tbl);
  if (in14_tbl != NULL) deallocate<int>(&in14_tbl);
  if (ex14_tbl != NULL) deallocate<int>(&ex14_tbl);
}

//
// Build tables
//
void CudaDomdecBonded::build_tbl(cudaStream_t stream) {

  if (domdec.get_numnode() == 1 && !tbl_upto_date) {
    nbond_tbl = nbond;
    nureyb_tbl = nureyb;
    nangle_tbl = nangle;
    ndihe_tbl = ndihe;
    nimdihe_tbl = nimdihe;
    ncmap_tbl = ncmap;
    nin14_tbl = nin14;
    nex14_tbl = nex14;

    if (nbond_tbl > 0) reallocate<int>(&bond_tbl, &bond_tbl_len, nbond_tbl);
    if (nureyb_tbl > 0) reallocate<int>(&ureyb_tbl, &ureyb_tbl_len, nureyb_tbl);
    if (nangle_tbl > 0) reallocate<int>(&angle_tbl, &angle_tbl_len, nangle_tbl);
    if (ndihe_tbl > 0) reallocate<int>(&dihe_tbl, &dihe_tbl_len, ndihe_tbl);
    if (nimdihe_tbl > 0) reallocate<int>(&imdihe_tbl, &imdihe_tbl_len, nimdihe_tbl);
    if (ncmap_tbl > 0) reallocate<int>(&cmap_tbl, &cmap_tbl_len, ncmap_tbl);
    if (nin14_tbl > 0) reallocate<int>(&in14_tbl, &in14_tbl_len, nin14_tbl);
    if (nex14_tbl > 0) reallocate<int>(&ex14_tbl, &ex14_tbl_len, nex14_tbl);

    int nthread = 512;
    int nblock = (nbond_tbl + nureyb_tbl + nangle_tbl + 
		  ndihe_tbl + nimdihe_tbl + ncmap_tbl +
		  nin14_tbl + nex14_tbl - 1)/nthread + 1;
    build_tbl_kernel_single<<< nblock, nthread, 0, stream >>>
      (nbond_tbl, bond_tbl,
       nureyb_tbl, ureyb_tbl,
       nangle_tbl, angle_tbl,
       ndihe_tbl, dihe_tbl,
       nimdihe_tbl, imdihe_tbl,
       ncmap_tbl, cmap_tbl,
       nin14_tbl, in14_tbl,
       nex14_tbl, ex14_tbl);
    cudaCheck(cudaGetLastError());

    tbl_upto_date = true;
  } else {
    std::cerr << "CudaDomdecBonded::build_tbl, numnode > 1 not implemented yet" << std::endl;
    exit(1);
  }

}
