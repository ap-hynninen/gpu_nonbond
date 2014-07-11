
#include <cassert>
#include "gpu_utils.h"
#include "CudaDomdec.h"

//
// Builds homezone
//
__global__ void build_homezone_kernel(const int ncoord, int* __restrict__ loc2glo) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < ncoord) {
    loc2glo[i] = i;
  }
}

//
// Calculates (x, y, z) shift
// (x0, y0, z0) = fractional origin
//
__global__ void calc_xyz_shift(const int ncoord, const int stride, const double* __restrict__ coord,
			       const double x0, const double y0, const double z0,
			       const double inv_boxx, const double inv_boxy, const double inv_boxz,
			       float3* __restrict__ xyz_shift) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < ncoord) {
    double x = coord[i]*inv_boxx;
    double y = coord[i+stride]*inv_boxy;
    double z = coord[i+stride*2]*inv_boxz;
    float3 shift;
    shift.x = ceilf(x0 - x);
    shift.y = ceilf(y0 - y);
    shift.z = ceilf(z0 - z);
    xyz_shift[i] = shift;
  }
}

//
// Re-order coordinates
//
__global__ void reorder_coord_kernel(const int ncoord, const int stride,
				     const int* __restrict__ ind_sorted,
				     const double* __restrict__ coord_src,
				     double* __restrict__ coord_dst) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  const int stride2 = stride*2;
  if (i < ncoord) {
    int j = ind_sorted[i];
    coord_dst[i]         = coord_src[j];
    coord_dst[i+stride]  = coord_src[j+stride];
    coord_dst[i+stride2] = coord_src[j+stride2];
  }
}

//
// Re-order xyz_shift
//
__global__ void reorder_xyz_shift_kernel(const int ncoord,
					 const int* __restrict__ ind_sorted,
					 const float3* __restrict__ xyz_shift_in,
					 float3* __restrict__ xyz_shift_out) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < ncoord) {
    int j = ind_sorted[i];
    xyz_shift_out[i] = xyz_shift_in[j];
  }
}

//
// Re-order mass
//
__global__ void reorder_mass_kernel(const int ncoord,
				    const int* __restrict__ ind_sorted,
				    const float* __restrict__ mass_in,
				    float* __restrict__ mass_out) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < ncoord) {
    int j = ind_sorted[i];
    mass_out[i] = mass_in[j];
  }
}

//#############################################################################################
//#############################################################################################
//#############################################################################################

//
// Class creator
//
CudaDomdec::CudaDomdec(int ncoord_glo, double boxx, double boxy, double boxz, double rnl,
		       int nx, int ny, int nz, int mynode) {
  this->ncoord_glo = ncoord_glo;
  this->boxx = boxx;
  this->boxy = boxy;
  this->boxz = boxz;
  this->rnl = rnl;
  this->nx = nx;
  this->ny = ny;
  this->nz = nz;
  this->numnode = nx*ny*nz;
  this->mynode = mynode;
  
  loc2glo_len = 0;
  loc2glo = NULL;

  xyz_shift0_len = 0;
  xyz_shift0 = NULL;

  xyz_shift1_len = 0;
  xyz_shift1 = NULL;

  mass_tmp_len = 0;
  mass_tmp = NULL;
}

//
// Class destructor
//
CudaDomdec::~CudaDomdec() {
  if (loc2glo != NULL) deallocate<int>(&loc2glo);
  if (xyz_shift0 != NULL) deallocate<float3>(&xyz_shift0);
  if (xyz_shift1 != NULL) deallocate<float3>(&xyz_shift1);
  if (mass_tmp != NULL) deallocate<float>(&mass_tmp);
}

//
// Builds coordinate distribution across all nodes
// NOTE: Here all nodes have all coordinates.
// NOTE: Used only in the beginning of dynamics
//
void CudaDomdec::build_homezone(cudaXYZ<double> *coord, cudaStream_t stream) {
  if (numnode == 1) {

    ncoord = coord->n;
    zone_ncoord[0] = coord->n;
    for (int i=1;i < 8;i++) zone_ncoord[i] = 0;

    reallocate<int>(&loc2glo, &loc2glo_len, coord->n);

    int nthread = 512;
    int nblock = (coord->n - 1)/nthread + 1;
    int shmem_size = 0;
    build_homezone_kernel<<< nblock, nthread, shmem_size, stream >>>
      (coord->n, loc2glo);
    cudaCheck(cudaGetLastError());
  } else {
    std::cerr << "CudaDomdec::build_homezone, numnode > 1 not implemented" << std::endl;
    exit(1);
  }
}

//
// Update coordinate distribution across all nodes
// NOTE: Used during dynamics
//
void CudaDomdec::update_homezone(cudaXYZ<double> *coord, cudaXYZ<double> *coord2, cudaStream_t stream) {
  /*
  int *h_loc2glo = new int[23558];
  copy_DtoH_sync<int>(loc2glo, h_loc2glo, 23558);
  for (int i=0;i < 10;i++)
    std::cerr << h_loc2glo[i] << std::endl;
  delete [] h_loc2glo;
  */
}

//
// Communicate coordinates
//
void CudaDomdec::comm_coord(cudaXYZ<double> *coord, const bool update, cudaStream_t stream) {

  // Calculate zone_pcoord
  zone_pcoord[0] = zone_ncoord[0];
  for (int i=1;i < 8;i++) {
    zone_pcoord[i] = zone_pcoord[i-1] + zone_ncoord[i];
  }

  // Calculate xyz_shift
  if (update) {
    double x0 = 0.0;
    double y0 = 0.0;
    double z0 = 0.0;
    double inv_boxx = 1.0/boxx;
    double inv_boxy = 1.0/boxy;
    double inv_boxz = 1.0/boxz;

    float fac = (numnode > 1) ? 1.2f : 1.0f;
    reallocate<float3>(&xyz_shift0, &xyz_shift0_len, zone_pcoord[7], fac);
    reallocate<float3>(&xyz_shift1, &xyz_shift1_len, zone_pcoord[7], fac);
    
    int nthread = 512;
    int nblock = (zone_pcoord[7] - 1)/nthread + 1;
    calc_xyz_shift<<< nblock, nthread, 0, stream >>>
      (zone_pcoord[7], coord->stride, coord->data,
       x0, y0, z0, inv_boxx, inv_boxy, inv_boxz, xyz_shift0);
    cudaCheck(cudaGetLastError());
  }

}

//
// Communicate forces
//
void CudaDomdec::comm_force(Force<long long int> *force, cudaStream_t stream) {
}

//
// Re-order coordinates using loc2glo: coord_src => coord_dst
//
void CudaDomdec::reorder_coord(cudaXYZ<double> *coord_src, cudaXYZ<double> *coord_dst,
			       const int* ind_sorted, cudaStream_t stream) {
  assert(coord_src->match(coord_dst));
  assert(zone_pcoord[7] == coord_src->n);

  if (numnode == 1) {
    int nthread = 512;
    int nblock = (zone_pcoord[7] - 1)/nthread + 1;
    reorder_coord_kernel<<< nblock, nthread, 0, stream >>>
      (zone_pcoord[7], coord_src->stride, ind_sorted, coord_src->data, coord_dst->data);
    cudaCheck(cudaGetLastError());
  } else {
    std::cerr << "CudaDomdec::reorder_coord, not ready for numnode > 1" << std::endl;
    exit(1);
  }

}

//
// Re-order xyz_shift
//
void CudaDomdec::reorder_xyz_shift(const int* ind_sorted, cudaStream_t stream) {

  int nthread = 512;
  int nblock = (zone_pcoord[7] - 1)/nthread + 1;
  reorder_xyz_shift_kernel<<< nblock, nthread, 0, stream >>>
    (zone_pcoord[7], ind_sorted, xyz_shift0, xyz_shift1);
  cudaCheck(cudaGetLastError());

  float3 *p = xyz_shift0;
  xyz_shift0 = xyz_shift1;
  xyz_shift1 = p;

  int t = xyz_shift0_len;
  xyz_shift0_len = xyz_shift1_len;
  xyz_shift1_len = t;
}

//
// Re-order mass
//
void CudaDomdec::reorder_mass(float *mass, const int* ind_sorted, cudaStream_t stream) {

  reallocate<float>(&mass_tmp, &mass_tmp_len, zone_pcoord[7], 1.2f);

  int nthread = 512;
  int nblock = (zone_pcoord[7] - 1)/nthread + 1;
  reorder_mass_kernel<<< nblock, nthread, 0, stream >>>
    (zone_pcoord[7], ind_sorted, mass, mass_tmp);
  cudaCheck(cudaGetLastError());

  copy_DtoD<float>(mass_tmp, mass, zone_pcoord[7], stream);
}
