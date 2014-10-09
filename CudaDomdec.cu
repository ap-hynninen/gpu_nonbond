#include <cassert>
#include "gpu_utils.h"
#include "CudaDomdec.h"

//
// Calculates (x, y, z) shift
// (x0, y0, z0) = fractional origin
//
__global__ void calc_xyz_shift(const int ncoord, 
			       const double* __restrict__ x,
			       const double* __restrict__ y,
			       const double* __restrict__ z,
			       const double x0, const double y0, const double z0,
			       const double inv_boxx, const double inv_boxy, const double inv_boxz,
			       const int* __restrict__ loc2glo,
			       const float lox, const float hix,
			       const float loy, const float hiy,
			       const float loz, const float hiz,
			       float3* __restrict__ xyz_shift,
			       char* __restrict__ coordLoc) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < ncoord) {
    double xi = x[i]*inv_boxx;
    double yi = y[i]*inv_boxy;
    double zi = z[i]*inv_boxz;
    float3 shift;
    shift.x = ceilf(x0 - xi);
    shift.y = ceilf(y0 - yi);
    shift.z = ceilf(z0 - zi);
    xyz_shift[i] = shift;
    float xf = (float)xi + shift.x;
    float yf = (float)yi + shift.y;
    float zf = (float)zi + shift.z;
    // (xf, yf, zf) is in range (0...1)
    int iglo = loc2glo[i];
    int loca = (xf >= lox && xf < hix) | ((yf >= loy && yf < hiy) << 1) | ((zf >= loz && zf < hiz) << 2);
    coordLoc[iglo] = (char)loca;
  }
}

//
// Re-order coordinates
//
__global__ void reorder_coord_kernel(const int ncoord,
				     const int* __restrict__ ind_sorted,
				     const double* __restrict__ x_src,
				     const double* __restrict__ y_src,
				     const double* __restrict__ z_src,
				     double* __restrict__ x_dst,
				     double* __restrict__ y_dst,
				     double* __restrict__ z_dst) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < ncoord) {
    int j = ind_sorted[i];
    x_dst[i] = x_src[j];
    y_dst[i] = y_src[j];
    z_dst[i] = z_src[j];
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

/*
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
*/

//#############################################################################################
//#############################################################################################
//#############################################################################################

//
// Class creator
//
CudaDomdec::CudaDomdec(int ncoord_glo, double boxx, double boxy, double boxz, double rnl,
		       int nx, int ny, int nz, int mynode, CudaMPI& cudaMPI) : 
  Domdec(ncoord_glo, boxx, boxy, boxz, rnl, nx, ny, nz, mynode), homezone(*this, cudaMPI), 
  D2Dcomm(*this, cudaMPI) {

  xyz_shift0_len = 0;
  xyz_shift0 = NULL;

  xyz_shift1_len = 0;
  xyz_shift1 = NULL;

  mass_tmp_len = 0;
  mass_tmp = NULL;

  allocate<char>(&coordLoc, ncoord_glo);
  clear_gpu_array_sync<char>(coordLoc, ncoord_glo);
}

//
// Class destructor
//
CudaDomdec::~CudaDomdec() {
  deallocate<char>(&coordLoc);
  if (xyz_shift0 != NULL) deallocate<float3>(&xyz_shift0);
  if (xyz_shift1 != NULL) deallocate<float3>(&xyz_shift1);
  if (mass_tmp != NULL) deallocate<float>(&mass_tmp);
}

//
// Builds coordinate distribution across all nodes
// NOTE: Here all nodes have all coordinates.
// NOTE: Used only in the beginning of dynamics
//
void CudaDomdec::build_homezone(hostXYZ<double>& coord) {
  this->clear_zone_ncoord();
  this->set_zone_ncoord(I, homezone.build(coord));
}

//
// Update coordinate distribution across all nodes
// Update is done according to coord, coord2 is a hangaround
// NOTE: Used during dynamics
//
void CudaDomdec::update_homezone(cudaXYZ<double>& coord, cudaXYZ<double>& coord2, cudaStream_t stream) {
  if (numnode > 1) {
    this->clear_zone_ncoord();
    this->set_zone_ncoord(I, homezone.update(coord, coord2, stream));
  }
}

//
// Communicate coordinates
//
void CudaDomdec::comm_coord(cudaXYZ<double>& coord, const bool update, cudaStream_t stream) {

  D2Dcomm.comm_coord(coord, homezone.get_loc2glo(), update);

  // Calculate xyz_shift
  if (update) {
    int nthread, nblock;

    // Calculate xyz shift
    double x0 = 0.0;
    double y0 = 0.0;
    double z0 = 0.0;

    // Re-allocate (xyz_shift0, xyz_shift1)
    float fac = (numnode > 1) ? 1.2f : 1.0f;
    reallocate<float3>(&xyz_shift0, &xyz_shift0_len, this->get_ncoord_tot(), fac);
    reallocate<float3>(&xyz_shift1, &xyz_shift1_len, this->get_ncoord_tot(), fac);    

    nthread = 512;
    nblock = (this->get_ncoord_tot() - 1)/nthread + 1;
    calc_xyz_shift<<< nblock, nthread, 0, stream >>>
      (this->get_ncoord_tot(), coord.x(), coord.y(), coord.z(),
       x0, y0, z0, this->get_inv_boxx(), this->get_inv_boxy(), this->get_inv_boxz(),
       this->get_loc2glo_ptr(),
       
       xyz_shift0, coordLoc);
    cudaCheck(cudaGetLastError());
  }

}

//
// Update communication (we're updating the local receive indices)
//
void CudaDomdec::comm_update(int* glo2loc, int* loc2loc, cudaStream_t stream) {
  D2Dcomm.update(glo2loc, loc2loc);
}

//
// Communicate forces
//
void CudaDomdec::comm_force(Force<long long int>& force, cudaStream_t stream) {
  D2Dcomm.comm_force(force);
}

//
// Re-order coordinates using ind_sorted: coord_src => coord_dst
//
void CudaDomdec::reorder_coord(cudaXYZ<double>& coord_src, cudaXYZ<double>& coord_dst,
			       const int* ind_sorted, cudaStream_t stream) {
  assert(coord_src.match(coord_dst));
  assert(this->get_ncoord_tot() == coord_src.size());

  if (numnode == 1) {
    int nthread = 512;
    int nblock = (this->get_ncoord_tot() - 1)/nthread + 1;
    reorder_coord_kernel<<< nblock, nthread, 0, stream >>>
      (this->get_ncoord_tot(), ind_sorted, coord_src.x(), coord_src.y(), coord_src.z(),
       coord_dst.x(), coord_dst.y(), coord_dst.z());
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
  int nblock = (this->get_ncoord_tot() - 1)/nthread + 1;
  reorder_xyz_shift_kernel<<< nblock, nthread, 0, stream >>>
    (this->get_ncoord_tot(), ind_sorted, xyz_shift0, xyz_shift1);
  cudaCheck(cudaGetLastError());

  float3 *p = xyz_shift0;
  xyz_shift0 = xyz_shift1;
  xyz_shift1 = p;

  int t = xyz_shift0_len;
  xyz_shift0_len = xyz_shift1_len;
  xyz_shift1_len = t;
}

/*
//
// Re-order mass
//
void CudaDomdec::reorder_mass(float *mass, const int* ind_sorted, cudaStream_t stream) {

  float fac = (numnode > 1) ? 1.2f : 1.0f;
  reallocate<float>(&mass_tmp, &mass_tmp_len, this->get_ncoord_tot(), fac);

  int nthread = 512;
  int nblock = (this->get_ncoord_tot() - 1)/nthread + 1;
  reorder_mass_kernel<<< nblock, nthread, 0, stream >>>
    (this->get_ncoord_tot(), ind_sorted, mass, mass_tmp);
  cudaCheck(cudaGetLastError());

  copy_DtoD<float>(mass_tmp, mass, this->get_ncoord_tot(), stream);
}
*/
