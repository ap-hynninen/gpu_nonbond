#include <cassert>
#include "gpu_utils.h"
#include "CudaDomdec.h"

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

//
// Choose z coordinates
//
__global__ void choose_z_coord_kernel(const int ncoord, const float* __restrict__ zbound_f,
				      const float rcut_f, const float invz, const float* __restrict__ shz,
				      const double* __restrict__ coord,
				      unsigned char* __restrict__ coord_tags) {
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int izone = tid/ncoord;
  const int i = tid - izone*ncoord;

  float z = ((float)coord[i])*invz + 0.5f;
  z -= floor(z);
  z += shz[izone] - zbound_f[izone];
  z = max(0.0f, z);
  unsigned char tag;
  if (z < rcut_f) {
    // In the zone
    tag = 1;
  } else {
    tag = 0;
  }
  coord_tags[i + izone*ncoord] = tag;
}


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
}

//
// Class destructor
//
CudaDomdec::~CudaDomdec() {
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
  this->zone_ncoord[0] = homezone.build(coord);
  for (int i=1;i < 8;i++) zone_ncoord[i] = 0;
  this->update_zone_pcoord();
}

//
// Update coordinate distribution across all nodes
// Update is done according to coord, coord2 is a hangaround
// NOTE: Used during dynamics
//
void CudaDomdec::update_homezone(cudaXYZ<double>& coord, cudaXYZ<double>& coord2, cudaStream_t stream) {
  if (numnode > 1) {
    this->zone_ncoord[0] = homezone.update(coord, coord2, stream);
    for (int i=1;i < 8;i++) zone_ncoord[i] = 0;
    this->update_zone_pcoord();
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

    /*
    // ---------------- Z -------------------
    int nreq = 0;
    for (int i=0;i < nz_comm;i++) {
      if (z_recv_count[i] > 0) {
	cuda_irecv(z_recv_buf[i], z_recv_count[i], z_recv_node[i], &reqbuf[nreq]);
	nreq++;
      }
    }

    for (int i=0;i < nz_comm;i++)
      h_z_boundary[i] = get_fz_boundary(homeix, homeiy, homeiz-i, rnl, r_bonded);

    nthread = 512;
    nblock = (zone_ncoord[0] - 1)/nthread + 1;
    choose_z_coord_kernel<<< nblock, nthread, 0, stream >>>
      (zone_ncoord[0], z_boundary, rnl/boxz, coord->data, coord_tags);

    if (mpi_cuda_aware) {
      for (int i=0;i < nz_comm;i++) {
	if (z_send_count[i] > 0) {
	  cuda_isend(z_send_buf[i], z_send_count[i], z_send_node[i], &reqbuf[nreq]);
	  nreq++;
	}
      }
    } else {
      for (int i=0;i < nz_comm;i++) {
	if (z_send_count[i] > 0) {
	}
      }
    }

    cuda_waitall(nreq, reqbuf);

    // ---------------- Y -------------------

    for (int i=0;i < ny_comm;i++)
      h_y_boundary[i] = get_fy_boundary(homeix, homeiy-i, homeiz, rnl, r_bonded);

    for (int i=0;i < ny_comm;i++)
      get_ex_boundary(homeix, homeiy-i, homeiz, yf, zf, &
		      z_bonded, q_checkbonded, cut, rcut_bonded);

    // ---------------- X -------------------

    for (int i=0;i < nx_comm;i++)
      get_fx_boundary(homeix-i, xf);

    for (int i=0;i < nx_comm;i++)
      get_ez_boundary(homeix-i, homeiy, xf, yf, y_bonded, q_checkbonded);

    for (int i=0;i < nx_comm;i++)
      get_ey_boundary(homeix-i, homeiy, homeiz, xf, zf, z_bonded, q_checkbonded, cut);

    for (int i=0;i < nx_comm;i++) {
      get_c_boundary(homeix-i, homeiy, homeiz, xf, yf, zf, &
			  y_bonded, z_bonded, q_checkbonded);
      get_z0_for_c(homeix-i, homeiy, homeiz, z0);
    }
    */

    // Calculate xyz shift
    double x0 = 0.0;
    double y0 = 0.0;
    double z0 = 0.0;

    float fac = (numnode > 1) ? 1.2f : 1.0f;
    reallocate<float3>(&xyz_shift0, &xyz_shift0_len, zone_pcoord[7], fac);
    reallocate<float3>(&xyz_shift1, &xyz_shift1_len, zone_pcoord[7], fac);
    
    nthread = 512;
    nblock = (zone_pcoord[7] - 1)/nthread + 1;
    calc_xyz_shift<<< nblock, nthread, 0, stream >>>
      (zone_pcoord[7], coord.stride, coord.data,
       x0, y0, z0, this->get_inv_boxx(), this->get_inv_boxy(), this->get_inv_boxz(), xyz_shift0);
    cudaCheck(cudaGetLastError());
  }

}

//
// Communicate forces
//
void CudaDomdec::comm_force(Force<long long int>& force, cudaStream_t stream) {
}

//
// Re-order coordinates using ind_sorted: coord_src => coord_dst
//
void CudaDomdec::reorder_coord(cudaXYZ<double>& coord_src, cudaXYZ<double>& coord_dst,
			       const int* ind_sorted, cudaStream_t stream) {
  assert(coord_src.match(&coord_dst));
  assert(zone_pcoord[7] == coord_src.n);

  if (numnode == 1) {
    int nthread = 512;
    int nblock = (zone_pcoord[7] - 1)/nthread + 1;
    reorder_coord_kernel<<< nblock, nthread, 0, stream >>>
      (zone_pcoord[7], coord_src.stride, ind_sorted, coord_src.data, coord_dst.data);
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
