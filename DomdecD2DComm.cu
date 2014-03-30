#include <iostream>
#include <cassert>
#include <cuda.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "DomdecD2DComm.h"

//
//
//
__global__ void pack_forces_kernel(const int n, const int *atomind,
				   const double *force, const int stride,
				   const double *xyz_tmp, double *buffer) {
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int stride2 = stride*2;

  if (tid < n) {
    int j = atomind[tid];
    buf[tid]         = force[j]         + xyz_tmp[j];
    buf[tid+stride]  = force[j+stride]  + xyz_tmp[j+stride];
    buf[tid+stride2] = force[j+stride2] + xyz_tmp[j+stride2];
  }
  
}

//
//
//
__global__ void unpack_forces_kernel(const int n, const int *atomind,
				     const doble *buffer, const int stride,
				     double *force) {
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int stride2 = stride*2;

  if (tid < n) {
    int j = atomind[tid];
    force[j]         += buffer[tid];
    force[j+stride]  += buffer[tid+stride];
    force[j+stride2] += buffer[tid+stride2];
  }

}

//####################################################################################

//
// Class creator
//
DomdecD2DComm::DomdecD2DComm(int nx_comm, int ny_comm, int nz_comm) {
  // Only y-direction implemented for now
  assert(nx_comm == 0);
  assert(nz_comm == 0);

  this->nx_comm = nx_comm;
  this->ny_comm = ny_comm;
  this->nz_comm = nz_comm;

  y_recv_ncoord_tot = 0;
  y_recv_ncoord = new int[ny_comm];
  y_recv_pos = new int[ny_comm];
  y_recv_node = new int[ny_comm];

  y_send_ncoord_tot = 0;
  y_send_ncoord = new int[ny_comm];
  y_send_pos = new int[ny_comm];
  y_send_node = new int[ny_comm];

  y_recv_atomind_len = 0;
  y_recv_atomind = NULL;

  y_recv_buf_len = 0;
  y_recv_buf = NULL;

  xyz_tmp_len = 0;
  xyz_tmp = NULL;

  ydir = new DomdecMPI(comm, YTAG, 2*ny_comm);
}

//
// Class destructor
//
DomdecD2DComm::~DomdecD2DComm() {
  if (y_recv_atomind != NULL) deallocate<int>(&y_recv_atomind);
  if (y_recv_buf != NULL) deallocate<double>(&y_recv_buf);
  if (xyz_tmp != NULL) deallocate<double>(&xyz_tmp);
  delete [] y_recv_ncoord;
  delete [] y_send_ncoord;
  delete [] y_recv_pos;
  delete [] y_send_pos;
  delete [] y_recv_node;
  delete [] y_send_node;

  //
  delete ydir;
}

//
// Setup communications
//
void DomdecD2DComm::setup_comm(int *y_recv_node_in, int *y_send_node_in) {
  for (int i=0;i < ny_comm) y_recv_node[i] = y_recv_node_in[i];
  for (int i=0;i < ny_comm) y_send_node[i] = y_send_node_in[i];
}

//
// Setup atom indices
//
void DomdecD2DComm::setup_atomind(int *y_recv_ncoord_in, int *h_y_recv_atomind_in,
				  int *y_send_ncoord_in, int *h_y_send_atomind_in) {

  for (int i=0;i < ny_comm) y_recv_ncoord[i] = y_recv_ncoord_in[i];
  for (int i=0;i < ny_comm) y_send_ncoord[i] = y_send_ncoord_in[i];

  y_recv_ncoord_tot = 0;
  for (int i=0;i < ny_comm) y_recv_ncoord_tot += y_recv_ncoord[i];
  y_recv_pos[0] = 0;
  for (int i=1;i < ny_comm) y_recv_pos[i] = y_recv_pos[i-1] + y_recv_ncoord[i];

  y_send_ncoord_tot = 0;
  for (int i=0;i < ny_comm) y_send_ncoord_tot += y_send_ncoord[i];
  y_send_pos[0] = 0;
  for (int i=1;i < ny_comm) y_send_pos[i] = y_send_pos[i-1] + y_send_ncoord[i];

  reallocate<int>(&y_recv_atomind, &y_recv_atomind_len, y_recv_ncoord_tot, 1.2f);
  reallocate<double>(&y_recv_buf, &y_recv_buf_len, 3*y_recv_ncoord_tot, 1.2f);
  copy_HtoD<int>(h_y_recv_atomind_in, y_recv_atomind, y_recv_ncoord_tot);

  reallocate<int>(&y_send_atomind, &y_send_atomind_len, y_send_ncoord_tot, 1.2f);
  reallocate<double>(&y_send_buf, &y_send_buf_len, 3*y_send_ncoord_tot, 1.2f);
  copy_HtoD<int>(h_y_send_atomind_in, y_send_atomind, y_send_ncoord_tot);
}

//
// Communicate forces
//
void DomdecD2DComm::transfer_force(int stride, double *force,
				   cudaStream_t stream) {
  int nthread;
  int nblock;

  nthread = 512;
  nblock = (y_recv_ncoord_tot - 1)/nthread + 1;
  pack_forces_kernel<<< nblock, nthread, 0, stream >>>
    (y_recv_ncoord_tot, y_recv_atomind, force, stride, xyz_tmp, y_recv_buf);

  // Now forces are packed into y_recv_buf on the device
  
  // Copy y_recv_buf to the host
  copy_DtoH<double>(y_recv_buf, h_y_recv_buf, 3*y_recv_ncoord_tot);
  
  // Send and receive forces in y direction
  ydir.clear_req();
  for (int i=0;i < ny_comm;i++) {
    if (y_send_ncoord[i] > 0) {
      ydir.irecv<double>(&y_send_buf[3*y_send_pos[i]], 3*y_send_ncoord[i],
			      y_send_node[i]);
    }
    if (y_recv_ncoord(2,i) > 0) {
      ydir.isend<double>(&y_recv_buf[3*y_recv_pos[i]], 3*y_recv_ncoord[i],
			      y_recv_node[i]);
    }
  }

  // Wait for forces from -y direction
  ydir.waitall();

  // Put forces from -y direction into a temporary array
  nthread = 512;
  nblock = (y_recv_ncoord_tot - 1)/nthread + 1;
  unpack_forces_kernel<<< nblock, nthread, 0, stream >>>
    (y_send_ncoord_tot, y_send_atomind, y_send_buf, stride, xyz_tmp);


  // Put forces from -y direction into correct arrays
  nthread = 512;
  nblock = (y_recv_ncoord_tot - 1)/nthread + 1;
  unpack_forces_kernel<<< nblock, nthread, 0, stream >>>
    (y_send_ncoord_tot, y_send_atomind, y_send_buf, stride, force);

}
