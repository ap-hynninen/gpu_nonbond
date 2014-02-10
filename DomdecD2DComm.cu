#include <iostream>
#include <cuda.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "DomdecD2DComm.h"

//
//
//
__global__ void pack_forces_kernel(const int n, const int *atomind, const int *atompos,
				   const double *force, const int stride,
				   double *buffer) {
  const int i = threadIdx.x + blockDim.x*blockIdx.x;

  if (i < n) {
    int j = atomind[i];
    int pos = atompos[ji];
    buf[pos]   = force[j];
    buf[pos+1] = force[j+stride];
    buf[pos+2] = force[j+stride*2];
  }
  
}

//
// Class creator
//
DomdecD2DComm::DomdecD2DComm() {

}

//
// Class destructor
//
DomdecD2DComm::~DomdecD2DComm() {

}

void DomdecD2DComm::transfer_force() {

  int nthread;
  int nblock;

  // Only z-direction implemented for now
  assert(nx_comm == 0);
  assert(ny_comm == 0);

  for (int i=0;i < nz_comm;i++) {
    nthread = 128;
    nblock = (z_recv_ncoord[i] - 1)/nthread + 1;
    pack_forces_kernel<<< nblock, nthread >>>
      (z_recv_ncoord[i], z_recv_atomind, z_recv_atompos, force, stride, buffer);
  }


  for (int i=0;i < nz_comm;i++) {
    if ((z_send_ncoord[i] > 0) && (z_recv_ncoord[i] > 0)) {
      gpu2gpu.sendrecv(z_recv_buffer[i], z_recv_ncoord[i], z_recv_node[i],
		       z_send_buffer[i], z_send_ncoord[i], z_send_node[i]);
    }
    if (z_recv_ncoord[i] > 0) {
    }
    if (z_send_ncoord[i] > 0) {
    }
  }

!!$    ! Communicate forces in z direction
!!$    do i=1,nz_comm
!!$       if (z_recv_ngroup(i) > 0 .and. z_send_ncoord(i) > 0) then
!!$          call mpi_sendrecv(z_recv_group(:,i), z_recv_ncoord(i), MPI_REAL8, &
!!$               z_recv_node(i), COORDBUF, &
!!$               z_send_group(:,i), z_send_ncoord(i), MPI_REAL8, z_send_node(i), &
!!$               COORDBUF, COMM_CHARMM, stat, ierror)
!!$          if (ierror /= MPI_SUCCESS) call WRNDIE(-5, &
!!$               '<DOMDEC_D2D_COMM>','ERROR IN MPI_SENDRECV IN Z DIRECTION')
!!$       elseif (z_recv_ngroup(i) > 0) then
!!$          call mpi_send(z_recv_group(:,i), z_recv_ncoord(i), MPI_REAL8, &
!!$               z_recv_node(i), COORDBUF, COMM_CHARMM, ierror)
!!$          if (ierror /= MPI_SUCCESS) call WRNDIE(-5, &
!!$               '<DOMDEC_D2D_COMM>','ERROR IN MPI_SEND IN Z DIRECTION')
!!$       elseif (z_send_ncoord(i) > 0) then
!!$          call mpi_recv(z_send_group(:,i), z_send_ncoord(i), MPI_REAL8, &
!!$               z_send_node(i), COORDBUF, COMM_CHARMM, stat, ierror)
!!$          if (ierror /= MPI_SUCCESS) call WRNDIE(-5, &
!!$               '<DOMDEC_D2D_COMM>','ERROR IN MPI_RECV IN Z DIRECTION')
!!$       endif
!!$    enddo


}
