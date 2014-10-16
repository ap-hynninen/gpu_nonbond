#include <iostream>
#include "mpi_utils.h"
#include "cuda_utils.h"
#include "CudaMPI.h"

/*
int CudaMPI::Isend(void *buf, int count, int dest, int tag, MPI_Request *request, void *h_buf) {
  if (CudaAware) {
    return MPI_Isend(buf, count, MPI_BYTE, dest, tag, comm, request);
  } else {
    copy_DtoH<char>((char *)buf, (char *)h_buf, count);
    return MPI_Isend(h_buf, count, MPI_BYTE, dest, tag, comm, request);
  }
}

int CudaMPI::Irecv(void *buf, int count, int source, int tag, MPI_Request *request, void *h_buf) {
  if (CudaAware) {
    return MPI_Irecv(buf, count, MPI_BYTE, source, tag, comm, request);
  } else {
    int retval = MPI_Irecv(h_buf, count, MPI_BYTE, source, tag, comm, request);
    copy_HtoD<char>((char *)h_buf, (char *)buf, count);
    return retval;
  }
}

int CudaMPI::Waitall(int nrequest, MPI_Request *request) {
  int retval = MPI_Waitall(nrequest, request, MPI_STATUSES_IGNORE);
  if (!CudaAware) {

  }
  retval;
}
*/

//
// NOTE: Seems like cuda aware MPI calls require MPI_Barrier() before sending/receiving.
// This was noticed under OpenMPI 1.8.1, gcc 4.8.3, cuda 6.5
//

int CudaMPI::Sendrecv(void *sendbuf, int sendcount, int dest, int sendtag,
		      void *recvbuf, int recvcount, int source, int recvtag, MPI_Status *status,
		      void *h_sendbuf, void *h_recvbuf) {
  int retval;

  if (CudaAware) {
    gpu_range_start("MPI_Sendrecv (CudaAware)");
    MPI_Barrier(comm);
    retval = MPI_Sendrecv(sendbuf, sendcount, MPI_BYTE, dest, sendtag,
			  recvbuf, recvcount, MPI_BYTE, source, recvtag, comm, status);
    gpu_range_stop();
  } else {
    copy_DtoH_sync<char>((char *)sendbuf, (char *)h_sendbuf, sendcount);
    gpu_range_start("MPI_Sendrecv");
    retval= MPI_Sendrecv(h_sendbuf, sendcount, MPI_BYTE, dest, sendtag,
			 h_recvbuf, recvcount, MPI_BYTE, source, recvtag, comm, status);
    gpu_range_stop();
    copy_HtoD_sync<char>((char *)h_recvbuf, (char *)recvbuf, recvcount);
  }

  return retval;
}

int CudaMPI::Send(void *buf, int count, int dest, int tag, void *h_buf) {
  int retval;

  if (CudaAware) {
    gpu_range_start("MPI_Send (CudaAware)");
    MPI_Barrier(comm);
    retval = MPI_Send(buf, count, MPI_BYTE, dest, tag, comm);
    gpu_range_stop();
  } else {
    copy_DtoH_sync<char>((char *)buf, (char *)h_buf, count);
    gpu_range_start("MPI_Send");
    retval = MPI_Send(h_buf, count, MPI_BYTE, dest, tag, comm);
    gpu_range_stop();
  }

  return retval;
}

int CudaMPI::Recv(void *buf, int count, int source, int tag, MPI_Status *status, void *h_buf) {
  int retval;

  if (CudaAware) {
    gpu_range_start("MPI_Recv (CudaAware)");
    MPI_Barrier(comm);
    retval = MPI_Recv(buf, count, MPI_BYTE, source, tag, comm, status);
    gpu_range_stop();
  } else {
    gpu_range_start("MPI_Recv");
    retval = MPI_Recv(h_buf, count, MPI_BYTE, source, tag, comm, status);
    gpu_range_stop();
    copy_HtoD_sync<char>((char *)h_buf, (char *)buf, count);
  }

  return retval;
}
