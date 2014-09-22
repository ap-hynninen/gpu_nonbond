#undef SEEK_SET
#undef SEEK_CUR
#undef SEEK_END
#include <mpi.h>
#include <iostream>
#include "mpi_utils.h"
#include "cuda_utils.h"
#include "CudaMPI.h"

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

int CudaMPI::Send(void *buf, int count, int dest, int tag, void *h_buf) {
  if (CudaAware) {
    return MPI_Send(buf, count, MPI_BYTE, dest, tag, comm);
  } else {
    copy_DtoH<char>((char *)buf, (char *)h_buf, count);
    return MPI_Send(h_buf, count, MPI_BYTE, dest, tag, comm);
  }
}

int CudaMPI::Recv(void *buf, int count, int source, int tag, MPI_Status *status, void *h_buf) {
  if (CudaAware) {
    return MPI_Recv(buf, count, MPI_BYTE, source, tag, comm, status);
  } else {
    int retval = MPI_Recv(h_buf, count, MPI_BYTE, source, tag, comm, status);
    copy_HtoD<char>((char *)h_buf, (char *)buf, count);
    return retval;
  }
}
