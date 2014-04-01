#include <iostream>
#include <cassert>
#include <mpi.h>
#include "DomdecMPI.h"

//
// Class creator
//
DomdecMPI::DomdecMPI(MPI_Comm comm, int tag, int nreq) {
  this->comm = comm;
  this->tag = tag;
  this->nreq = nreq;
  req = new MPI_Request[nreq];
  ireq = 0;
}

//
// Class destructor
//
DomdecMPI::~DomdecMPI() {
  delete [] req;
}

//
// Clear requests
//
void DomdecMPI::clear_req() {
  ireq = 0;
}

//
// Send
//
template<typename T>
void DomdecMPI::isend(T *buf, int count, int dest) {
  if (ireq+1 >= nreq) {
    std::cerr << "DomdecMPI::isend request overflow" << std::endl;
    exit(1);
  }
  int ierror = MPI_Isend((void *)buf, count*sizeof(T), MPI_BYTE,
			 dest, tag, comm, req[ireq++]);
  if (ierror != MPI_SUCCESS) {
    std::cerr << "DomdecMPI::isend MPI_Isend failed" << std::endl;
    exit(1);
  }
}

//
// Receive
//
template<typename T>
void DomdecMPI::irecv(T *buf, int count, int src) {
  if (ireq+1 >= nreq) {
    std::cerr << "DomdecMPI::irecv request overflow" << std::endl;
    exit(1);
  }
  int ierror = MPI_Irecv((void *)buf, count*sizeof(T), MPI_BYTE,
			 src, tag, comm, req[ireq++]);
  if (ierror != MPI_SUCCESS) {
    std::cerr << "DomdecMPI::irecv MPI_Irecv failed" << std::endl;
    exit(1);
  }
}

//
// Wait all
//
void DomdecMPI::waitall() {
  if (nreq > 0) {
    int ierror = MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    if (ierror != MPI_SUCCESS) {
      std::cerr << "DomdecMPI::waitall MPI_Waitall failed" << std::endl;
      exit(1);
    }
  }
}


//
// Explicit instances
//
template class DomdecMPI::isend<double>(double *buf, int count, int dest);
template class DomdecMPI::irecv<double>(double *buf, int count, int dest);
