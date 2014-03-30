#ifndef DOMDECMPI_H
#define DOMDECMPI_H
#include <mpi.h>

class DomdecMPI {

private:

  int tag;

  int nreq;
  int ireq;

  MPI_Comm comm;
  MPI_Request *req;

public:

  DomdecMPI(MPI_Comm comm, int tag, int nreq);
  ~DomdecMPI();

  void clear_req();
  template<typename T> void isend(T *buf, int count, int dest);
  template<typename T> void irecv(T *buf, int count, int src);
  void waitall();

};

#endif // DOMDECMPI_H
