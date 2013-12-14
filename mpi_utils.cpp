#ifdef USE_MPI

#include <mpi.h>
#include <iostream>
#include "mpi_utils.h"

void start_mpi(int argc, char *argv[], int &numnode, int &mynode) {

  MPICheck(MPI_Init(&argc, &argv));
  MPICheck(MPI_Comm_size(MPI_COMM_WORLD, &numnode));
  MPICheck(MPI_Comm_rank(MPI_COMM_WORLD, &mynode));

  if (mynode == 0) std::cout << "numnode = " << numnode << std::endl;
  std::cout << "mynode = " << mynode << std::endl;
}

void stop_mpi() {
  MPICheck(MPI_Finalize());
}

#endif // USE_MPI
