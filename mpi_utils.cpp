#include <mpi.h>
#include <iostream>
#include "mpi_utils.h"

void start_mpi(int argc, char *argv[], int &numnode, int &mynode) {

  MPICheck(MPI_Init(&argc, &argv));
  MPICheck(MPI_Comm_size(MPI_COMM_WORLD, &numnode));
  MPICheck(MPI_Comm_rank(MPI_COMM_WORLD, &mynode));

  if (mynode == 0) std::cout << "MPI started, numnode = " << numnode << std::endl;

}

void stop_mpi() {
  MPICheck(MPI_Finalize());
}

//
// Returns the local rank set by environment variable
// Returns -1 if no local rank found
//
int get_env_local_rank() {
  char *localRankStr = NULL;
  int rank;

  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
    // OpenMPI found
    rank = atoi(localRankStr);
  } else if ((localRankStr = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
    // MVAPICH found
    rank = atoi(localRankStr);
  } else {
    rank = -1;
  }

  return rank;
}
