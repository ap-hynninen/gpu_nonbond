#include "mpi_utils.h"
#include <iostream>

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

//
// Returns the local size set by environment variable
// Returns -1 if no local size found
//
int get_env_local_size() {
  char *localRankStr = NULL;
  int size;

  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv("OMPI_COMM_WORLD_LOCAL_SIZE")) != NULL) {
    // OpenMPI found
    size = atoi(localRankStr);
  } else if ((localRankStr = getenv("MV2_COMM_WORLD_LOCAL_SIZE")) != NULL) {
    // MVAPICH found
    size = atoi(localRankStr);
  } else {
    size = -1;
  }

  return size;
}




//
// Concatenate list of integers among all nodes and place the the result in root
//
void MPI_Concatenate(int* sendbuf, int nsend, int* recvbuf, int root, MPI_Comm comm) {
  // Get number of nodes
  int numnode;
  MPICheck(MPI_Comm_size(comm, &numnode));

  int* nrecv = new int[numnode];
  int* precv = new int[numnode];

  // Send the number of entries to root
  MPICheck(MPI_Gather(&nsend, 1, MPI_INT, nrecv, 1, MPI_INT, root, comm));

  // Calculate position where to store the result
  precv[0] = 0;
  for (int i=1;i < numnode;i++) precv[i] = precv[i-1] + nrecv[i-1];

  MPICheck(MPI_Gatherv(sendbuf, nsend, MPI_INT, recvbuf, nrecv, precv, MPI_INT, root, comm));

  delete [] nrecv;
  delete [] precv;
}
