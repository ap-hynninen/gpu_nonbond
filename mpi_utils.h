#include <mpi.h>
#include <stdlib.h>

#define MPICheck(stmt) do {						\
    int err = stmt;							\
    if (err != MPI_SUCCESS) {						\
      printf("Error running %s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
      exit(1);								\
    }									\
  } while(0)

void start_mpi(int argc, char *argv[], int &numnode, int &mynode);
void stop_mpi();
int get_env_local_rank();
int get_env_local_size();

void MPI_Concat(int* sendbuf, int nsend, int* recvbuf, int root, MPI_Comm comm);

void MPI_Allconcat_T(void* sendbuf, int nsend, void* recvbuf, MPI_Comm comm, int sizeofT);

#ifdef __cplusplus
template<class T>
void MPI_Allconcat(T* sendbuf, int nsend, T* recvbuf, MPI_Comm comm) {
  MPI_Allconcat_T(sendbuf, nsend, recvbuf, comm, sizeof(T));
}
#endif
