
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
