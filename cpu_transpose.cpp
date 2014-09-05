#include <mpi.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include "CpuMultiNodeMatrix3d.h"
#include "mpi_utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int numnode;
int mynode;

int main(int argc, char *argv[]) {

  start_mpi(argc, argv, numnode, mynode);
  //  time_transpose();

#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
  {
    if (mynode == 0) {
      std::cout << "number of OpenMP threads = " << omp_get_num_threads() << std::endl;
    }
  }
#endif

  const int N = 64;

  CpuMatrix3d<float> q(N, N, N, "test_data/q_real_double.txt");
  //CpuMatrix3d<float> q(N, N, N);
  CpuMatrix3d<float> q_t(N, N, N);
  q.transpose_xyz_yzx_ref(&q_t);

  int ny = (int)ceil(sqrt(numnode*(double)N/(double)N));
  int nz = numnode/ny;
  while (ny*nz != numnode) {
    ny--;
    nz = numnode/ny;
  }
  if (ny == 0 || nz == 0) {
    std::cerr << "ny or nz is zero" << std::endl;
    exit(1);
  }

  if (mynode == 0)
    std::cout << "ny = " << ny << " nz = " << nz << std::endl;

  {
    CpuMultiNodeMatrix3d<float> mat(N, N, N, 1, ny, nz, mynode, "test_data/q_real_double.txt");
    //CpuMultiNodeMatrix3d<float> mat(N, N, N, 1, ny, nz, mynode);
    CpuMultiNodeMatrix3d<float> mat_t(N, N, N, 1, ny, nz, mynode);

    double max_diff;
    bool mat_comp = mat.compare(&q, 0.0, max_diff);
    if (!mat_comp) {
      std::cout << "mat vs. q comparison FAILED" << std::endl;
    } else {
      if (mynode == 0) std::cout << "mat vs. q comparison OK" << std::endl;
    }
    
    // Setup transposes
    mat.setup_transpose_xyz_yzx(&mat_t);
    mat_t.setup_transpose_xyz_yzx(&mat);
    
    const int nrep = 200000;
    MPICheck(MPI_Barrier( MPI_COMM_WORLD));
    double begin = MPI_Wtime();
    for (int i=0;i < nrep;i++) {
      mat.transpose_xyz_yzx(&mat_t);
    }
    MPICheck(MPI_Barrier( MPI_COMM_WORLD));
    double end = MPI_Wtime();
    
    double time_spent = end - begin;
    
    if (mynode == 0) {
      std::cout << "nrep = " << nrep << std::endl;
      std::cout << "time_spent (sec) = " << time_spent
		<< " per transpose (micro sec) = "
		<< time_spent*1.0e6/(double)(nrep*2) << std::endl;
    }
    
    mat.transpose_xyz_yzx(&mat_t);
    mat_comp = mat_t.compare(&q_t, 0.0, max_diff);
    if (!mat_comp) {
      std::cout << "mat_t vs. q_t comparison FAILED" << std::endl;
    } else {
      if (mynode == 0) std::cout << "mat_t vs. q_t comparison OK" << std::endl;
    }

  }

  stop_mpi();

  return 0;
}

