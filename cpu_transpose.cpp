#include <mpi.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include "CpuMultiNodeMatrix3d.h"
#include "mpi_utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif

template<typename T>
void test(const int N, const int TILEDIM, const int nthread, const int ny, const int nz);

int numnode;
int mynode;

int main(int argc, char *argv[]) {

  start_mpi(argc, argv, numnode, mynode);
  //  time_transpose();

  int nthread = 1;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
  {
    nthread = omp_get_num_threads();
    if (mynode == 0) {
      std::cout << "number of OpenMP threads = " << nthread << std::endl;
    }
  }
#endif

  float** buf_th = new float*[nthread];
  for (int i=0;i < nthread;i++) {
    buf_th[i] = new float[64*64];
  }

  int N = 64;

  CpuMatrix3d<float> q(N, N, N, "test_data/q_real_double.txt");
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

  if (mynode == 0) {
    std::cout << "ny = " << ny << " nz = " << nz << std::endl;
  }

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
    
    const int nrep = 20000;
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
    
    const int nrep = 20000;
    MPICheck(MPI_Barrier( MPI_COMM_WORLD));
    double begin = MPI_Wtime();
    for (int i=0;i < nrep;i++) {
      mat.transpose_xyz_yzx_tiled(&mat_t, 64, buf_th);
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
    
    mat.transpose_xyz_yzx_tiled(&mat_t, 64, buf_th);
    mat_comp = mat_t.compare(&q_t, 0.0, max_diff);
    if (!mat_comp) {
      std::cout << "mat_t vs. q_t comparison FAILED" << std::endl;
    } else {
      if (mynode == 0) std::cout << "mat_t vs. q_t comparison OK" << std::endl;
    }

  }

  for (int i=0;i < nthread;i++) {
    delete [] buf_th[i];
  }
  delete [] buf_th;

  for (N=64;N <= 512;N*=2) {
    for (int tiledim=32;tiledim <= N;tiledim*=2) {
      test<float>(N, tiledim, nthread, ny, nz);
    }
  }

  stop_mpi();

  return 0;
}

template<typename T>
void test(const int N, const int TILEDIM, const int nthread, const int ny, const int nz) {

  CpuMultiNodeMatrix3d<T> mat(N, N, N, 1, ny, nz, mynode);
  CpuMultiNodeMatrix3d<T> mat_t(N, N, N, 1, ny, nz, mynode);

  T** buf_th = new T*[nthread];
  for (int i=0;i < nthread;i++) {
    buf_th[i] = new T[TILEDIM*TILEDIM];
  }

  // Setup transposes
  mat.setup_transpose_xyz_yzx(&mat_t);
  mat_t.setup_transpose_xyz_yzx(&mat);
    
  const int nrep = 1000000/((N/64)*(N/64)*(N/64));
  MPICheck(MPI_Barrier( MPI_COMM_WORLD));
  double begin = MPI_Wtime();
  for (int i=0;i < nrep;i++) {
    mat.transpose_xyz_yzx_tiled(&mat_t, TILEDIM, buf_th);
  }
  MPICheck(MPI_Barrier( MPI_COMM_WORLD));
  double end = MPI_Wtime();
    
  double time_spent = end - begin;
    
  if (mynode == 0) {
    std::cout << "N = " << N << " TILEDIM = " << TILEDIM << std::endl;
    std::cout << "nrep = " << nrep << std::endl;
    std::cout << "time_spent (sec) = " << time_spent
	      << " per transpose (micro sec) = "
	      << time_spent*1.0e6/(double)(nrep*2) << std::endl;
  }

  for (int i=0;i < nthread;i++) {
    delete [] buf_th[i];
  }
  delete [] buf_th;

}
