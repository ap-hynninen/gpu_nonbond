#include <mpi.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cstdlib>
#include "CpuMultiNodeMatrix3d.h"
#include "mpi_utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif

template<typename T>
bool test_correctness(const int NX, const int NY, const int NZ,
		      const int tiledim, const int ny, const int nz);
template<typename T>
void test_performance(const int N, const int TILEDIM, const int ny, const int nz);

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

  int N = 64;

  //CpuMatrix3d<float> q(N, N, N, "test_data/q_real_double.txt");
  //CpuMatrix3d<float> q_t(N, N, N);
  //q.transpose_yzx_ref(q_t);

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

  /*
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
    mat.setup_transpose_yzx(&mat_t);
    mat_t.setup_transpose_yzx(&mat);
    
    const int nrep = 20000;
    MPICheck(MPI_Barrier( MPI_COMM_WORLD));
    double begin = MPI_Wtime();
    for (int i=0;i < nrep;i++) {
      mat.transpose_yzx(&mat_t);
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
    
    mat.transpose_yzx(&mat_t);
    mat_comp = mat_t.compare(&q_t, 0.0, max_diff);
    if (!mat_comp) {
      std::cout << "mat_t vs. q_t comparison FAILED" << std::endl;
    } else {
      if (mynode == 0) std::cout << "mat_t vs. q_t comparison OK" << std::endl;
    }

  }
  */

  /*
  {
    CpuMultiNodeMatrix3d<float> mat(N, N, N, 1, ny, nz, mynode, 64, "test_data/q_real_double.txt");
    CpuMultiNodeMatrix3d<float> mat_t(N, N, N, 1, ny, nz, mynode);

    double max_diff;
    bool mat_comp = mat.compare(q, 0.0, max_diff);
    if (!mat_comp) {
      std::cout << "mat vs. q comparison FAILED" << std::endl;
    } else {
      if (mynode == 0) std::cout << "mat vs. q comparison OK" << std::endl;
    }
    
    // Setup transposes
    mat.setup_transpose_yzx(mat_t);
    mat_t.setup_transpose_yzx(mat);
    
    const int nrep = 20000;
    MPICheck(MPI_Barrier( MPI_COMM_WORLD));
    double begin = MPI_Wtime();
    for (int i=0;i < nrep;i++) {
      mat.transpose_yzx(mat_t);
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
    
    mat.transpose_yzx(mat_t);
    mat_comp = mat_t.compare(q_t, 0.0, max_diff);
    if (!mat_comp) {
      std::cout << "mat_t vs. q_t comparison FAILED" << std::endl;
    } else {
      if (mynode == 0) std::cout << "mat_t vs. q_t comparison OK" << std::endl;
    }

  }
  */

  // ------------------------------------------------------------
  // Test correctness
  // ------------------------------------------------------------
  srand(time(NULL));
  for (int i=0;i < 20;i++) {
    // Take random matrix size
    int NX = rand() % 256 + 4;
    int NY = rand() % 256 + 4;
    int NZ = rand() % 256 + 4;
    int tiledim = rand() % 256 + 4;
    //NX = 64;
    //NY = 64;
    //NZ = 64;
    //tiledim = 64;
    if (mynode == 0) {
      std::cout << "NX = " << NX << " NY = " << NY << " NZ = " << NZ 
		<< " tiledim = " << tiledim << " ...";
    }
    if (test_correctness<double>(NX, NY, NZ, tiledim, ny, nz)) {
      if (mynode == 0) std::cout << "OK" << std::endl;
    } else {
      if (mynode == 0) std::cout << "FAILED" << std::endl;
      return 1;
    }
  }

  /*
  // ------------------------------------------------------------
  // Test performance
  // ------------------------------------------------------------
  for (N=64;N <= 512;N*=2) {
    for (int tiledim=32;tiledim <= N;tiledim*=2) {
      test_performance<float>(N, tiledim, ny, nz);
    }
  }
  // ------------------------------------------------------------
  */

  stop_mpi();

  return 0;
}

//
// Test correctness
//
template<typename T>
bool test_correctness(const int NX, const int NY, const int NZ,
		      const int tiledim, const int ny, const int nz) {

  CpuMultiNodeMatrix3d<T> mat_xyz(NX, NY, NZ, 1, ny, nz, mynode, tiledim);
  for (int z=0;z < NZ;z++) {
    for (int y=0;y < NY;y++) {
      for (int x=0;x < NX;x++) {
	mat_xyz.setData(x, y, z, (T)(x + y*NX + z*NX*NY));
      }
    }
  }

  CpuMultiNodeMatrix3d<T> mat_yzx(NY, NZ, NX, 1, ny, nz, mynode, tiledim);
  mat_xyz.setup_transpose_yzx(mat_yzx);
  mat_xyz.transpose_yzx(mat_yzx);
  for (int x=0;x < NX;x++) {
    for (int z=0;z < NZ;z++) {
      for (int y=0;y < NY;y++) {
	if (mat_yzx.hasData(y,z,x)) {
	  if (mat_yzx.getData(y,z,x) != (T)(x + y*NX + z*NX*NY)) {
	    std::cout << std::endl << "YZX: x = " << x << " y = " << y << " z = " << z << std::endl;
	    return false;
	  }
	}
      }
    }
  }

  CpuMultiNodeMatrix3d<T> mat_zxy(NZ, NX, NY, 1, ny, nz, mynode, tiledim);
  mat_xyz.setup_transpose_zxy(mat_zxy);
  mat_xyz.transpose_zxy(mat_zxy);
  for (int y=0;y < NY;y++) {
    for (int x=0;x < NX;x++) {
      for (int z=0;z < NZ;z++) {
	if (mat_zxy.hasData(z,x,y)) {
	  if (mat_zxy.getData(z,x,y) != (T)(x + y*NX + z*NX*NY)) {
	    std::cout << std::endl << "ZXY: x = " << x << " y = " << y << " z = " << z << std::endl;
	    return false;
	  }
	}
      }
    }
  }

  return true;
}

//
// Test performance
//
template<typename T>
void test_performance(const int N, const int TILEDIM, const int ny, const int nz) {

  CpuMultiNodeMatrix3d<T> mat(N, N, N, 1, ny, nz, mynode, TILEDIM);
  CpuMultiNodeMatrix3d<T> mat_t(N, N, N, 1, ny, nz, mynode, TILEDIM);

  // Setup transposes
  mat.setup_transpose_yzx(mat_t);

  double begin, end, time_yzx, time_zxy;
    
  const int nrep = 1000000/((N/64)*(N/64)*(N/64));
  MPICheck(MPI_Barrier( MPI_COMM_WORLD));
  begin = MPI_Wtime();
  for (int i=0;i < nrep;i++) {
    mat.CpuMatrix3d<T>::transpose_yzx(mat_t);
  }
  MPICheck(MPI_Barrier( MPI_COMM_WORLD));
  end = MPI_Wtime();
  time_yzx = end - begin;

  mat.setup_transpose_zxy(mat_t);

  MPICheck(MPI_Barrier( MPI_COMM_WORLD));
  begin = MPI_Wtime();
  for (int i=0;i < nrep;i++) {
    mat.CpuMatrix3d<T>::transpose_zxy(mat_t);
  }
  MPICheck(MPI_Barrier( MPI_COMM_WORLD));
  end = MPI_Wtime();
  time_zxy = end - begin;
    
  if (mynode == 0) {
    std::cout << "N = " << N << " TILEDIM = " << TILEDIM << std::endl;
    std::cout << "nrep = " << nrep << std::endl;
    std::cout << "time(s) = " << time_yzx << " " << time_zxy
	      << " per transpose (us) = "
	      << time_yzx*1.0e6/(double)(nrep*2) << " "
	      << time_zxy*1.0e6/(double)(nrep*2) << std::endl;
  }

}
