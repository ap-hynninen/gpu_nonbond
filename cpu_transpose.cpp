#include <iostream>
#include <math.h>
#include <time.h>
#include "mpi_utils.h"
#include "CpuMultiNodeMatrix3d.h"

int numnode;
int mynode;

int main(int argc, char *argv[]) {

  start_mpi(argc, argv, numnode, mynode);
  //  time_transpose();

  CpuMatrix3d<float> q(64, 64, 64, "test_data/q_real_double.txt");
  CpuMatrix3d<float> q_t(64, 64, 64);
  q.transpose_xyz_yzx_ref(&q_t);

  int ny = (int)ceil(sqrt(numnode*64.0/64.0));
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

  CpuMultiNodeMatrix3d<float> mat(64, 64, 64, 1, ny, nz, mynode, "test_data/q_real_double.txt");
  CpuMultiNodeMatrix3d<float> mat_t(64, 64, 64, 1, ny, nz, mynode);

  double max_diff;
  bool mat_comp = mat.compare(&q, 0.0, max_diff);
  if (!mat_comp) {
    std::cout << "mat vs. q comparison FAILED" << std::endl;
  } else {
    if (mynode == 0) std::cout << "mat vs. q comparison OK" << std::endl;
  }

  MPICheck(MPI_Barrier( MPI_COMM_WORLD));

  clock_t begin = clock();
  for (int i=0;i < 5000;i++) {
    MPICheck(MPI_Barrier( MPI_COMM_WORLD));
    mat.setup_transpose_xyz_yzx(&mat_t);
    MPICheck(MPI_Barrier( MPI_COMM_WORLD));
    mat_t.setup_transpose_xyz_yzx(&mat);
  }
  clock_t end = clock();

  double time_spent = (double)(end - begin)*1000.0 / CLOCKS_PER_SEC;

  std::cout << "time_spent (ms) = " << time_spent << std::endl;

  mat.transpose_xyz_yzx();
  mat_comp = mat_t.compare(&q_t, 0.0, max_diff);
  if (!mat_comp) {
    std::cout << "mat_t vs. q_t comparison FAILED" << std::endl;
  } else {
    if (mynode == 0) std::cout << "mat_t vs. q_t comparison OK" << std::endl;
  }

  stop_mpi();

  return 0;
}

