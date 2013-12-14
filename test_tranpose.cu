#include <iostream>
#include <cuda.h>
#include <cufft.h>
#include "gpu_utils.h"

//
//
//
void time_transpose() {

  const int NUM_REP = 100;
  const int nfftx = 64;
  const int nffty = 64;
  const int nfftz = 64;
  Matrix3d<float> A(nfftx, nffty, nfftz, "test_data/q_real_double.txt");
  //  Matrix3d<float> A(nfftx, nffty, nfftz);
  Matrix3d<float> B(nfftx, nffty, nfftz);
  Matrix3d<float> C(nfftx, nffty, nfftz);

  cudaEvent_t start_event, stop_event;
  cudaCheck(cudaEventCreate(&start_event));
  cudaCheck(cudaEventCreate(&stop_event));
  float ms;
  double max_diff;

  // Copy
  A.copy(&B);
  cudaCheck(cudaEventRecord(start_event,0));
  for (int i=0;i < NUM_REP;i++)
    A.copy(&B);
  cudaCheck(cudaEventRecord(stop_event,0));
  cudaCheck(cudaEventSynchronize(stop_event));
  cudaCheck(cudaEventElapsedTime(&ms, start_event, stop_event));
  std::cout << "copy:" << std::endl;
  std::cout << "time (ms) = " << ms << std::endl;
  std::cout << "GB/s = " << 2*nfftx*nffty*nfftz*sizeof(float)*1e-6*NUM_REP/ms << std::endl;

  // Transpose (x,y,z) -> (y,z,x)
  A.transpose_xyz_yzx(&B);
  cudaCheck(cudaEventRecord(start_event,0));
  for (int i=0;i < NUM_REP;i++)
    A.transpose_xyz_yzx(&B);
  cudaCheck(cudaEventRecord(stop_event,0));
  cudaCheck(cudaEventSynchronize(stop_event));
  cudaCheck(cudaEventElapsedTime(&ms, start_event, stop_event));
  A.transpose_xyz_yzx_host(&C);
  if (!B.compare(&C, 0.0, max_diff)) {
    std::cout << "Error in transpose_xyz_yzx" << std::endl;
    return;
  }
  std::cout << "transpose_xyz_yzx:" << std::endl;
  std::cout << "time (ms) = " << ms << std::endl;
  std::cout << "GB/s = " << 2*nfftx*nffty*nfftz*sizeof(float)*1e-6*NUM_REP/ms << std::endl;

  // Transpose (x,y,z) -> (z,x,y)
  A.transpose_xyz_zxy(&B);
  cudaCheck(cudaEventRecord(start_event,0));
  for (int i=0;i < NUM_REP;i++)
    A.transpose_xyz_zxy(&B);
  cudaCheck(cudaEventRecord(stop_event,0));
  cudaCheck(cudaEventSynchronize(stop_event));
  cudaCheck(cudaEventElapsedTime(&ms, start_event, stop_event));
  A.transpose_xyz_zxy_host(&C);
  if (!B.compare(&C, 0.0, max_diff)) {
    std::cout << "Error in transpose_xyz_zxy" << std::endl;
    return;
  }
  std::cout << "transpose_xyz_zxy:" << std::endl;
  std::cout << "time (ms) = " << ms << std::endl;
  std::cout << "GB/s = " << 2*nfftx*nffty*nfftz*sizeof(float)*1e-6*NUM_REP/ms << std::endl;
}
