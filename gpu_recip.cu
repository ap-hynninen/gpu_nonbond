#include <iostream>
#include <cuda.h>
#include <cufft.h>
#include "gpu_utils.h"
#include "XYZQ.h"
#include "Bspline.h"
#include "Grid.h"
#include "Force.h"

void time_transpose();
void test();

int main(int argc, char *argv[]) {

  int gpu_ind = 0;
  cudaCheck(cudaSetDevice(gpu_ind));

  cudaCheck(cudaThreadSynchronize());
  
  cudaDeviceProp gpu_prop;
  cudaCheck(cudaGetDeviceProperties(&gpu_prop, gpu_ind));

  int cuda_driver_version;
  cudaCheck(cudaDriverGetVersion(&cuda_driver_version));

  int cuda_rt_version;
  cudaCheck(cudaRuntimeGetVersion(&cuda_rt_version));

  printf("Using CUDA device (%d) %s\n",gpu_ind,gpu_prop.name);
  printf("Using CUDA driver version %d\n",cuda_driver_version);
  printf("Using CUDA runtime version %d\n",cuda_rt_version);

  //  time_transpose();

  test();

  return 0;
}

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

//
// Test the code using data in test_data/ -directory
//
void test() {

  // Settings for the data:
  const double boxx = 62.23;
  const double boxy = 62.23;
  const double boxz = 62.23;
  const double kappa = 0.320;
  const int ncoord = 23558;
  const int nfftx = 64;
  const int nffty = 64;
  const int nfftz = 64;
  const int order = 4;

  // Number of MPI nodes & current node index
  int nnode = 1;
  int mynode = 0;

  // Setup reciprocal vectors
  double recip[9];
  for (int i=0;i < 9;i++) recip[i] = 0;
  recip[0] = 1.0/boxx;
  recip[4] = 1.0/boxy;
  recip[8] = 1.0/boxz;

  // Load comparison data
  Matrix3d<float> q(nfftx, nffty, nfftz, "test_data/q_real_double.txt");
  Matrix3d<float2> q_xfft(nfftx/2+1, nffty, nfftz, "test_data/q_comp1_double.txt");
  Matrix3d<float2> q_zfft(nfftz, nfftx/2+1, nffty, "test_data/q_comp5_double.txt");
  Matrix3d<float2> q_zfft_summed(nfftz, nfftx/2+1, nffty, "test_data/q_comp6_double.txt");
  Matrix3d<float2> q_comp7(nfftz, nfftx/2+1, nffty, "test_data/q_comp7_double.txt");
  Matrix3d<float2> q_comp9(nffty, nfftz, nfftx/2+1, "test_data/q_comp9_double.txt");
  Matrix3d<float2> q_comp10(nfftx/2+1, nffty, nfftz, "test_data/q_comp10_double.txt");
  Matrix3d<float> q_solved(nfftx, nffty, nfftz, "test_data/q_real2_double.txt");

  Force<double> force_comp("test_data/force.txt");
  Force<double> force(ncoord);
  force.setzero();

  // Load coordinates
  XYZQ xyzq("test_data/xyzq.txt");

  // Create Bspline and Grid objects
  Bspline<float> bspline(ncoord, order, nfftx, nffty, nfftz);
  Grid<long long int, float, float2> grid(nfftx, nffty, nfftz, order, nnode, mynode);

  double tol = 1.0e-5;
  double max_diff;

  bspline.set_recip<double>(recip);
  bspline.calc_prefac();

  grid.print_info();

  bspline.fill_bspline(xyzq.xyzq, xyzq.ncoord);

  grid.spread_charge(xyzq.ncoord, bspline);

  /*
  if (!q.compare(grid.charge_grid, tol, max_diff)) {
    std::cout<< "q comparison FAILED" << std::endl;
    return;
  } else {
    std::cout<< "q comparison OK (tolerance " << tol << " max difference "<< max_diff << ")" << std::endl;
  }
  */

  tol = 0.002;
  grid.r2c_fft();

  /*
  if (!q_zfft.compare(grid.zfft_grid, tol, max_diff)) {
    std::cout<< "q_zfft comparison FAILED" << std::endl;
    return;
  } else {
    std::cout<< "q_zfft comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }
  */

  tol = 1.0e-6;
  grid.scalar_sum(recip, kappa, bspline.prefac_x, bspline.prefac_y, bspline.prefac_z);
  /*
  if (!q_zfft_summed.compare(grid.zfft_grid, tol, max_diff)) {
    std::cout<< "q_zfft_summed comparison FAILED" << std::endl;
    q_zfft_summed.print(0,10,0,0,0,0);
    std::cout<<"====================================="<<std::endl;
    grid.zfft_grid->print(0,10,0,0,0,0);
    return;
  } else {
    std::cout<< "q_zfft_summed comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }
  */

  /*
  tol = 1.0e-6;
  grid.z_fft_c2c(grid.zfft_grid->data, CUFFT_INVERSE);
  if (!q_comp7.compare(grid.zfft_grid, tol, max_diff)) {
    std::cout<< "q_comp7 comparison FAILED" << std::endl;
    return;
  } else {
    std::cout<< "q_comp7 comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

  tol = 3.0e-6;
  grid.zfft_grid->transpose_xyz_zxy(grid.yfft_grid);
  grid.y_fft_c2c(grid.yfft_grid->data, CUFFT_INVERSE);
  if (!q_comp9.compare(grid.yfft_grid, tol, max_diff)) {
    std::cout<< "q_comp9 comparison FAILED" << std::endl;
    return;
  } else {
    std::cout<< "q_comp9 comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

  tol = 3.0e-6;
  grid.yfft_grid->transpose_xyz_zxy(grid.xfft_grid);
  if (!q_comp10.compare(grid.xfft_grid, tol, max_diff)) {
    std::cout<< "q_comp10 comparison FAILED" << std::endl;
    return;
  } else {
    std::cout<< "q_comp10 comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }
  */

  tol = 1.0e-5;
  grid.c2r_fft();
  /*
  if (!q_solved.compare(grid.charge_grid, tol, max_diff)) {
    std::cout<< "q_solved comparison FAILED" << std::endl;
    return;
  } else {
    std::cout<< "q_solved comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }
  */

  // Calculate forces
  grid.gather_force(ncoord, recip, bspline, ncoord, (long long int *)force.data);

  tol = 3.2e-4;
  if (!force_comp.compare(&force, tol, max_diff)) {
  } else {
    std::cout<< "force comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

}
