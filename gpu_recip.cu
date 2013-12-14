#include <iostream>
#include <cuda.h>
#include "cuda_utils.h"
#include "XYZQ.h"
#include "Bspline.h"
#include "Grid.h"
#include "Force.h"
#ifdef USE_MPI
#include "mpi_utils.h"
#endif
#include "MultiNodeMatrix3d.h"

void test();

int main(int argc, char *argv[]) {

  int numnode = 1;
  int mynode = 0;

#ifdef USE_MPI
  start_mpi(argc, argv, numnode, mynode);
#endif

  start_gpu(numnode, mynode);

  //  time_transpose();

  MultiNodeMatrix3d<float> mat0(64, 64, 64, 1, 2, 2, 0);
  MultiNodeMatrix3d<float> mat1(64, 64, 64, 1, 2, 2, 1);
  MultiNodeMatrix3d<float> mat2(64, 64, 64, 1, 2, 2, 2);
  MultiNodeMatrix3d<float> mat3(64, 64, 64, 1, 2, 2, 3);


  MultiNodeMatrix3d<float> mat0_t(64, 64, 64, 1, 2, 2, 0);
  MultiNodeMatrix3d<float> mat1_t(64, 64, 64, 1, 2, 2, 1);
  MultiNodeMatrix3d<float> mat2_t(64, 64, 64, 1, 2, 2, 2);
  MultiNodeMatrix3d<float> mat3_t(64, 64, 64, 1, 2, 2, 3);

  mat0.print_info();

  //  mat1.print_info();
  //  mat2.print_info();
  //  mat3.print_info();


  mat0.transpose_xyz_yzx(&mat0_t);
  //  mat1.transpose_xyz_yzx(&mat1_t);
  //  mat2.transpose_xyz_yzx(&mat2_t);
  //  mat3.transpose_xyz_yzx(&mat3_t);

  //  test();

#ifdef USE_MPI
  stop_mpi();
#endif

  return 0;
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
  const FFTtype fft_type = BOX;

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

  Force<float> force_comp("test_data/force.txt");
  Force<float> force(ncoord);

  // Load coordinates
  XYZQ xyzq("test_data/xyzq.txt");

  // Create Bspline and Grid objects
  Bspline<float> bspline(ncoord, order, nfftx, nffty, nfftz);
  Grid<long long int, float, float2> grid(nfftx, nffty, nfftz, order, fft_type, nnode, mynode);

  double tol = 1.0e-5;
  double max_diff;

  bspline.set_recip<double>(recip);
  bspline.calc_prefac();

  grid.print_info();

  bspline.fill_bspline(xyzq.xyzq, xyzq.ncoord);

  // Warm up
  /*
  grid.spread_charge(xyzq.ncoord, bspline);
  grid.r2c_fft();
  grid.scalar_sum(recip, kappa, bspline.prefac_x, bspline.prefac_y, bspline.prefac_z);
  grid.c2r_fft();
  grid.gather_force(ncoord, recip, bspline, force.stride, force.data);
  */

  // Run
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
  if (fft_type == BOX) {
    Matrix3d<float2> q_zfft_t(nfftx/2+1, nffty, nfftz);
    q_zfft.transpose_xyz_yzx(&q_zfft_t);
    if (!q_zfft_t.compare(grid.fft_grid, tol, max_diff)) {
      grid.fft_grid->print(0,32,0,0,0,0);
      std::cout<<"--------------------------------------------------"<<std::endl;
      q_zfft_t.print(0,32,0,0,0,0);
      std::cout<< "q_zfft_t comparison FAILED" << std::endl;
      return;
    } else {
      std::cout<< "q_zfft_t comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
    }
  } else {
    if (!q_zfft.compare(grid.zfft_grid, tol, max_diff)) {
      std::cout<< "q_zfft comparison FAILED" << std::endl;
      return;
    } else {
      std::cout<< "q_zfft comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
    }
  }

  return;

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
  grid.solved_grid->scale(1.0f/(float)(nfftx*nffty*nfftz));
  if (!q.compare(grid.solved_grid, tol, max_diff)) {
    std::cout<< "q comparison FAILED" << std::endl;
    return;
  } else {
    std::cout<< "q comparison OK (tolerance " << tol << " max difference "<< max_diff << ")" << std::endl;
  }
  return;
  */

  /*
  if (!q_solved.compare(grid.solved_grid, tol, max_diff)) {
    std::cout<< "q_solved comparison FAILED" << std::endl;
    return;
  } else {
    std::cout<< "q_solved comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }
  */

  // Calculate forces
  grid.gather_force(ncoord, recip, bspline, force.stride, force.data);

  tol = 3.2e-4;
  if (!force_comp.compare(&force, tol, max_diff)) {
  } else {
    std::cout<< "force comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

}
