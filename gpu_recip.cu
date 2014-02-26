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

void test4();
void test6();

int numnode = 1;
int mynode = 0;

int main(int argc, char *argv[]) {

#ifdef USE_MPI
  start_mpi(argc, argv, numnode, mynode);
#endif

  start_gpu(numnode, mynode);

  //  time_transpose();

  /*
  Matrix3d<float> q(64, 64, 64, "test_data/q_real_double.txt");
  Matrix3d<float> q_t(64, 64, 64);
  q.transpose_xyz_yzx(&q_t);

  MultiNodeMatrix3d<float> mat(64, 64, 64, 1, 1, 2, mynode, "test_data/q_real_double.txt");
  MultiNodeMatrix3d<float> mat_t(64, 64, 64, 1, 1, 2, mynode);

  double max_diff;
  bool mat_comp = mat.compare(&q, 0.0, max_diff);
  if (!mat_comp) {
    std::cout << "mat vs. q comparison FAILED" << std::endl;
  } else {
    if (mynode == 0) std::cout << "mat vs. q comparison OK" << std::endl;
  }

  mat.setup_transpose_xyz_yzx(&mat_t);

  mat.transpose_xyz_yzx();
  mat_comp = mat_t.compare(&q_t, 0.0, max_diff);
  if (!mat_comp) {
    std::cout << "mat_t vs. q_t comparison FAILED" << std::endl;
  } else {
    if (mynode == 0) std::cout << "mat_t vs. q_t comparison OK" << std::endl;
  }
  */
  
  test4();

  test6();

#ifdef USE_MPI
  stop_mpi();
#endif

  return 0;
}

//
// Test the code using data in test_data/ -directory
//
void test4() {

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
  const double energy_comp = 0.5939730270E+03;
  const double virial_comp[9] = {-0.5282341105E+03, 0.5596133994E+01, -0.6254453157E+01,
				 0.5596133994E+01, -0.5054581254E+03, -0.7652312605E-01,
				 -0.6254453157E+01, -0.7652312605E-01, -0.5575246246E+03};
  /*
0.1788697374E+01
0.1590726511E+01   -0.1685194095E-01    0.1883435446E-01    0.1522137390E+01    0.2322833086E-03    0.1678929560E+01
   */

  // Setup reciprocal vectors
  double recip[9];
  for (int i=0;i < 9;i++) recip[i] = 0;
  recip[0] = 1.0/boxx;
  recip[4] = 1.0/boxy;
  recip[8] = 1.0/boxz;

  // Load comparison data
  /*
  Matrix3d<float> q(nfftx, nffty, nfftz, "test_data/q_real_double.txt");
  Matrix3d<float2> q_xfft(nfftx/2+1, nffty, nfftz, "test_data/q_comp1_double.txt");
  Matrix3d<float2> q_zfft(nfftz, nfftx/2+1, nffty, "test_data/q_comp5_double.txt");
  Matrix3d<float2> q_zfft_summed(nfftz, nfftx/2+1, nffty, "test_data/q_comp6_double.txt");
  Matrix3d<float2> q_comp7(nfftz, nfftx/2+1, nffty, "test_data/q_comp7_double.txt");
  Matrix3d<float2> q_comp9(nffty, nfftz, nfftx/2+1, "test_data/q_comp9_double.txt");
  Matrix3d<float2> q_comp10(nfftx/2+1, nffty, nfftz, "test_data/q_comp10_double.txt");
  Matrix3d<float> q_solved(nfftx, nffty, nfftz, "test_data/q_real2_double.txt");
  */

  Force<float> force_comp("test_data/force_recip_4.txt");
  Force<float> force(ncoord);

  // Load coordinates
  XYZQ xyzq("test_data/xyzq.txt");

  // Create Bspline and Grid objects
  Bspline<float> bspline(ncoord, order, nfftx, nffty, nfftz);
  //Grid<long long int, float, float2> grid(nfftx, nffty, nfftz, order, fft_type, numnode, mynode);
  Grid<int, float, float2> grid(nfftx, nffty, nfftz, order, fft_type, numnode, mynode);

  double tol = 1.0e-5;
  double max_diff;

  bspline.set_recip<double>(recip);

  grid.print_info();

  bspline.fill_bspline(xyzq.xyzq, xyzq.ncoord);

  // Warm up
  //grid.spread_charge(xyzq.ncoord, bspline);
  grid.spread_charge(xyzq.xyzq, xyzq.ncoord, recip);
  grid.r2c_fft();
  grid.scalar_sum(recip, kappa, true, true);
  grid.c2r_fft();
  //grid.gather_force(ncoord, recip, bspline, force.stride, force.data);
  grid.gather_force(xyzq.xyzq, xyzq.ncoord, recip, force.stride, force.data);

  double energy, virial[9];
  grid.get_energy_virial(true, true, &energy, virial);
  tol = 1.2e-3;
  max_diff = fabs(energy_comp - energy);
  if (isnan(energy) || max_diff > tol) {
    std::cout<< "energy comparison FAILED" << std::endl;
    std::cout<< "energy_comp = " << energy_comp << std::endl;
    std::cout<< "energy      = " << energy << std::endl;
    return;
  } else {
    std::cout<< "energy comparison OK (tolerance " << tol << " max difference "
	     << max_diff << ")" << std::endl;
  }
  tol = 2.0e-3;
  max_diff = 0.0;
  bool vir_nan = false;
  for (int i=0;i < 9;i++) {
    max_diff = max(max_diff, fabs(virial_comp[i] - virial[i]));
    vir_nan = vir_nan | isnan(virial[i]);
  }
  if (max_diff > tol) {
    std::cout<< "virial comparison FAILED" << std::endl;
    std::cout<< "virial_comp | virial" << std::endl;
    for (int i=0;i < 9;i++) {
      std::cout << virial_comp[i] << " " << virial[i] << std::endl;
    }
    return;
  } else {
    std::cout<< "virial comparison OK (tolerance " << tol << " max difference "
	     << max_diff << ")" << std::endl;
  }


  // Run
  //grid.spread_charge(xyzq.ncoord, bspline);
  grid.spread_charge(xyzq.xyzq, xyzq.ncoord, recip);
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
  if (fft_type == BOX) {
    Matrix3d<float2> q_zfft_t(nfftx/2+1, nffty, nfftz);
    q_zfft.transpose_xyz_yzx(&q_zfft_t);
    if (!q_zfft_t.compare(grid.fft_grid, tol, max_diff)) {
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
  */

  tol = 1.0e-6;
  grid.scalar_sum(recip, kappa, false, false);
  /*
  if (fft_type == BOX) {
    Matrix3d<float2> q_zfft_summed_t(nfftx/2+1, nffty, nfftz);
    q_zfft_summed.transpose_xyz_yzx(&q_zfft_summed_t);
    if (!q_zfft_summed_t.compare(grid.fft_grid, tol, max_diff)) {
      std::cout<< "q_zfft_summed_t comparison FAILED" << std::endl;
      return;
    } else {
      std::cout<< "q_zfft_summed_t comparison OK (tolerance "<<tol<<" max difference "<<max_diff << ")" << std::endl;
    }
  } else {
    if (!q_zfft_summed.compare(grid.zfft_grid, tol, max_diff)) {
      std::cout<< "q_zfft_summed comparison FAILED" << std::endl;
      return;
    } else {
      std::cout<< "q_zfft_summed comparison OK (tolerance "<<tol<<" max difference "<<max_diff << ")" << std::endl;
    }
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
  //  grid.gather_force(ncoord, recip, bspline, force.stride, force.data);
  grid.gather_force(xyzq.xyzq, xyzq.ncoord, recip, force.stride, force.data);

  tol = 3.3e-4;
  if (!force_comp.compare(&force, tol, max_diff)) {
    std::cout<<"force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"force comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

}

//
// Test the code using data in test_data/ -directory
//
void test6() {

  // Settings for the data:
  const double boxx = 62.23;
  const double boxy = 62.23;
  const double boxz = 62.23;
  const double kappa = 0.320;
  const int ncoord = 23558;
  const int nfftx = 64;
  const int nffty = 64;
  const int nfftz = 64;
  const int order = 6;
  const FFTtype fft_type = BOX;
  const double energy_comp = 0.5951144996E+03;
  const double virial_comp[9] = {-0.5314776507E+03, 0.5670623182E+01, -0.6175456643E+01,
				 0.5670623182E+01, -0.5086150173E+03, -0.1026251692E+00,
				 -0.6175456643E+01, -0.1026251692E+00, -0.5606459148E+03};

  /*
 0.1792135021E+01
    0.1600494101E+01   -0.1707630375E-01    0.1859633108E-01    0.1531644169E+01    0.3107774643E-03    0.1688329264E+01
   */

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

  Force<float> force_comp("test_data/force_recip_6.txt");
  Force<float> force(ncoord);

  // Load coordinates
  XYZQ xyzq("test_data/xyzq.txt");

  // Create Bspline and Grid objects
  Grid<int, float, float2> grid(nfftx, nffty, nfftz, order, fft_type, numnode, mynode);

  double tol = 1.0e-5;
  double max_diff;

  grid.print_info();

  // Warm up
  //grid.spread_charge(xyzq.ncoord, bspline);
  grid.spread_charge(xyzq.xyzq, xyzq.ncoord, recip);
  grid.r2c_fft();
  grid.scalar_sum(recip, kappa, true, true);
  grid.c2r_fft();
  //grid.gather_force(ncoord, recip, bspline, force.stride, force.data);
  grid.gather_force(xyzq.xyzq, xyzq.ncoord, recip, force.stride, force.data);

  double energy, virial[9];
  grid.get_energy_virial(true, true, &energy, virial);
  tol = 1.3e-3;
  max_diff = fabs(energy_comp - energy);
  if (isnan(energy) || max_diff > tol) {
    std::cout<< "energy comparison FAILED" << std::endl;
    std::cout<< "energy_comp = " << energy_comp << std::endl;
    std::cout<< "energy      = " << energy << std::endl;
    return;
  } else {
    std::cout<< "energy comparison OK (tolerance " << tol << " max difference "
	     << max_diff << ")" << std::endl;
  }
  tol = 2.0e-3;
  max_diff = 0.0;
  bool vir_nan = false;
  for (int i=0;i < 9;i++) {
    max_diff = max(max_diff, fabs(virial_comp[i] - virial[i]));
    vir_nan = vir_nan | isnan(virial[i]);
  }
  if (max_diff > tol) {
    std::cout<< "virial comparison FAILED" << std::endl;
    std::cout<< "virial_comp | virial" << std::endl;
    for (int i=0;i < 9;i++) {
      std::cout << virial_comp[i] << " " << virial[i] << std::endl;
    }
    return;
  } else {
    std::cout<< "virial comparison OK (tolerance " << tol << " max difference "
	     << max_diff << ")" << std::endl;
  }


  // Run
  //grid.spread_charge(xyzq.ncoord, bspline);
  grid.spread_charge(xyzq.xyzq, xyzq.ncoord, recip);
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
  if (fft_type == BOX) {
    Matrix3d<float2> q_zfft_t(nfftx/2+1, nffty, nfftz);
    q_zfft.transpose_xyz_yzx(&q_zfft_t);
    if (!q_zfft_t.compare(grid.fft_grid, tol, max_diff)) {
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
  */

  tol = 1.0e-6;
  grid.scalar_sum(recip, kappa, false, false);
  /*
  if (fft_type == BOX) {
    Matrix3d<float2> q_zfft_summed_t(nfftx/2+1, nffty, nfftz);
    q_zfft_summed.transpose_xyz_yzx(&q_zfft_summed_t);
    if (!q_zfft_summed_t.compare(grid.fft_grid, tol, max_diff)) {
      std::cout<< "q_zfft_summed_t comparison FAILED" << std::endl;
      return;
    } else {
      std::cout<< "q_zfft_summed_t comparison OK (tolerance "<<tol<<" max difference "<<max_diff << ")" << std::endl;
    }
  } else {
    if (!q_zfft_summed.compare(grid.zfft_grid, tol, max_diff)) {
      std::cout<< "q_zfft_summed comparison FAILED" << std::endl;
      return;
    } else {
      std::cout<< "q_zfft_summed comparison OK (tolerance "<<tol<<" max difference "<<max_diff << ")" << std::endl;
    }
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
  //  grid.gather_force(ncoord, recip, bspline, force.stride, force.data);
  grid.gather_force(xyzq.xyzq, xyzq.ncoord, recip, force.stride, force.data);

  tol = 3.6e-4;
  if (!force_comp.compare(&force, tol, max_diff)) {
    std::cout<<"force comparison FAILED"<<std::endl;
  } else {
    std::cout<<"force comparison OK (tolerance " << tol << " max difference " << max_diff << ")" << std::endl;
  }

}
