#ifndef GRID_H
#define GRID_H

#include <cufft.h>
#include "Bspline.h"
#include "Matrix3d.h"

//
// AT  = Accumulation Type
// CT  = Calculation Type (real)
// CT2 = Calculation Type (complex)
//
template <typename AT, typename CT, typename CT2>
class Grid {

public:

  // Grid data arrays
  CT *data1;
  CT *data2;

  int data1_len;
  int data2_len;

  Matrix3d<AT> *accum_grid;    // data1
  Matrix3d<CT> *charge_grid;   // data2
  Matrix3d<CT2> *xfft_grid;    // data2
  Matrix3d<CT2> *yfft_grid;    // data1
  Matrix3d<CT2> *zfft_grid;    // data2

private:

  // Order of interpolation
  int order;

  // Size of the entire grid
  int nfftx;
  int nffty;
  int nfftz;
  
  // Region boundaries in real space1
  int x0, x1;
  int y0, y1;
  int z0, z1;
  
  // Writing region in real space
  int xlo, xhi;
  int ylo, yhi;
  int zlo, zhi;

  // Writing region size on real space
  int xsize;
  int ysize;
  int zsize;

  // Total size of the data array
  int data_size;

  // Plans for FFT
  cufftHandle x_r2c_plan;
  cufftHandle y_c2c_plan;
  cufftHandle z_c2c_plan;
  cufftHandle x_c2r_plan;

  void init(int x0, int x1, int y0, int y1, int z0, int z1, int order, 
	  bool y_land_locked, bool z_land_locked);

 public:
  Grid(int nfftx, int nffty, int nfftz, int order, int nnode, int mynode);
  ~Grid();

  void print_info();

  void spread_charge(const int ncoord, const Bspline<CT> &bspline);

  void scalar_sum(const double* recip, const double kappa,
		  const CT* prefac_x, const CT* prefac_y, const CT* prefac_z);

  void gather_force(const int ncoord, const Bspline<CT> &bspline, AT* force);

  void make_fft_plans();

  void x_fft_r2c();
  void x_fft_c2r();
  void y_fft_c2c(int direction);
  void z_fft_c2c(int direction);
  void r2c_fft();
  void c2r_fft();

  //  void test_copy();
  //  void test_transpose();

};

#endif // GRID_H
