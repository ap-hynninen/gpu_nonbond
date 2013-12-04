#ifndef GRID_H
#define GRID_H

#include <cufft.h>
#include "Bspline.h"
#include "Matrix3d.h"

//
// AT = Accumulation Type
// CT = Calculation Type
//
template <typename AT, typename CT>
class Grid {

public:

  // Grid data arrays
  Matrix3d<CT> mat1;
  Matrix3d<CT> mat2;

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

  cufftHandle xf_plan;

  void init(int x0, int x1, int y0, int y1, int z0, int z1, int order, 
	  bool y_land_locked, bool z_land_locked);

 public:
  Grid(int nfftx, int nffty, int nfftz, int order, int nnode, int mynode);
  ~Grid();

  void print_info();

  void spread_charge(const int ncoord, const Bspline<CT> &bspline);

  void make_fft_plans();

  void x_fft_r2c();
  void real2complex_fft();

  void test_copy();
  void test_transpose();

};

#endif // GRID_H
