#ifndef GRID_H
#define GRID_H

#include <cufft.h>
#include "Bspline.h"

template <typename T>
class Grid {

private:

  // Grid data arrays
  T* data_T;
  float* data_float;

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

  template <typename B, typename B3>
  void spread_charge(const int ncoord, const Bspline<B, B3> &bspline);

  void make_fft_plans();

  void real2complex_fft();

};

#endif // GRID_H
