#ifndef GRID_H
#define GRID_H

#include <cufft.h>
#include "Bspline.h"
#include "Matrix3d.h"

enum FFTtype {COLUMN, SLAB, BOX};

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

  // Type of FFT
  FFTtype fft_type;

  Matrix3d<AT> *accum_grid;    // data1
  Matrix3d<CT> *charge_grid;   // data2
  Matrix3d<CT> *solved_grid;   // data2

  // For COLUMN FFT
  Matrix3d<CT2> *xfft_grid;    // data2
  Matrix3d<CT2> *yfft_grid;    // data1
  Matrix3d<CT2> *zfft_grid;    // data2

  // For SLAB FFT. Also uses "zfft_grid" from above
  Matrix3d<CT2> *xyfft_grid;   // data2

  // For BOX FFT
  Matrix3d<CT2> *fft_grid;     // data2

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

  // Plans for "COLUMN" FFT
  cufftHandle x_r2c_plan;
  cufftHandle y_c2c_plan;
  cufftHandle z_c2c_plan;
  cufftHandle x_c2r_plan;

  // Plans for "SLAB" FFT. Also uses "z_c2c_plan" form above
  cufftHandle xy_r2c_plan;
  cufftHandle xy_c2r_plan;

  // Plans for "BOX" FFT
  cufftHandle r2c_plan;
  cufftHandle c2r_plan;

  // true for using multiple GPUs for the FFTs
  bool multi_gpu;

  // data for multi-gpus
  cudaLibXtDesc *multi_data;
  CT2 *host_data;
  CT *host_tmp;

  void init(int x0, int x1, int y0, int y1, int z0, int z1, int order, 
	  bool y_land_locked, bool z_land_locked);

  void make_fft_plans();

 public:
  Grid(int nfftx, int nffty, int nfftz, int order, FFTtype fft_type, int nnode, int mynode);
  ~Grid();

  void print_info();

  void spread_charge(const int ncoord, const Bspline<CT> &bspline);

  void scalar_sum(const double* recip, const double kappa,
		  CT* prefac_x, CT* prefac_y, CT* prefac_z);

  void gather_force(const int ncoord, const double* recip, const Bspline<CT> &bspline,
		    const int stride, CT* force);

  void x_fft_r2c(CT2 *data);
  void x_fft_c2r(CT2 *data);
  void y_fft_c2c(CT2 *data, const int direction);
  void z_fft_c2c(CT2 *data, const int direction);
  void r2c_fft();
  void c2r_fft();

  //  void test_copy();
  //  void test_transpose();

};

#endif // GRID_H
