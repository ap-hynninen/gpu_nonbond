#ifndef GRID_H
#define GRID_H

#include <cuda.h>
#include <cufft.h>
#include <cufftXt.h>
#include "Bspline.h"
#include "Matrix3d.h"

enum FFTtype {COLUMN, SLAB, BOX};

struct RecipEnergyVirial_t {
  // Energy
  double energy;

  // Virials
  double virial[6];
};


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

  // CCELEC is 1/ (4 pi eps ) in AKMA units, conversion from SI
  // units: CCELEC = e*e*Na / (4*pi*eps*1Kcal*1A)
  //
  //      parameter :: CCELEC=332.0636D0 ! old value of dubious origin
  //      parameter :: CCELEC=331.843D0  ! value from 1986-1987 CRC Handbook
  //                                   ! of Chemistry and Physics
  //  real(chm_real), parameter ::  &
  //       CCELEC_amber    = 332.0522173D0, &
  //       CCELEC_charmm   = 332.0716D0   , &
  //       CCELEC_discover = 332.054D0    , &
  //       CCELEC_namd     = 332.0636D0   

  static const double ccelec = 332.0716;

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

#if CUDA_VERSION >= 6000
  // data for multi-gpus
  cudaLibXtDesc *multi_data;
  CT2 *host_data;
  CT *host_tmp;
#endif

  // Stream where all computation takes place
  cudaStream_t stream;

  // Prefactor arrays
  CT* prefac_x;
  CT* prefac_y;
  CT* prefac_z;

  RecipEnergyVirial_t *h_energy_virial;

  void init(int x0, int x1, int y0, int y1, int z0, int z1, int order, 
	  bool y_land_locked, bool z_land_locked);

  void make_fft_plans();

  void calc_prefac();

 public:
  Grid(int nfftx, int nffty, int nfftz, int order, FFTtype fft_type, int nnode, int mynode, 
       cudaStream_t stream=0);
  ~Grid();

  void print_info();

  void spread_charge(const int ncoord, const Bspline<CT> &bspline);
  void spread_charge(const float4 *xyzq, const int ncoord, const double *recip);

  void scalar_sum(const double* recip, const double kappa,
		  const bool calc_energy, const bool calc_virial);

  void gather_force(const int ncoord, const double* recip, const Bspline<CT> &bspline,
		    const int stride, CT* force);
  void gather_force(const float4 *xyzq, const int ncoord, const double* recip,
		    const int stride, CT* force);

  void clear_energy_virial();
  void get_energy_virial(bool prev_calc_energy, bool prev_calc_virial,
			 double *energy, double *virial);

  void x_fft_r2c(CT2 *data);
  void x_fft_c2r(CT2 *data);
  void y_fft_c2c(CT2 *data, const int direction);
  void z_fft_c2c(CT2 *data, const int direction);
  void r2c_fft();
  void c2r_fft();

  int get_nfftx() {return nfftx;}
  int get_nffty() {return nffty;}
  int get_nfftz() {return nfftz;}
  int get_order() {return order;}
  void set_order(int order);

  //  void test_copy();
  //  void test_transpose();

};

#endif // GRID_H
