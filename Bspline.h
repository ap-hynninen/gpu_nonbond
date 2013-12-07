#ifndef BSPLINE_H
#define BSPLINE_H

#include "gridp_t.h"

template <typename T> class Bspline {

private:

  // Length of the B-spline data arrays
  int theta_len;
  int dtheta_len;

  // Size of the FFT
  int nfftx;
  int nffty;
  int nfftz;

  // B-spline order
  int order;

  // Length of gridp
  int gridp_len;

  // Reciprocal vectors
  T *recip;

public:

  // B-spline data
  T *theta;
  T *dtheta;

  // prefac arrays
  T* prefac_x;
  T* prefac_y;
  T* prefac_z;

  // Grid positions and charge (int x, int y, int z, float q)
  gridp_t *gridp;

private:

  void set_ncoord(const int ncoord);
  void dftmod(double *bsp_mod, const double *bsp_arr, const int nfft);
  void fill_bspline_host(const double w, double *array, double *darray);

public:

  Bspline(const int ncoord, const int order, const int nfftx, const int nffty, const int nfftz);
  ~Bspline();

  template <typename B>
  void set_recip(const B *recip);

  void fill_bspline(const float4 *xyzq, const int ncoord);
  void calc_prefac();
};

#endif // BSPLINE_H
