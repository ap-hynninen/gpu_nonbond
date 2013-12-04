#ifndef BSPLINE_H
#define BSPLINE_H

#include "gridp_t.h"

template <typename T> class Bspline {

private:

  // Length of the B-spline data arrays
  int theta_len;
  int dtheta_len;

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

  // Grid positions and charge (int x, int y, int z, float q)
  gridp_t *gridp;

private:

  void init(const int ncoord);

  template <typename B>
  void set_recip(const B *recip);

public:

  Bspline(const int ncoord, const int order, const double *recip);
  ~Bspline();
  void fill_bspline(const float4 *xyzq, const int ncoord, 
		    const int nfftx, const int nffty, const int nfftz);
};

#endif // BSPLINE_H
