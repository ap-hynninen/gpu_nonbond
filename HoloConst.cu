#include <iostream>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "HoloConst.h"

struct HoloConstSettings_t {
  // Solvents
  double mO_div_mH2O;
  double mH_div_mH2O;
  double ra, rc, rb, ra_inv, rc2;
  int nsolvent;
  const int3* __restrict__ solvent_ind;

  // Pairs
  int npair;
  int2* pair_ind;
  double *pair_constr;
  double *pair_mass;

  // Trips
  int ntrip;
  int3* trip_ind;
  double *trip_constr;
  double *trip_mass;

  // Quads
  int nquad;
  int4* quad_ind;
  double *quad_constr;
  double *quad_mass;

  double shake_tol;
  int max_niter;

  int stride;
  int stride2;
  const double* __restrict__ xyz0;
  const double* __restrict__ xyz1;
  double* __restrict__ xyz2;
};

static HoloConstSettings_t h_setup;
static __constant__ HoloConstSettings_t d_setup;

static texture<int2, 1, cudaReadModeElementType> xyz0_texref;
static texture<int2, 1, cudaReadModeElementType> xyz1_texref;
static int2* xyz0_texref_pointer = NULL;
static int2* xyz1_texref_pointer = NULL;
static int texref_stride = 0;

template <int t>
__forceinline__ __device__ double load_coord(const int ind) {
  if (t == 0) {
#if __CUDA_ARCH__ < 350
    int2 val = tex1Dfetch(xyz0_texref, ind);
    return __hiloint2double(val.y, val.x);
#else
    return __ldg(&d_setup.xyz0[ind]);
#endif
  } else {
#if __CUDA_ARCH__ < 350
    int2 val = tex1Dfetch(xyz1_texref, ind);
    return __hiloint2double(val.y, val.x);
#else
    return __ldg(&d_setup.xyz1[ind]);
#endif
  }
}

__forceinline__ __device__ void pair_calc(int imol) {
    int2 ind = d_setup.pair_ind[imol];

    // Load coordinates
    double x0i = load_coord<0>(ind.x);
    double y0i = load_coord<0>(ind.x+d_setup.stride);
    double z0i = load_coord<0>(ind.x+d_setup.stride2);
    double x0j = load_coord<0>(ind.y);
    double y0j = load_coord<0>(ind.y+d_setup.stride);
    double z0j = load_coord<0>(ind.y+d_setup.stride2);

    double x1i = load_coord<1>(ind.x);
    double y1i = load_coord<1>(ind.x+d_setup.stride);
    double z1i = load_coord<1>(ind.x+d_setup.stride2);
    double x1j = load_coord<1>(ind.y);
    double y1j = load_coord<1>(ind.y+d_setup.stride);
    double z1j = load_coord<1>(ind.y+d_setup.stride2);

    double xpij = x1i - x1j;
    double ypij = y1i - y1j;
    double zpij = z1i - z1j;
    double rijsq = xpij*xpij + ypij*ypij + zpij*zpij;

    double diff = d_setup.pair_constr[imol] - rijsq;

    double xrij = x0i - x0j;
    double yrij = y0i - y0j;
    double zrij = z0i - z0j;

    double rrijsq = xrij*xrij + yrij*yrij + zrij*zrij;
    double rijrijp = xrij*xpij  + yrij*ypij  + zrij*zpij;
    double lambda = 2.0*(-rijrijp + sqrt(rijrijp*rijrijp+rrijsq*diff))/(rrijsq);

    double hmassi_val = d_setup.pair_mass[imol*2];
    double hmassj_val = d_setup.pair_mass[imol*2+1];

    x1i += hmassi_val*lambda*xrij;
    y1i += hmassi_val*lambda*yrij;
    z1i += hmassi_val*lambda*zrij;
    x1j -= hmassj_val*lambda*xrij;
    y1j -= hmassj_val*lambda*yrij;
    z1j -= hmassj_val*lambda*zrij;

    // Store results
    d_setup.xyz2[ind.x]         = x1i;
    d_setup.xyz2[ind.x+d_setup.stride]  = y1i;
    d_setup.xyz2[ind.x+d_setup.stride2] = z1i;
    d_setup.xyz2[ind.y]         = x1j;
    d_setup.xyz2[ind.y+d_setup.stride]  = y1j;
    d_setup.xyz2[ind.y+d_setup.stride2] = z1j;
}

__forceinline__ __device__ void trip_calc(int imol) {
    int3 ind = d_setup.trip_ind[imol];

    // Load coordinates
    double x0i = load_coord<0>(ind.x);
    double y0i = load_coord<0>(ind.x+d_setup.stride);
    double z0i = load_coord<0>(ind.x+d_setup.stride2);
    double x0j = load_coord<0>(ind.y);
    double y0j = load_coord<0>(ind.y+d_setup.stride);
    double z0j = load_coord<0>(ind.y+d_setup.stride2);
    double x0k = load_coord<0>(ind.z);
    double y0k = load_coord<0>(ind.z+d_setup.stride);
    double z0k = load_coord<0>(ind.z+d_setup.stride2);

    double x1i = load_coord<1>(ind.x);
    double y1i = load_coord<1>(ind.x+d_setup.stride);
    double z1i = load_coord<1>(ind.x+d_setup.stride2);
    double x1j = load_coord<1>(ind.y);
    double y1j = load_coord<1>(ind.y+d_setup.stride);
    double z1j = load_coord<1>(ind.y+d_setup.stride2);
    double x1k = load_coord<1>(ind.z);
    double y1k = load_coord<1>(ind.z+d_setup.stride);
    double z1k = load_coord<1>(ind.z+d_setup.stride2);

    double xrij = x0i - x0j;
    double yrij = y0i - y0j;
    double zrij = z0i - z0j;
    double xrik = x0i - x0k;
    double yrik = y0i - y0k;
    double zrik = z0i - z0k;

    double rrijsq = xrij*xrij + yrij*yrij + zrij*zrij;
    double rriksq = xrik*xrik + yrik*yrik + zrik*zrik;
    double rijrik = xrij*xrik + yrij*yrik + zrij*zrik;

    double mmi = d_setup.trip_mass[imol*5];
    double mmj = d_setup.trip_mass[imol*5+1];
    double mmk = d_setup.trip_mass[imol*5+2];
    double mij = d_setup.trip_mass[imol*5+3];
    double mik = d_setup.trip_mass[imol*5+4];

    double acorr1 = mij*mij*rrijsq;
    double acorr2 = mij*mmi*2.0*rijrik;
    double acorr3 = mmi*mmi*rriksq;
    double acorr4 = mmi*mmi*rrijsq;
    double acorr5 = mik*mmi*2.0*rijrik;
    double acorr6 = mik*mik*rriksq;

    double xpij = x1i - x1j;
    double ypij = y1i - y1j;
    double zpij = z1i - z1j;
    double xpik = x1i - x1k;
    double ypik = y1i - y1k;
    double zpik = z1i - z1k;

    double rijsq = xpij*xpij + ypij*ypij + zpij*zpij;
    double riksq = xpik*xpik + ypik*ypik + zpik*zpik;
    double dij = d_setup.trip_constr[imol*2]   - rijsq;
    double dik = d_setup.trip_constr[imol*2+1] - riksq;
    double rijrijp = xrij*xpij + yrij*ypij + zrij*zpij;
    double rijrikp = xrij*xpik + yrij*ypik + zrij*zpik;
    double rikrijp = xpij*xrik + ypij*yrik + zpij*zrik;
    double rikrikp = xrik*xpik + yrik*ypik + zrik*zpik;
    double dinv=0.5/(rijrijp*rikrikp*mij*mik - rijrikp*rikrijp*mmi*mmi);
    
    double a12 = dinv*( rikrikp*mik*(dij) - rikrijp*mmi*(dik));
    double a13 = dinv*(-mmi*rijrikp*(dij) + rijrijp*mij*(dik));

    double a120 = 0.0;
    double a130 = 0.0;

    int niter = 0;
    while ((fabs(a120-a12) > d_setup.shake_tol || 
	    fabs(a130-a13) > d_setup.shake_tol) && 
	   (niter < d_setup.max_niter)) {
      a120 = a12;
      a130 = a13;
      double a12corr = acorr1*a12*a12 + acorr2*a12*a13 + acorr3*a13*a13;
      double a13corr = acorr4*a12*a12 + acorr5*a12*a13 + acorr6*a13*a13;
      a12 = dinv*( rikrikp*mik*(dij-a12corr) - rikrijp*mmi*(dik-a13corr));
      a13 = dinv*(-mmi*rijrikp*(dij-a12corr) + rijrijp*mij*(dik-a13corr));
      niter++;
    }

    x1i += mmi*(xrij*a12+xrik*a13);
    x1j -= mmj*(xrij*a12);
    x1k -= mmk*(xrik*a13);
    y1i += mmi*(yrij*a12+yrik*a13);
    y1j -= mmj*(yrij*a12);
    y1k -= mmk*(yrik*a13);
    z1i += mmi*(zrij*a12+zrik*a13);
    z1j -= mmj*(zrij*a12);
    z1k -= mmk*(zrik*a13);

    d_setup.xyz2[ind.x]         = x1i;
    d_setup.xyz2[ind.x+d_setup.stride]  = y1i;
    d_setup.xyz2[ind.x+d_setup.stride2] = z1i;
    d_setup.xyz2[ind.y]         = x1j;
    d_setup.xyz2[ind.y+d_setup.stride]  = y1j;
    d_setup.xyz2[ind.y+d_setup.stride2] = z1j;
    d_setup.xyz2[ind.z]         = x1k;
    d_setup.xyz2[ind.z+d_setup.stride]  = y1k;
    d_setup.xyz2[ind.z+d_setup.stride2] = z1k;

}

__forceinline__ __device__ void quad_calc(int imol) {
    int4 ind = d_setup.quad_ind[imol];

    // Load coordinates
    double x0i = load_coord<0>(ind.x);
    double y0i = load_coord<0>(ind.x+d_setup.stride);
    double z0i = load_coord<0>(ind.x+d_setup.stride2);
    double x0j = load_coord<0>(ind.y);
    double y0j = load_coord<0>(ind.y+d_setup.stride);
    double z0j = load_coord<0>(ind.y+d_setup.stride2);
    double x0k = load_coord<0>(ind.z);
    double y0k = load_coord<0>(ind.z+d_setup.stride);
    double z0k = load_coord<0>(ind.z+d_setup.stride2);
    double x0l = load_coord<0>(ind.w);
    double y0l = load_coord<0>(ind.w+d_setup.stride);
    double z0l = load_coord<0>(ind.w+d_setup.stride2);

    double x1i = load_coord<1>(ind.x);
    double y1i = load_coord<1>(ind.x+d_setup.stride);
    double z1i = load_coord<1>(ind.x+d_setup.stride2);
    double x1j = load_coord<1>(ind.y);
    double y1j = load_coord<1>(ind.y+d_setup.stride);
    double z1j = load_coord<1>(ind.y+d_setup.stride2);
    double x1k = load_coord<1>(ind.z);
    double y1k = load_coord<1>(ind.z+d_setup.stride);
    double z1k = load_coord<1>(ind.z+d_setup.stride2);
    double x1l = load_coord<1>(ind.w);
    double y1l = load_coord<1>(ind.w+d_setup.stride);
    double z1l = load_coord<1>(ind.w+d_setup.stride2);

    double xrij = x0i - x0j;
    double yrij = y0i - y0j;
    double zrij = z0i - z0j;
    double xrik = x0i - x0k;
    double yrik = y0i - y0k;
    double zrik = z0i - z0k;       
    double xril = x0i - x0l;
    double yril = y0i - y0l;
    double zril = z0i - z0l;

    //i2 = ammi_ind(iconst)

    double rrijsq = xrij*xrij + yrij*yrij + zrij*zrij;
    double rriksq = xrik*xrik + yrik*yrik + zrik*zrik;
    double rrilsq = xril*xril + yril*yril + zril*zril;
    double rijrik = xrij*xrik + yrij*yrik + zrij*zrik;
    double rijril = xrij*xril + yrij*yril + zrij*zril;
    double rikril = xrik*xril + yrik*yril + zrik*zril;

    double mmi = d_setup.quad_mass[imol*7];
    double mmj = d_setup.quad_mass[imol*7+1];
    double mmk = d_setup.quad_mass[imol*7+2];
    double mml = d_setup.quad_mass[imol*7+3];
    double mij = d_setup.quad_mass[imol*7+4];
    double mik = d_setup.quad_mass[imol*7+5];
    double mil = d_setup.quad_mass[imol*7+6];

    double acorr1  =     mij*mij*rrijsq;
    double acorr2  = 2.0*mij*mmi*rijrik;
    double acorr3  =     mmi*mmi*rriksq;
    double acorr4  =     mmi*mmi*rrijsq;
    double acorr5  = 2.0*mik*mmi*rijrik;
    double acorr6  =     mik*mik*rriksq;
    double acorr7  = 2.0*mij*mmi*rijril;
    double acorr8  = 2.0*mmi*mmi*rikril;
    double acorr9  =     mmi*mmi*rrilsq;
    double acorr10 = 2.0*mmi*mmi*rijril;
    double acorr11 = 2.0*mmi*mik*rikril;
    double acorr12 = 2.0*mmi*mmi*rijrik;
    double acorr13 = 2.0*mmi*mil*rijril;
    double acorr14 = 2.0*mmi*mil*rikril;
    double acorr15 =     mil*mil*rrilsq;

    double xpij = x1i - x1j;
    double ypij = y1i - y1j;
    double zpij = z1i - z1j;
    double xpik = x1i - x1k;
    double ypik = y1i - y1k;
    double zpik = z1i - z1k;
    double xpil = x1i - x1l;
    double ypil = y1i - y1l;
    double zpil = z1i - z1l;

    double rijsq = xpij*xpij + ypij*ypij + zpij*zpij;
    double riksq = xpik*xpik + ypik*ypik + zpik*zpik;
    double rilsq = xpil*xpil + ypil*ypil + zpil*zpil;
    double dij = d_setup.quad_constr[imol*3]   - rijsq;
    double dik = d_setup.quad_constr[imol*3+1] - riksq;
    double dil = d_setup.quad_constr[imol*3+2] - rilsq;
    double rijrijp = xrij*xpij + yrij*ypij + zrij*zpij;
    double rijrikp = xrij*xpik + yrij*ypik + zrij*zpik;
    double rijrilp = xrij*xpil + yrij*ypil + zrij*zpil;
    double rikrijp = xrik*xpij + yrik*ypij + zrik*zpij;
    double rikrikp = xrik*xpik + yrik*ypik + zrik*zpik;
    double rikrilp = xrik*xpil + yrik*ypil + zrik*zpil;
    double rilrijp = xril*xpij + yril*ypij + zril*zpij;
    double rilrikp = xril*xpik + yril*ypik + zril*zpik;
    double rilrilp = xril*xpil + yril*ypil + zril*zpil;
    double d1 = mik*mil*rikrikp*rilrilp - mmi*mmi*rikrilp*rilrikp;
    double d2 = mmi*mil*rikrijp*rilrilp - mmi*mmi*rikrilp*rilrijp;
    double d3 = mmi*mmi*rikrijp*rilrikp - mik*mmi*rikrikp*rilrijp;
    double d4 = mmi*mil*rijrikp*rilrilp - mmi*mmi*rijrilp*rilrikp;
    double d5 = mij*mil*rijrijp*rilrilp - mmi*mmi*rijrilp*rilrijp;
    double d6 = mij*mmi*rijrijp*rilrikp - mmi*mmi*rijrikp*rilrijp;
    double d7 = mmi*mmi*rijrikp*rikrilp - mmi*mik*rijrilp*rikrikp;
    double d8 = mij*mmi*rijrijp*rikrilp - mmi*mmi*rijrilp*rikrijp;
    double d9 = mij*mik*rijrijp*rikrikp - mmi*mmi*rijrikp*rikrijp;

    double dinv = 0.5/(rijrijp*mij*d1 - mmi*rijrikp*d2 + mmi*rijrilp*d3);
    
    double a12 = dinv*( d1*dij - d2*dik + d3*dil);
    double a13 = dinv*(-d4*dij + d5*dik - d6*dil);
    double a14 = dinv*( d7*dij - d8*dik + d9*dil);

    double a120 = 0.0;
    double a130 = 0.0;
    double a140 = 0.0;
    
    int niter = 0;

    while ((fabs(a120-a12) > d_setup.shake_tol || 
	    fabs(a130-a13) > d_setup.shake_tol || 
	    fabs(a140-a14) > d_setup.shake_tol) &&
	   (niter  <  d_setup.max_niter)) {
      a120 = a12;
      a130 = a13;
      a140 = a14;

      double a12corr = acorr1*a12*a12 + acorr2*a12*a13 + acorr3*a13*a13
	+ acorr7*a12*a14 + acorr8*a13*a14 + acorr9*a14*a14;
      double a13corr = acorr4*a12*a12 + acorr5*a12*a13 + acorr6*a13*a13
	+ acorr10*a12*a14 + acorr11*a13*a14 + acorr9*a14*a14;
      double a14corr = acorr4*a12*a12 + acorr12*a12*a13 + acorr3*a13*a13
	+ acorr13*a12*a14 + acorr14*a13*a14 + acorr15*a14*a14;

      a12 = dinv*( d1*(dij-a12corr) - d2*(dik-a13corr) + d3*(dil-a14corr));
      a13 = dinv*( -d4*(dij-a12corr) + d5*(dik-a13corr) - d6*(dil-a14corr));
      a14 = dinv*( d7*(dij-a12corr) - d8*(dik-a13corr) + d9*(dil-a14corr));
      niter++;
    }

    x1i  += mmi*(xrij*a12+xrik*a13+xril*a14);
    y1i  += mmi*(yrij*a12+yrik*a13+yril*a14);
    z1i  += mmi*(zrij*a12+zrik*a13+zril*a14);
    x1j  -= mmj*(xrij*a12);
    y1j  -= mmj*(yrij*a12);
    z1j  -= mmj*(zrij*a12);
    x1k  -= mmk*(xrik*a13);
    y1k  -= mmk*(yrik*a13);
    z1k  -= mmk*(zrik*a13);
    x1l  -= mml*(xril*a14);
    y1l  -= mml*(yril*a14);
    z1l  -= mml*(zril*a14);

    d_setup.xyz2[ind.x]         = x1i;
    d_setup.xyz2[ind.x+d_setup.stride]  = y1i;
    d_setup.xyz2[ind.x+d_setup.stride2] = z1i;
    d_setup.xyz2[ind.y]         = x1j;
    d_setup.xyz2[ind.y+d_setup.stride]  = y1j;
    d_setup.xyz2[ind.y+d_setup.stride2] = z1j;
    d_setup.xyz2[ind.z]         = x1k;
    d_setup.xyz2[ind.z+d_setup.stride]  = y1k;
    d_setup.xyz2[ind.z+d_setup.stride2] = z1k;
    d_setup.xyz2[ind.w]         = x1l;
    d_setup.xyz2[ind.w+d_setup.stride]  = y1l;
    d_setup.xyz2[ind.w+d_setup.stride2] = z1l;
}

__forceinline__ __device__ void solvent_calc(int imol) {

    int3 ind = d_setup.solvent_ind[imol];

    // Load coordinates
    double x0i = load_coord<0>(ind.x);
    double y0i = load_coord<0>(ind.x+d_setup.stride);
    double z0i = load_coord<0>(ind.x+d_setup.stride2);
    double x0j = load_coord<0>(ind.y);
    double y0j = load_coord<0>(ind.y+d_setup.stride);
    double z0j = load_coord<0>(ind.y+d_setup.stride2);
    double x0k = load_coord<0>(ind.z);
    double y0k = load_coord<0>(ind.z+d_setup.stride);
    double z0k = load_coord<0>(ind.z+d_setup.stride2);

    double x1i = load_coord<1>(ind.x);
    double y1i = load_coord<1>(ind.x+d_setup.stride);
    double z1i = load_coord<1>(ind.x+d_setup.stride2);
    double x1j = load_coord<1>(ind.y);
    double y1j = load_coord<1>(ind.y+d_setup.stride);
    double z1j = load_coord<1>(ind.y+d_setup.stride2);
    double x1k = load_coord<1>(ind.z);
    double y1k = load_coord<1>(ind.z+d_setup.stride);
    double z1k = load_coord<1>(ind.z+d_setup.stride2);

    //
    // Convert to primed coordinates
    //

    // Calculate center of mass for (x1, y1, z1)
    double xcm = x1i*d_setup.mO_div_mH2O + (x1j + x1k)*d_setup.mH_div_mH2O;
    double ycm = y1i*d_setup.mO_div_mH2O + (y1j + y1k)*d_setup.mH_div_mH2O;
    double zcm = z1i*d_setup.mO_div_mH2O + (z1j + z1k)*d_setup.mH_div_mH2O;

    // Calculate (x1, y1, z1) with center of mass at origin
    double xa1 = x1i - xcm;
    double ya1 = y1i - ycm;
    double za1 = z1i - zcm;
    double xb1 = x1j - xcm;
    double yb1 = y1j - ycm;
    double zb1 = z1j - zcm;
    double xc1 = x1k - xcm;
    double yc1 = y1k - ycm;
    double zc1 = z1k - zcm;

    double xb0 = x0j - x0i;
    double yb0 = y0j - y0i;
    double zb0 = z0j - z0i;
    double xc0 = x0k - x0i;
    double yc0 = y0k - y0i;
    double zc0 = z0k - z0i;

    // (xb0, yb0, zb0), (xc0, yc0, zc0), and (xa1, ya1, za1) define the primed coordinate set:
    // * X'Y' plane is parallel to the plane defined by (xb0, yb0, zb0) and (xc0, yc0, zc0)
    // * Y'Z' plane contains (xa1, ya1, za1)

    double xakszd = yb0 * zc0 - zb0 * yc0;
    double yakszd = zb0 * xc0 - xb0 * zc0;
    double zakszd = xb0 * yc0 - yb0 * xc0;
    double xaksxd = ya1 * zakszd - za1 * yakszd;
    double yaksxd = za1 * xakszd - xa1 * zakszd;
    double zaksxd = xa1 * yakszd - ya1 * xakszd;
    double xaksyd = yakszd * zaksxd - zakszd * yaksxd;
    double yaksyd = zakszd * xaksxd - xakszd * zaksxd;
    double zaksyd = xakszd * yaksxd - yakszd * xaksxd;

    double axlng_inv = rsqrt(xaksxd * xaksxd + yaksxd * yaksxd + zaksxd * zaksxd);
    double aylng_inv = rsqrt(xaksyd * xaksyd + yaksyd * yaksyd + zaksyd * zaksyd);
    double azlng_inv = rsqrt(xakszd * xakszd + yakszd * yakszd + zakszd * zakszd);

    double trans11 = xaksxd * axlng_inv;
    double trans21 = yaksxd * axlng_inv;
    double trans31 = zaksxd * axlng_inv;
    double trans12 = xaksyd * aylng_inv;
    double trans22 = yaksyd * aylng_inv;
    double trans32 = zaksyd * aylng_inv;
    double trans13 = xakszd * azlng_inv;
    double trans23 = yakszd * azlng_inv;
    double trans33 = zakszd * azlng_inv;

    // Calculate necessary primed coordinates
    double xb0p = trans11 * xb0 + trans21 * yb0 + trans31 * zb0;
    double yb0p = trans12 * xb0 + trans22 * yb0 + trans32 * zb0;
    double xc0p = trans11 * xc0 + trans21 * yc0 + trans31 * zc0;
    double yc0p = trans12 * xc0 + trans22 * yc0 + trans32 * zc0;
    double za1p = trans13 * xa1 + trans23 * ya1 + trans33 * za1;
    double xb1p = trans11 * xb1 + trans21 * yb1 + trans31 * zb1;
    double yb1p = trans12 * xb1 + trans22 * yb1 + trans32 * zb1;
    double zb1p = trans13 * xb1 + trans23 * yb1 + trans33 * zb1;
    double xc1p = trans11 * xc1 + trans21 * yc1 + trans31 * zc1;
    double yc1p = trans12 * xc1 + trans22 * yc1 + trans32 * zc1;
    double zc1p = trans13 * xc1 + trans23 * yc1 + trans33 * zc1;

    //
    // Calculate rotation angles (phi, psi, theta)
    //

    double sinphi = za1p * d_setup.ra_inv;
    double cosphi = sqrt(1.0 - sinphi*sinphi);
    double sinpsi = (zb1p - zc1p) / (d_setup.rc2 * cosphi);
    double cospsi = sqrt(1.0 - sinpsi*sinpsi);

    double ya2p =   d_setup.ra * cosphi;
    double xb2p = - d_setup.rc * cospsi;
    double yb2p = - d_setup.rb * cosphi - d_setup.rc *sinpsi * sinphi;
    double yc2p = - d_setup.rb * cosphi + d_setup.rc *sinpsi * sinphi;

    //          xb2p =  -half * sqrt(hhhh - (yb2p-yc2p) * (yb2p-yc2p) - (zb1p-zc1p) * (zb1p-zc1p))

    double alpha = (xb2p * (xb0p-xc0p) + yb0p * yb2p + yc0p * yc2p);
    double beta  = (xb2p * (yc0p-yb0p) + xb0p * yb2p + xc0p * yc2p);
    double gamma = xb0p * yb1p - xb1p * yb0p + xc0p * yc1p - xc1p * yc0p;

    double alpha_beta = alpha * alpha + beta * beta;
    double sintheta = (alpha*gamma - beta * sqrt(alpha_beta - gamma*gamma)) / alpha_beta;

    double costheta = sqrt(1.0 - sintheta*sintheta);
    double xa3p = -ya2p * sintheta;
    double ya3p =  ya2p * costheta;
    double za3p =  za1p;
    double xb3p =  xb2p * costheta - yb2p * sintheta;
    double yb3p =  xb2p * sintheta + yb2p * costheta;
    double zb3p =  zb1p;
    double xc3p = -xb2p * costheta - yc2p * sintheta;
    double yc3p = -xb2p * sintheta + yc2p * costheta;
    double zc3p =  zc1p;

    d_setup.xyz2[ind.x]         = xcm + trans11 * xa3p + trans12 * ya3p + trans13 * za3p;
    d_setup.xyz2[ind.x+d_setup.stride]  = ycm + trans21 * xa3p + trans22 * ya3p + trans23 * za3p;
    d_setup.xyz2[ind.x+d_setup.stride2] = zcm + trans31 * xa3p + trans32 * ya3p + trans33 * za3p;
    d_setup.xyz2[ind.y]         = xcm + trans11 * xb3p + trans12 * yb3p + trans13 * zb3p;
    d_setup.xyz2[ind.y+d_setup.stride]  = ycm + trans21 * xb3p + trans22 * yb3p + trans23 * zb3p;
    d_setup.xyz2[ind.y+d_setup.stride2] = zcm + trans31 * xb3p + trans32 * yb3p + trans33 * zb3p;
    d_setup.xyz2[ind.z]         = xcm + trans11 * xc3p + trans12 * yc3p + trans13 * zc3p;
    d_setup.xyz2[ind.z+d_setup.stride]  = ycm + trans21 * xc3p + trans22 * yc3p + trans23 * zc3p;
    d_setup.xyz2[ind.z+d_setup.stride2] = zcm + trans31 * xc3p + trans32 * yc3p + trans33 * zc3p;
}

__global__ void all_kernels() {
  const int imol = threadIdx.x + blockDim.x*blockIdx.x;

  if (imol < d_setup.nsolvent) {
    solvent_calc(imol);
  } else if (imol < d_setup.nsolvent + d_setup.npair) {
    pair_calc(imol - d_setup.nsolvent);
  } else if (imol < d_setup.nsolvent + d_setup.npair + d_setup.ntrip) {
    trip_calc(imol - d_setup.nsolvent - d_setup.npair);
  } else if (imol < d_setup.nsolvent + d_setup.npair + d_setup.ntrip + d_setup.nquad) {
    quad_calc(imol - d_setup.nsolvent - d_setup.npair - d_setup.ntrip);
  }
}

//################################################################################################

//
// Class creator
//
HoloConst::HoloConst() {

  max_niter = 1000;
  shake_tol = 1.0e-8;

  nsolvent = 0;
  solvent_ind_len = 0;
  solvent_ind = NULL;

  npair = 0;
  pair_ind_len = 0;
  pair_ind = NULL;
  pair_constr_len = 0;
  pair_constr = NULL;
  pair_mass_len = 0;
  pair_mass = NULL;

  ntrip = 0;
  trip_ind_len = 0;
  trip_ind = NULL;
  trip_constr_len = 0;
  trip_constr = NULL;
  trip_mass_len = 0;
  trip_mass = NULL;

  nquad = 0;
  quad_ind_len = 0;
  quad_ind = NULL;
  quad_constr_len = 0;
  quad_constr = NULL;
  quad_mass_len = 0;
  quad_mass = NULL;

  xyz0_texref_pointer = NULL;
  xyz1_texref_pointer = NULL;
  texref_stride = 0;

  if (get_cuda_arch() < 350) {
    use_textures = true;
  } else {
    use_textures = false;
  }
}

//
// Class destructor
//
HoloConst::~HoloConst() {

  if (xyz0_texref_pointer != NULL) {
    cudaCheck(cudaUnbindTexture(xyz0_texref));
    xyz0_texref_pointer = NULL;
  }

  if (xyz1_texref_pointer != NULL) {
    cudaCheck(cudaUnbindTexture(xyz1_texref));
    xyz1_texref_pointer = NULL;
  }

  if (solvent_ind != NULL) deallocate<int3>(&solvent_ind);

  if (pair_ind != NULL) deallocate<int2>(&pair_ind);
  if (pair_constr != NULL) deallocate<double>(&pair_constr);
  if (pair_mass != NULL) deallocate<double>(&pair_mass);

  if (trip_ind != NULL) deallocate<int3>(&trip_ind);
  if (trip_constr != NULL) deallocate<double>(&trip_constr);
  if (trip_mass != NULL) deallocate<double>(&trip_mass);

  if (quad_ind != NULL) deallocate<int4>(&quad_ind);
  if (quad_constr != NULL) deallocate<double>(&quad_constr);
  if (quad_mass != NULL) deallocate<double>(&quad_mass);

}

//
// Setup
//
void HoloConst::setup_solvent_parameters(double mO, double mH, double rOHsq, double rHHsq) {

  double mH2O = mO + mH + mH;
  mO_div_mH2O = mO/mH2O;
  mH_div_mH2O = mH/mH2O;

  // Setup ra, rb, rc
  ra = mH_div_mH2O*sqrt(4.0*rOHsq - rHHsq);
  ra_inv = 1.0/ra;
  rb = ra*mO/(2.0*mH);
  rc = sqrt(rHHsq)/2.0;
  rc2 = 2.0*rc;

}

//
// Setups all ind, mass, and constr arrays
//
void HoloConst::setup_ind_mass_constr(int npair, int2 *h_pair_ind,
				      double *h_pair_constr, double *h_pair_mass,
				      int ntrip, int3 *h_trip_ind,
				      double *h_trip_constr, double *h_trip_mass,
				      int nquad, int4 *h_quad_ind,
				      double *h_quad_constr, double *h_quad_mass,
				      int nsolvent, int3 *h_solvent_ind) {

  // Copy ind, mass, and constr from CPU to GPU
  set_pair_ind(npair, h_pair_ind, h_pair_constr, h_pair_mass);
  set_trip_ind(ntrip, h_trip_ind, h_trip_constr, h_trip_mass);
  set_quad_ind(nquad, h_quad_ind, h_quad_constr, h_quad_mass);
  set_solvent_ind(nsolvent, h_solvent_ind);

}

//
// Updates h_setup and d_setup if neccessary
//
void HoloConst::update_setup(int stride, double *xyz0, double *xyz1, double *xyz2,
			     cudaStream_t stream) {

  bool update = false;

  update |= h_setup.nsolvent != nsolvent;
  update |= h_setup.solvent_ind != solvent_ind;
  update |= h_setup.mO_div_mH2O != mO_div_mH2O;
  update |= h_setup.mH_div_mH2O != mH_div_mH2O;
  update |= h_setup.ra != ra;
  update |= h_setup.rc != rc;
  update |= h_setup.rb != rb;
  update |= h_setup.ra_inv != ra_inv;
  update |= h_setup.rc2 != rc2;

  update |= h_setup.npair != npair;
  update |= h_setup.pair_ind != pair_ind;
  update |= h_setup.pair_constr != pair_constr;
  update |= h_setup.pair_mass != pair_mass;

  update |= h_setup.ntrip != ntrip;
  update |= h_setup.trip_ind != trip_ind;
  update |= h_setup.trip_constr != trip_constr;
  update |= h_setup.trip_mass != trip_mass;

  update |= h_setup.nquad != nquad;
  update |= h_setup.quad_ind != quad_ind;
  update |= h_setup.quad_constr != quad_constr;
  update |= h_setup.quad_mass != quad_mass;

  update |= h_setup.shake_tol != shake_tol;
  update |= h_setup.max_niter != max_niter;

  update |= h_setup.stride != stride;
  update |= h_setup.stride2 != stride*2;
  update |= h_setup.xyz0 != xyz0;
  update |= h_setup.xyz1 != xyz1;
  update |= h_setup.xyz2 != xyz2;

  if (update) {

    h_setup.nsolvent = nsolvent;
    h_setup.solvent_ind = solvent_ind;
    h_setup.mO_div_mH2O = mO_div_mH2O;
    h_setup.mH_div_mH2O = mH_div_mH2O;
    h_setup.ra = ra;
    h_setup.rc = rc;
    h_setup.rb = rb;
    h_setup.ra_inv = ra_inv;
    h_setup.rc2 = rc2;
    
    h_setup.npair = npair;
    h_setup.pair_ind = pair_ind;
    h_setup.pair_constr = pair_constr;
    h_setup.pair_mass = pair_mass;
    
    h_setup.ntrip = ntrip;
    h_setup.trip_ind = trip_ind;
    h_setup.trip_constr = trip_constr;
    h_setup.trip_mass = trip_mass;
    
    h_setup.nquad = nquad;
    h_setup.quad_ind = quad_ind;
    h_setup.quad_constr = quad_constr;
    h_setup.quad_mass = quad_mass;
    
    h_setup.shake_tol = shake_tol;
    h_setup.max_niter = max_niter;

    h_setup.stride = stride;
    h_setup.stride2 = stride*2;
    h_setup.xyz0 = xyz0;
    h_setup.xyz1 = xyz1;
    h_setup.xyz2 = xyz2;

    cudaCheck(cudaMemcpyToSymbolAsync(d_setup, &h_setup, sizeof(HoloConstSettings_t),
				      0, cudaMemcpyHostToDevice, stream));
  }

}

//
// Setups solvent_ind -table
//
void HoloConst::set_solvent_ind(int nsolvent, int3 *h_solvent_ind) {

  this->nsolvent = nsolvent;

  reallocate<int3>(&solvent_ind, &solvent_ind_len, nsolvent, 1.5f);
  copy_HtoD<int3>(h_solvent_ind, solvent_ind, nsolvent);

}

//
// Setups pair_ind -table
//
void HoloConst::set_pair_ind(int npair, int2 *h_pair_ind,
			     double *h_pair_constr, double *h_pair_mass) {

  this->npair = npair;

  reallocate<int2>(&pair_ind, &pair_ind_len, npair, 1.5f);
  copy_HtoD<int2>(h_pair_ind, pair_ind, npair);

  reallocate<double>(&pair_constr, &pair_constr_len, npair, 1.5f);
  copy_HtoD<double>(h_pair_constr, pair_constr, npair);

  reallocate<double>(&pair_mass, &pair_mass_len, npair*2, 1.5f);
  copy_HtoD<double>(h_pair_mass, pair_mass, npair*2);

}

//
// Setups trip_ind -table
//
void HoloConst::set_trip_ind(int ntrip, int3 *h_trip_ind,
			     double *h_trip_constr, double *h_trip_mass) {

  this->ntrip = ntrip;

  reallocate<int3>(&trip_ind, &trip_ind_len, ntrip, 1.5f);
  copy_HtoD<int3>(h_trip_ind, trip_ind, ntrip);

  reallocate<double>(&trip_constr, &trip_constr_len, ntrip*2, 1.5f);
  copy_HtoD<double>(h_trip_constr, trip_constr, ntrip*2);

  reallocate<double>(&trip_mass, &trip_mass_len, ntrip*5, 1.5f);
  copy_HtoD<double>(h_trip_mass, trip_mass, ntrip*5);

}

//
// Setups quad_ind -table
//
void HoloConst::set_quad_ind(int nquad, int4 *h_quad_ind,
			     double *h_quad_constr, double *h_quad_mass) {

  this->nquad = nquad;

  reallocate<int4>(&quad_ind, &quad_ind_len, nquad, 1.5f);
  copy_HtoD<int4>(h_quad_ind, quad_ind, nquad);

  reallocate<double>(&quad_constr, &quad_constr_len, nquad*3, 1.5f);
  copy_HtoD<double>(h_quad_constr, quad_constr, nquad*3);

  reallocate<double>(&quad_mass, &quad_mass_len, nquad*7, 1.5f);
  copy_HtoD<double>(h_quad_mass, quad_mass, nquad*7);

}

//
// Setup texture references for xyz0 and xyz1
//
void HoloConst::setup_textures(double *xyz0, double *xyz1, int stride) {

  assert(xyz0 != NULL);
  assert(xyz1 != NULL);

  // Unbind texture
  if (xyz0_texref_pointer != (int2 *)xyz0 || stride != texref_stride) {
    cudaCheck(cudaUnbindTexture(xyz0_texref));
    xyz0_texref_pointer = NULL;
  }
  if (xyz0_texref_pointer == NULL) {
    // Bind texture
    xyz0_texref.normalized = 0;
    xyz0_texref.filterMode = cudaFilterModePoint;
    xyz0_texref.addressMode[0] = cudaAddressModeClamp;
    xyz0_texref.channelDesc.x = 32;
    xyz0_texref.channelDesc.y = 32;
    xyz0_texref.channelDesc.z = 0;
    xyz0_texref.channelDesc.w = 0;
    xyz0_texref.channelDesc.f = cudaChannelFormatKindUnsigned;
    cudaCheck(cudaBindTexture(NULL, xyz0_texref, (int2 *)xyz0, stride*3*sizeof(int2)));
    xyz0_texref_pointer = (int2 *)xyz0;
  }

  // Unbind texture
  if (xyz1_texref_pointer != (int2 *)xyz1 || stride != texref_stride) {
    cudaCheck(cudaUnbindTexture(xyz1_texref));
    xyz1_texref_pointer = NULL;
  }
  if (xyz1_texref_pointer == NULL) {
    // Bind texture
    xyz1_texref.normalized = 0;
    xyz1_texref.filterMode = cudaFilterModePoint;
    xyz1_texref.addressMode[0] = cudaAddressModeClamp;
    xyz1_texref.channelDesc.x = 32;
    xyz1_texref.channelDesc.y = 32;
    xyz1_texref.channelDesc.z = 0;
    xyz1_texref.channelDesc.w = 0;
    xyz1_texref.channelDesc.f = cudaChannelFormatKindUnsigned;
    cudaCheck(cudaBindTexture(NULL, xyz1_texref, (int2 *)xyz1, stride*3*sizeof(int2)));
    xyz1_texref_pointer = (int2 *)xyz1;
  }

  texref_stride = stride;
}

//
// Apply constraints
//
void HoloConst::apply(cudaXYZ<double> *xyz0, cudaXYZ<double> *xyz1, cudaStream_t stream) {

  assert(xyz0->match(xyz1));


  int stride = xyz0->stride;

  update_setup(stride, xyz0->data, xyz1->data, xyz1->data);

  if (use_textures) setup_textures(xyz0->data, xyz1->data, stride);

  int nthread = 128;
  int nblock = (nsolvent + npair + ntrip + nquad - 1)/nthread + 1;
  all_kernels<<< nblock, nthread, 0, stream >>>();
  cudaCheck(cudaGetLastError());


}
