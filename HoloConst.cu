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
  int npair_type;
  int2* pair_ind;
  bond_t* pair_indtype;
  double *pair_constr;
  double *pair_mass;

  // Trips
  int ntrip;
  int ntrip_type;
  int3* trip_ind;
  angle_t* trip_indtype;
  double *trip_constr;
  double *trip_mass;

  // Quads
  int nquad;
  int nquad_type;
  int4* quad_ind;
  dihe_t* quad_indtype;
  double *quad_constr;
  double *quad_mass;

  // Buffer for all constraints and masses
  int ntot_type;
  double *constr_mass;

  double shake_tol;
  int max_niter;

  const double* __restrict__ xyz[2][3];
  double* __restrict__ xyz2[3];
};

static HoloConstSettings_t h_setup;
static __constant__ HoloConstSettings_t d_setup;

// x0 = xyz_texref[0][0]
// y0 = xyz_texref[0][1]
// z0 = xyz_texref[0][2]
// x1 = xyz_texref[1][0]
// y1 = xyz_texref[1][1]
// z1 = xyz_texref[1][2]
static texture<int2, 1, cudaReadModeElementType> xyz_texref00;
static texture<int2, 1, cudaReadModeElementType> xyz_texref01;
static texture<int2, 1, cudaReadModeElementType> xyz_texref02;
static texture<int2, 1, cudaReadModeElementType> xyz_texref10;
static texture<int2, 1, cudaReadModeElementType> xyz_texref11;
static texture<int2, 1, cudaReadModeElementType> xyz_texref12;

texture<int2, 1, cudaReadModeElementType>& get_xyz_texref(const int i, const int j) {
  switch(i) {
  case 0:
    switch(j) {
    case 0:
      return xyz_texref00;
    case 1:
      return xyz_texref01;
    case 2:
      return xyz_texref02;
    default:
      std::cerr << "get_xyz_texref, index out of bounds: i=" << i << " j=" << j << std::endl;
      exit(1);
    }
  case 1:
    switch(j) {
    case 0:
      return xyz_texref10;
    case 1:
      return xyz_texref11;
    case 2:
      return xyz_texref12;
    default:
      std::cerr << "get_xyz_texref, index out of bounds: i=" << i << " j=" << j << std::endl;
      exit(1);
    }
  default:
    std::cerr << "get_xyz_texref, index out of bounds: i=" << i << " j=" << j << std::endl;
    exit(1);
  }
}

static int2* xyz_texref_pointer[2][3];
static int texref_size[2][3];

template <int i, int j>
__forceinline__ __device__ double load_coord(const int ind) {
#if __CUDA_ARCH__ < 350
  int2 val;
  if (i == 0) {
    if (j == 0) {
      val = tex1Dfetch(xyz_texref00, ind);
    } else if (j == 1) {
      val = tex1Dfetch(xyz_texref01, ind);
    } else if (j == 2) {
      val = tex1Dfetch(xyz_texref02, ind);
    }
  } else if (i == 1) {
    if (j == 0) {
      val = tex1Dfetch(xyz_texref10, ind);
    } else if (j == 1) {
      val = tex1Dfetch(xyz_texref11, ind);
    } else if (j == 2) {
      val = tex1Dfetch(xyz_texref12, ind);
    }
  }
  return __hiloint2double(val.y, val.x);
#else
  return __ldg(&d_setup.xyz[i][j][ind]);
#endif
}

template<bool use_indexed>
__forceinline__ __device__ void pair_calc(const int imol,
					  const double* __restrict__ sh_constr,
					  const double* __restrict__ sh_mass) {
  int i, j, itype;
  if (use_indexed) {
    bond_t bond = d_setup.pair_indtype[imol];
    i     = bond.i;
    j     = bond.j;
    itype = bond.itype;
  } else {
    int2 ind = d_setup.pair_ind[imol];
    i = ind.x;
    j = ind.y;
    itype = imol;
  }

  // Load coordinates
  double x0i = load_coord<0, 0>(i);
  double y0i = load_coord<0, 1>(i);
  double z0i = load_coord<0, 2>(i);
  double x0j = load_coord<0, 0>(j);
  double y0j = load_coord<0, 1>(j);
  double z0j = load_coord<0, 2>(j);

  double x1i = load_coord<1, 0>(i);
  double y1i = load_coord<1, 1>(i);
  double z1i = load_coord<1, 2>(i);
  double x1j = load_coord<1, 0>(j);
  double y1j = load_coord<1, 1>(j);
  double z1j = load_coord<1, 2>(j);

  double xpij = x1i - x1j;
  double ypij = y1i - y1j;
  double zpij = z1i - z1j;
  double rijsq = xpij*xpij + ypij*ypij + zpij*zpij;

  double diff;
  if (use_indexed) {
    diff = sh_constr[itype] - rijsq;
  } else {
    diff = d_setup.pair_constr[itype] - rijsq;
  }

  double xrij = x0i - x0j;
  double yrij = y0i - y0j;
  double zrij = z0i - z0j;

  double rrijsq = xrij*xrij + yrij*yrij + zrij*zrij;
  double rijrijp = xrij*xpij  + yrij*ypij  + zrij*zpij;
  double lambda = 2.0*(-rijrijp + sqrt(rijrijp*rijrijp+rrijsq*diff))/(rrijsq);

  double hmassi_val, hmassj_val;
  if (use_indexed) {
    hmassi_val = sh_mass[itype*2];
    hmassj_val = sh_mass[itype*2+1];
  } else {
    hmassi_val = d_setup.pair_mass[itype*2];
    hmassj_val = d_setup.pair_mass[itype*2+1];
  }

  x1i += hmassi_val*lambda*xrij;
  y1i += hmassi_val*lambda*yrij;
  z1i += hmassi_val*lambda*zrij;
  x1j -= hmassj_val*lambda*xrij;
  y1j -= hmassj_val*lambda*yrij;
  z1j -= hmassj_val*lambda*zrij;

  // Store results
  d_setup.xyz2[0][i] = x1i;
  d_setup.xyz2[1][i] = y1i;
  d_setup.xyz2[2][i] = z1i;
  d_setup.xyz2[0][j] = x1j;
  d_setup.xyz2[1][j] = y1j;
  d_setup.xyz2[2][j] = z1j;
}

template<bool use_indexed>
__forceinline__ __device__ void trip_calc(const int imol,
					  const double* __restrict__ sh_constr,
					  const double* __restrict__ sh_mass) {
  int i, j, k, itype;
  if (use_indexed) {
    angle_t angle = d_setup.trip_indtype[imol];
    i     = angle.i;
    j     = angle.j;
    k     = angle.k;
    itype = angle.itype;
  } else {
    int3 ind = d_setup.trip_ind[imol];
    i = ind.x;
    j = ind.y;
    k = ind.z;
    itype = imol;
  }

  // Load coordinates
  double x0i = load_coord<0, 0>(i);
  double y0i = load_coord<0, 1>(i);
  double z0i = load_coord<0, 2>(i);
  double x0j = load_coord<0, 0>(j);
  double y0j = load_coord<0, 1>(j);
  double z0j = load_coord<0, 2>(j);
  double x0k = load_coord<0, 0>(k);
  double y0k = load_coord<0, 1>(k);
  double z0k = load_coord<0, 2>(k);

  double x1i = load_coord<1, 0>(i);
  double y1i = load_coord<1, 1>(i);
  double z1i = load_coord<1, 2>(i);
  double x1j = load_coord<1, 0>(j);
  double y1j = load_coord<1, 1>(j);
  double z1j = load_coord<1, 2>(j);
  double x1k = load_coord<1, 0>(k);
  double y1k = load_coord<1, 1>(k);
  double z1k = load_coord<1, 2>(k);

  double xrij = x0i - x0j;
  double yrij = y0i - y0j;
  double zrij = z0i - z0j;
  double xrik = x0i - x0k;
  double yrik = y0i - y0k;
  double zrik = z0i - z0k;

  double rrijsq = xrij*xrij + yrij*yrij + zrij*zrij;
  double rriksq = xrik*xrik + yrik*yrik + zrik*zrik;
  double rijrik = xrij*xrik + yrij*yrik + zrij*zrik;

  double mmi, mmj, mmk, mij, mik;
  if (use_indexed) {
    mmi = sh_mass[itype*5];
    mmj = sh_mass[itype*5+1];
    mmk = sh_mass[itype*5+2];
    mij = sh_mass[itype*5+3];
    mik = sh_mass[itype*5+4];
  } else {
    mmi = d_setup.trip_mass[itype*5];
    mmj = d_setup.trip_mass[itype*5+1];
    mmk = d_setup.trip_mass[itype*5+2];
    mij = d_setup.trip_mass[itype*5+3];
    mik = d_setup.trip_mass[itype*5+4];
  }

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
  double dij, dik;
  if (use_indexed) {
    dij = sh_constr[itype*2]   - rijsq;
    dik = sh_constr[itype*2+1] - riksq;
  } else {
    dij = d_setup.trip_constr[itype*2]   - rijsq;
    dik = d_setup.trip_constr[itype*2+1] - riksq;
  }
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

  d_setup.xyz2[0][i] = x1i;
  d_setup.xyz2[1][i] = y1i;
  d_setup.xyz2[2][i] = z1i;
  d_setup.xyz2[0][j] = x1j;
  d_setup.xyz2[1][j] = y1j;
  d_setup.xyz2[2][j] = z1j;
  d_setup.xyz2[0][k] = x1k;
  d_setup.xyz2[1][k] = y1k;
  d_setup.xyz2[2][k] = z1k;
}

template<bool use_indexed>
__forceinline__ __device__ void quad_calc(const int imol,
					  const double* __restrict__ sh_constr,
					  const double* __restrict__ sh_mass) {
  int i, j, k, l, itype;
  if (use_indexed) {
    dihe_t dihe = d_setup.quad_indtype[imol];
    i     = dihe.i;
    j     = dihe.j;
    k     = dihe.k;
    l     = dihe.l;
    itype = dihe.itype;
  } else {
    int4 ind = d_setup.quad_ind[imol];
    i = ind.x;
    j = ind.y;
    k = ind.z;
    l = ind.w;
    itype = imol;
  }

  // Load coordinates
  double x0i = load_coord<0, 0>(i);
  double y0i = load_coord<0, 1>(i);
  double z0i = load_coord<0, 2>(i);
  double x0j = load_coord<0, 0>(j);
  double y0j = load_coord<0, 1>(j);
  double z0j = load_coord<0, 2>(j);
  double x0k = load_coord<0, 0>(k);
  double y0k = load_coord<0, 1>(k);
  double z0k = load_coord<0, 2>(k);
  double x0l = load_coord<0, 0>(l);
  double y0l = load_coord<0, 1>(l);
  double z0l = load_coord<0, 2>(l);

  double x1i = load_coord<1, 0>(i);
  double y1i = load_coord<1, 1>(i);
  double z1i = load_coord<1, 2>(i);
  double x1j = load_coord<1, 0>(j);
  double y1j = load_coord<1, 1>(j);
  double z1j = load_coord<1, 2>(j);
  double x1k = load_coord<1, 0>(k);
  double y1k = load_coord<1, 1>(k);
  double z1k = load_coord<1, 2>(k);
  double x1l = load_coord<1, 0>(l);
  double y1l = load_coord<1, 1>(l);
  double z1l = load_coord<1, 2>(l);

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

  double mmi, mmj, mmk, mml, mij, mik, mil;
  if (use_indexed) {
    mmi = sh_mass[itype*7];
    mmj = sh_mass[itype*7+1];
    mmk = sh_mass[itype*7+2];
    mml = sh_mass[itype*7+3];
    mij = sh_mass[itype*7+4];
    mik = sh_mass[itype*7+5];
    mil = sh_mass[itype*7+6];
  } else {
    mmi = d_setup.quad_mass[itype*7];
    mmj = d_setup.quad_mass[itype*7+1];
    mmk = d_setup.quad_mass[itype*7+2];
    mml = d_setup.quad_mass[itype*7+3];
    mij = d_setup.quad_mass[itype*7+4];
    mik = d_setup.quad_mass[itype*7+5];
    mil = d_setup.quad_mass[itype*7+6];
  }

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
  double dij, dik, dil;
  if (use_indexed) {
    dij = sh_constr[itype*3]   - rijsq;
    dik = sh_constr[itype*3+1] - riksq;
    dil = sh_constr[itype*3+2] - rilsq;
  } else {
    dij = d_setup.quad_constr[itype*3]   - rijsq;
    dik = d_setup.quad_constr[itype*3+1] - riksq;
    dil = d_setup.quad_constr[itype*3+2] - rilsq;
  }
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

  d_setup.xyz2[0][i] = x1i;
  d_setup.xyz2[1][i] = y1i;
  d_setup.xyz2[2][i] = z1i;
  d_setup.xyz2[0][j] = x1j;
  d_setup.xyz2[1][j] = y1j;
  d_setup.xyz2[2][j] = z1j;
  d_setup.xyz2[0][k] = x1k;
  d_setup.xyz2[1][k] = y1k;
  d_setup.xyz2[2][k] = z1k;
  d_setup.xyz2[0][l] = x1l;
  d_setup.xyz2[1][l] = y1l;
  d_setup.xyz2[2][l] = z1l;
}

__forceinline__ __device__ void solvent_calc(int imol) {

    int3 ind = d_setup.solvent_ind[imol];

    // Load coordinates
    double x0i = load_coord<0, 0>(ind.x);
    double y0i = load_coord<0, 1>(ind.x);
    double z0i = load_coord<0, 2>(ind.x);
    double x0j = load_coord<0, 0>(ind.y);
    double y0j = load_coord<0, 1>(ind.y);
    double z0j = load_coord<0, 2>(ind.y);
    double x0k = load_coord<0, 0>(ind.z);
    double y0k = load_coord<0, 1>(ind.z);
    double z0k = load_coord<0, 2>(ind.z);

    double x1i = load_coord<1, 0>(ind.x);
    double y1i = load_coord<1, 1>(ind.x);
    double z1i = load_coord<1, 2>(ind.x);
    double x1j = load_coord<1, 0>(ind.y);
    double y1j = load_coord<1, 1>(ind.y);
    double z1j = load_coord<1, 2>(ind.y);
    double x1k = load_coord<1, 0>(ind.z);
    double y1k = load_coord<1, 1>(ind.z);
    double z1k = load_coord<1, 2>(ind.z);

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

    d_setup.xyz2[0][ind.x] = xcm + trans11 * xa3p + trans12 * ya3p + trans13 * za3p;
    d_setup.xyz2[1][ind.x] = ycm + trans21 * xa3p + trans22 * ya3p + trans23 * za3p;
    d_setup.xyz2[2][ind.x] = zcm + trans31 * xa3p + trans32 * ya3p + trans33 * za3p;
    d_setup.xyz2[0][ind.y] = xcm + trans11 * xb3p + trans12 * yb3p + trans13 * zb3p;
    d_setup.xyz2[1][ind.y] = ycm + trans21 * xb3p + trans22 * yb3p + trans23 * zb3p;
    d_setup.xyz2[2][ind.y] = zcm + trans31 * xb3p + trans32 * yb3p + trans33 * zb3p;
    d_setup.xyz2[0][ind.z] = xcm + trans11 * xc3p + trans12 * yc3p + trans13 * zc3p;
    d_setup.xyz2[1][ind.z] = ycm + trans21 * xc3p + trans22 * yc3p + trans23 * zc3p;
    d_setup.xyz2[2][ind.z] = zcm + trans31 * xc3p + trans32 * yc3p + trans33 * zc3p;
}

template<bool use_indexed>
__global__ void all_kernels() {
  // Shared memory, only used when use_indexed = true
  // Requires: npair_type*sizeof(double)*3 + ntrip_type*sizeof(double)*7 + nquad_type*sizeof(double)*10
  extern __shared__ double sh_constr_mass[];
  double* __restrict__ sh_pair_constr;
  double* __restrict__ sh_pair_mass;
  double* __restrict__ sh_trip_constr;
  double* __restrict__ sh_trip_mass;
  double* __restrict__ sh_quad_constr;
  double* __restrict__ sh_quad_mass;
  if (use_indexed) {
    int pos = 0;
    sh_pair_constr = &sh_constr_mass[pos];
    pos += d_setup.npair_type;
    sh_pair_mass   = &sh_constr_mass[pos];
    pos += d_setup.npair_type*2;

    sh_trip_constr = &sh_constr_mass[pos];
    pos += d_setup.ntrip_type*2;
    sh_trip_mass   = &sh_constr_mass[pos];
    pos += d_setup.ntrip_type*5;

    sh_quad_constr = &sh_constr_mass[pos];
    pos += d_setup.nquad_type*3;
    sh_quad_mass   = &sh_constr_mass[pos];

    int t;
    t = threadIdx.x;
    while (t < d_setup.ntot_type) {
      sh_constr_mass[t] = d_setup.constr_mass[t];
      t += blockDim.x;
    }
    __syncthreads();
  }

  const int imol = threadIdx.x + blockDim.x*blockIdx.x;

  if (imol < d_setup.nsolvent) {
    solvent_calc(imol);
  } else if (imol < d_setup.nsolvent + d_setup.npair) {
    pair_calc<use_indexed>(imol - d_setup.nsolvent,
			   sh_pair_constr, sh_pair_mass);
  } else if (imol < d_setup.nsolvent + d_setup.npair + d_setup.ntrip) {
    trip_calc<use_indexed>(imol - d_setup.nsolvent - d_setup.npair,
    			   sh_trip_constr, sh_trip_mass);
  } else if (imol < d_setup.nsolvent + d_setup.npair + d_setup.ntrip + d_setup.nquad) {
    quad_calc<use_indexed>(imol - d_setup.nsolvent - d_setup.npair - d_setup.ntrip,
			   sh_quad_constr, sh_quad_mass);
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
  pair_indtype = NULL;
  npair_type = 0;
  //pair_constr_len = 0;
  pair_constr = NULL;
  //pair_mass_len = 0;
  pair_mass = NULL;

  ntrip = 0;
  trip_ind_len = 0;
  trip_ind = NULL;
  trip_indtype = NULL;
  ntrip_type = 0;
  //trip_constr_len = 0;
  trip_constr = NULL;
  //trip_mass_len = 0;
  trip_mass = NULL;

  nquad = 0;
  quad_ind_len = 0;
  quad_ind = NULL;
  quad_indtype = NULL;
  nquad_type = 0;
  //quad_constr_len = 0;
  quad_constr = NULL;
  //quad_mass_len = 0;
  quad_mass = NULL;

  ntot_type = 0;
  constr_mass_len = 0;
  constr_mass = NULL;

  for (int i=0;i < 2;i++) {
    for (int j=0;j < 3;j++) {
      xyz_texref_pointer[i][j] = NULL;
      texref_size[i][j] = 0;
    }
  }

  if (get_cuda_arch() < 350) {
    use_textures = true;
  } else {
    use_textures = false;
  }

  use_indexed = false;
  use_settle = false;
}

//
// Class destructor
//
HoloConst::~HoloConst() {

  for (int i=0;i < 2;i++)
    for (int j=0;j < 3;j++)
      if (xyz_texref_pointer[i][j] != NULL) {
	cudaCheck(cudaUnbindTexture(get_xyz_texref(i,j)));
	xyz_texref_pointer[i][j] = NULL;
      }

  if (solvent_ind != NULL) deallocate<int3>(&solvent_ind);

  if (pair_ind != NULL) deallocate<int2>(&pair_ind);
  if (pair_indtype != NULL) deallocate<bond_t>(&pair_indtype);
  //if (pair_constr != NULL) deallocate<double>(&pair_constr);
  //if (pair_mass != NULL) deallocate<double>(&pair_mass);

  if (trip_ind != NULL) deallocate<int3>(&trip_ind);
  if (trip_indtype != NULL) deallocate<angle_t>(&trip_indtype);
  //if (trip_constr != NULL) deallocate<double>(&trip_constr);
  //if (trip_mass != NULL) deallocate<double>(&trip_mass);

  if (quad_ind != NULL) deallocate<int4>(&quad_ind);
  if (quad_indtype != NULL) deallocate<dihe_t>(&quad_indtype);
  //if (quad_constr != NULL) deallocate<double>(&quad_constr);
  //if (quad_mass != NULL) deallocate<double>(&quad_mass);

  if (constr_mass != NULL) deallocate<double>(&constr_mass);

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
// Setup SETTLE parameters
//
// massP = mass of heavy atoms
// massH = mass of light (hydrogen) atoms
// rPHsq = squared distance between P and H
// rHHsq = squared distance between H and H
//
void HoloConst::setup_settle_parameters(const int nsettle,
					const double* h_massP, const double* h_massH,
					const double* h_rPHsq, const double* h_rHHsq) {

  use_settle = true;
  //this->nsettle = nsettle;

}

void HoloConst::realloc_constr_mass(const int npair_type, const int ntrip_type, const int nquad_type) {
  ntot_type = npair_type*3 + ntrip_type*7 + nquad_type*10;
  reallocate<double>(&constr_mass, &constr_mass_len, ntot_type, 1.0f);
  int pos = 0;
  pair_constr = &constr_mass[pos];
  pos += npair_type;
  pair_mass   = &constr_mass[pos];
  pos += npair_type*2;

  trip_constr = &constr_mass[pos];
  pos += ntrip_type*2;
  trip_mass   = &constr_mass[pos];
  pos += ntrip_type*5;
    
  quad_constr = &constr_mass[pos];
  pos += nquad_type*3;
  quad_mass   = &constr_mass[pos];
}

//
// Setups all ind, mass, and constr arrays
//
void HoloConst::setup_ind_mass_constr(const int npair, const int2 *h_pair_ind,
				      const double *h_pair_constr, const double *h_pair_mass,
				      const int ntrip, const int3 *h_trip_ind,
				      const double *h_trip_constr, const double *h_trip_mass,
				      const int nquad, const int4 *h_quad_ind,
				      const double *h_quad_constr, const double *h_quad_mass,
				      const int nsolvent, const int3 *h_solvent_ind) {
  use_indexed = false;
  realloc_constr_mass(npair, ntrip, nquad);
  // Copy ind, mass, and constr from CPU to GPU
  set_pair(npair, h_pair_ind, h_pair_constr, h_pair_mass);
  set_trip(ntrip, h_trip_ind, h_trip_constr, h_trip_mass);
  set_quad(nquad, h_quad_ind, h_quad_constr, h_quad_mass);
  set_solvent(nsolvent, h_solvent_ind);

}

/*
//
// Setups all ind, mass, and constr arrays using local->global mapping
//
void HoloConst::setup_ind_mass_constr(const int npair, const int2 *global_pair_ind,
				      const double *global_pair_constr, const double *global_pair_mass,
				      const int ntrip, const int3 *global_trip_ind,
				      const double *global_trip_constr, const double *global_trip_mass,
				      const int nquad, const int4 *global_quad_ind,
				      const double *global_quad_constr, const double *global_quad_mass,
				      const int nsolvent, const int3 *global_solvent_ind,
				      const int* loc2glo) {
  use_indexed = false;
  // Copy ind, mass, and constr from CPU to GPU
  set_pair(npair, global_pair_ind, global_pair_constr, global_pair_mass, loc2glo);
  set_trip(ntrip, global_trip_ind, global_trip_constr, global_trip_mass, loc2glo);
  set_quad(nquad, global_quad_ind, global_quad_constr, global_quad_mass, loc2glo);
  set_solvent(nsolvent, global_solvent_ind, loc2glo);
}
*/

//
// Setup indexed
//
void HoloConst::setup_indexed(const int npair, const bond_t* h_pair_indtype,
			      const int npair_type, const double* h_pair_constr, const double* h_pair_mass,
			      const int ntrip, const angle_t* h_trip_indtype,
			      const int ntrip_type, const double* h_trip_constr, const double* h_trip_mass,
			      const int nquad, const dihe_t* h_quad_indtype,
			      const int nquad_type, const double* h_quad_constr, const double* h_quad_mass,
			      const int nsolvent, const int3* h_solvent_ind) {
  use_indexed = true;
  realloc_constr_mass(npair_type, ntrip_type, nquad_type);
  set_pair(npair, h_pair_indtype, npair_type, h_pair_constr, h_pair_mass);
  set_trip(ntrip, h_trip_indtype, ntrip_type, h_trip_constr, h_trip_mass);
  set_quad(nquad, h_quad_indtype, nquad_type, h_quad_constr, h_quad_mass);
  set_solvent(nsolvent, h_solvent_ind);
}

/*
//
// Setup indexed
//
void HoloConst::setup_indexed(const int npair, const bond_t* h_pair_indtype,
			      const int npair_type, const double* h_pair_constr, const double* h_pair_mass,
			      const int nquad, const dihe_t* h_quad_indtype,
			      const int nquad_type, const double* h_quad_constr, const double* h_quad_mass,
			      const int nsettle, const angle_t* h_settle_indtype) {
  assert(use_settle);
  use_indexed = true;
  realloc_constr_mass(npair_type, ntrip_type, nquad_type);
  set_pair(npair, h_pair_indtype, npair_type, h_pair_constr, h_pair_mass);
  set_quad(nquad, h_quad_indtype, nquad_type, h_quad_constr, h_quad_mass);
}
*/

//
// Updates h_setup and d_setup if neccessary
//
void HoloConst::update_setup(cudaXYZ<double>& xyz0, cudaXYZ<double>& xyz1, cudaXYZ<double>& xyz2,
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
  update |= h_setup.npair_type != npair_type;
  update |= h_setup.pair_ind != pair_ind;
  update |= h_setup.pair_indtype != pair_indtype;
  update |= h_setup.pair_constr != pair_constr;
  update |= h_setup.pair_mass != pair_mass;

  update |= h_setup.ntrip != ntrip;
  update |= h_setup.ntrip_type != ntrip_type;
  update |= h_setup.trip_ind != trip_ind;
  update |= h_setup.trip_indtype != trip_indtype;
  update |= h_setup.trip_constr != trip_constr;
  update |= h_setup.trip_mass != trip_mass;

  update |= h_setup.nquad != nquad;
  update |= h_setup.nquad_type != nquad_type;
  update |= h_setup.quad_ind != quad_ind;
  update |= h_setup.quad_indtype != quad_indtype;
  update |= h_setup.quad_constr != quad_constr;
  update |= h_setup.quad_mass != quad_mass;

  update |= h_setup.ntot_type != ntot_type;
  update |= h_setup.constr_mass != constr_mass;

  update |= h_setup.shake_tol != shake_tol;
  update |= h_setup.max_niter != max_niter;

  update |= h_setup.xyz[0][0] != xyz0.x();
  update |= h_setup.xyz[0][1] != xyz0.y();
  update |= h_setup.xyz[0][2] != xyz0.z();

  update |= h_setup.xyz[1][0] != xyz1.x();
  update |= h_setup.xyz[1][1] != xyz1.y();
  update |= h_setup.xyz[1][2] != xyz1.z();

  update |= h_setup.xyz2[0] != xyz2.x();
  update |= h_setup.xyz2[1] != xyz2.y();
  update |= h_setup.xyz2[2] != xyz2.z();

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
    h_setup.npair_type = npair_type;
    h_setup.pair_ind = pair_ind;
    h_setup.pair_indtype = pair_indtype;
    h_setup.pair_constr = pair_constr;
    h_setup.pair_mass = pair_mass;
    
    h_setup.ntrip = ntrip;
    h_setup.ntrip_type = ntrip_type;
    h_setup.trip_ind = trip_ind;
    h_setup.trip_indtype = trip_indtype;
    h_setup.trip_constr = trip_constr;
    h_setup.trip_mass = trip_mass;
    
    h_setup.nquad = nquad;
    h_setup.nquad_type = nquad_type;
    h_setup.quad_ind = quad_ind;
    h_setup.quad_indtype = quad_indtype;
    h_setup.quad_constr = quad_constr;
    h_setup.quad_mass = quad_mass;

    h_setup.ntot_type = ntot_type;
    h_setup.constr_mass = constr_mass;
    
    h_setup.shake_tol = shake_tol;
    h_setup.max_niter = max_niter;

    h_setup.xyz[0][0] = xyz0.x();
    h_setup.xyz[0][1] = xyz0.y();
    h_setup.xyz[0][2] = xyz0.z();

    h_setup.xyz[1][0] = xyz1.x();
    h_setup.xyz[1][1] = xyz1.y();
    h_setup.xyz[1][2] = xyz1.z();

    h_setup.xyz2[0] = xyz2.x();
    h_setup.xyz2[1] = xyz2.y();
    h_setup.xyz2[2] = xyz2.z();

    cudaCheck(cudaMemcpyToSymbolAsync(d_setup, &h_setup, sizeof(HoloConstSettings_t),
				      0, cudaMemcpyHostToDevice, stream));
  }

}

//
// Setups solvent_ind -table
//
void HoloConst::set_solvent(const int nsolvent, const int3 *h_solvent_ind) {

  this->nsolvent = nsolvent;

  if (nsolvent > 0) {
    reallocate<int3>(&solvent_ind, &solvent_ind_len, nsolvent, 1.5f);
    copy_HtoD<int3>(h_solvent_ind, solvent_ind, nsolvent);
  }
}

//
// Setups solvent_ind -table using local->global mapping
//
void HoloConst::set_solvent(const int nsolvent, const int3 *global_solvent_ind, const int *loc2glo) {

  this->nsolvent = nsolvent;

  if (nsolvent > 0) {
    reallocate<int3>(&solvent_ind, &solvent_ind_len, nsolvent, 1.5f);
    map_to_local_array<int3>(nsolvent, loc2glo, global_solvent_ind, solvent_ind);
  }
}

//
// Setups pair_ind -table
//
void HoloConst::set_pair(const int npair, const int2 *h_pair_ind,
			 const double *h_pair_constr, const double *h_pair_mass) {

  this->npair = npair;
  this->npair_type = npair;

  if (npair > 0) {
    reallocate<int2>(&pair_ind, &pair_ind_len, npair, 1.5f);
    copy_HtoD<int2>(h_pair_ind, pair_ind, npair);
  }

  if (npair > 0) {
    //reallocate<double>(&pair_constr, &pair_constr_len, npair, 1.5f);
    copy_HtoD<double>(h_pair_constr, pair_constr, npair);
  }

  if (npair > 0) {
    //reallocate<double>(&pair_mass, &pair_mass_len, npair*2, 1.5f);
    copy_HtoD<double>(h_pair_mass, pair_mass, npair*2);
  }
}

//
// Setups pair_ind -table
//
void HoloConst::set_pair(const int npair, const bond_t* h_pair_indtype,
			 const int npair_type, const double *h_pair_constr, const double *h_pair_mass) {

  this->npair = npair;
  this->npair_type = npair_type;
  
  if (npair > 0) {
    reallocate<bond_t>(&pair_indtype, &pair_ind_len, npair, 1.5f);
    copy_HtoD<bond_t>(h_pair_indtype, pair_indtype, npair);
  }

  if (npair_type > 0) {
    //reallocate<double>(&pair_constr, &pair_constr_len, npair_type, 1.5f);
    copy_HtoD<double>(h_pair_constr, pair_constr, npair_type);
  }

  if (npair_type > 0) {
    //reallocate<double>(&pair_mass, &pair_mass_len, npair_type*2, 1.5f);
    copy_HtoD<double>(h_pair_mass, pair_mass, npair_type*2);
  }
}

/*
//
// Setups pair_ind -table using local->global mapping
//
void HoloConst::set_pair(const int npair, const int2 *global_pair_ind,
			 const double *global_pair_constr, const double *global_pair_mass,
			 const int *loc2glo) {

  this->npair = npair;
  this->npair_type = npair;

  reallocate<int2>(&pair_ind, &pair_ind_len, npair, 1.5f);
  reallocate<double>(&pair_constr, &pair_constr_len, npair, 1.5f);
  reallocate<double>(&pair_mass, &pair_mass_len, npair*2, 1.5f);

  map_to_local_array<int2>(npair, loc2glo, global_pair_ind, pair_ind);
  map_to_local_array<double>(npair, loc2glo, global_pair_constr, pair_constr);
  map_to_local_array<double2>(npair, loc2glo, (double2 *)global_pair_mass, (double2 *)pair_mass);
}
*/

//
// Setups trip_ind -table
//
void HoloConst::set_trip(const int ntrip, const int3 *h_trip_ind,
			 const double *h_trip_constr, const double *h_trip_mass) {

  this->ntrip = ntrip;
  this->ntrip_type = ntrip;

  if (ntrip > 0) {
    reallocate<int3>(&trip_ind, &trip_ind_len, ntrip, 1.5f);
    copy_HtoD<int3>(h_trip_ind, trip_ind, ntrip);
  }

  if (ntrip > 0) {
    //reallocate<double>(&trip_constr, &trip_constr_len, ntrip*2, 1.5f);
    copy_HtoD<double>(h_trip_constr, trip_constr, ntrip*2);
  }

  if (ntrip > 0) {
    //reallocate<double>(&trip_mass, &trip_mass_len, ntrip*5, 1.5f);
    copy_HtoD<double>(h_trip_mass, trip_mass, ntrip*5);
  }
}

//
// Setups trip_ind -table
//
void HoloConst::set_trip(const int ntrip, const angle_t* h_trip_indtype,
			 const int ntrip_type, const double *h_trip_constr, const double *h_trip_mass) {

  this->ntrip = ntrip;
  this->ntrip_type = ntrip_type;

  if (ntrip > 0) {
    reallocate<angle_t>(&trip_indtype, &trip_ind_len, ntrip, 1.5f);
    copy_HtoD<angle_t>(h_trip_indtype, trip_indtype, ntrip);
  }

  if (ntrip_type > 0) {
    //reallocate<double>(&trip_constr, &trip_constr_len, ntrip_type*2, 1.5f);
    copy_HtoD<double>(h_trip_constr, trip_constr, ntrip_type*2);
  }

  if (ntrip_type > 0) {
    //reallocate<double>(&trip_mass, &trip_mass_len, ntrip_type*5, 1.5f);
    copy_HtoD<double>(h_trip_mass, trip_mass, ntrip_type*5);
  }
}

struct double5 {
  double x1, x2, x3, x4, x5;
};

/*
//
// Setups trip_ind -table using local->global mapping
//
void HoloConst::set_trip(const int ntrip, const int3 *global_trip_ind,
			 const double *global_trip_constr, const double *global_trip_mass,
			 const int *loc2glo) {

  this->ntrip = ntrip;
  this->ntrip_type = ntrip;

  reallocate<int3>(&trip_ind, &trip_ind_len, ntrip, 1.5f);
  reallocate<double>(&trip_constr, &trip_constr_len, ntrip*2, 1.5f);
  reallocate<double>(&trip_mass, &trip_mass_len, ntrip*5, 1.5f);

  map_to_local_array<int3>(ntrip, loc2glo, global_trip_ind, trip_ind);
  map_to_local_array<double2>(ntrip, loc2glo, (double2 *)global_trip_constr, (double2 *)trip_constr);
  map_to_local_array<double5>(ntrip, loc2glo, (double5 *)global_trip_mass, (double5 *)trip_mass);
}
*/

//
// Setups quad_ind -table
//
void HoloConst::set_quad(const int nquad, const int4 *h_quad_ind,
			 const double *h_quad_constr, const double *h_quad_mass) {

  this->nquad = nquad;
  this->nquad_type = nquad;

  if (nquad > 0) {
    reallocate<int4>(&quad_ind, &quad_ind_len, nquad, 1.5f);
    copy_HtoD<int4>(h_quad_ind, quad_ind, nquad);
  }

  if (nquad > 0) {
    //reallocate<double>(&quad_constr, &quad_constr_len, nquad*3, 1.5f);
    copy_HtoD<double>(h_quad_constr, quad_constr, nquad*3);
  }

  if (nquad > 0) {
    //reallocate<double>(&quad_mass, &quad_mass_len, nquad*7, 1.5f);
    copy_HtoD<double>(h_quad_mass, quad_mass, nquad*7);
  }
}

//
// Setups quad_ind -table
//
void HoloConst::set_quad(const int nquad, const dihe_t* h_quad_indtype,
			 const int nquad_type, const double *h_quad_constr, const double *h_quad_mass) {

  this->nquad = nquad;
  this->nquad_type = nquad_type;

  if (nquad > 0) {
    reallocate<dihe_t>(&quad_indtype, &quad_ind_len, nquad, 1.5f);
    copy_HtoD<dihe_t>(h_quad_indtype, quad_indtype, nquad);
  }

  if (nquad_type > 0) {
    //reallocate<double>(&quad_constr, &quad_constr_len, nquad_type*3, 1.5f);
    copy_HtoD<double>(h_quad_constr, quad_constr, nquad_type*3);
  }

  if (nquad_type > 0) {
    //reallocate<double>(&quad_mass, &quad_mass_len, nquad_type*7, 1.5f);
    copy_HtoD<double>(h_quad_mass, quad_mass, nquad_type*7);
  }
}

struct double7 {
  double x1, x2, x3, x4, x5, x6, x7;
};

/*
//
// Setups quad_ind -table using local->global mapping
//
void HoloConst::set_quad(const int nquad, const int4 *global_quad_ind,
			 const double *global_quad_constr, const double *global_quad_mass,
			 const int *loc2glo) {

  this->nquad = nquad;
  this->nquad_type = nquad;

  reallocate<int4>(&quad_ind, &quad_ind_len, nquad, 1.5f);
  reallocate<double>(&quad_constr, &quad_constr_len, nquad*3, 1.5f);
  reallocate<double>(&quad_mass, &quad_mass_len, nquad*7, 1.5f);

  map_to_local_array<int4>(nquad, loc2glo, global_quad_ind, quad_ind);
  map_to_local_array<double3>(nquad, loc2glo, (double3 *)global_quad_constr, (double3 *)quad_constr);
  map_to_local_array<double7>(nquad, loc2glo, (double7 *)global_quad_mass, (double7 *)quad_mass);
}
*/

//
// Setup texture references for xyz0 and xyz1
//
void HoloConst::setup_textures(cudaXYZ<double>& xyz, int i) {
  assert(xyz.x() != NULL);
  assert(xyz.y() != NULL);
  assert(xyz.z() != NULL);

  double* xyzp[3] = {xyz.x(), xyz.y(), xyz.z()};
  for (int j=0;j < 3;j++) {
    // Unbind texture
    if (xyz_texref_pointer[i][j] != (int2 *)xyzp[j] || xyz.size() != texref_size[i][j]) {
      cudaCheck(cudaUnbindTexture(get_xyz_texref(i,j)));
      xyz_texref_pointer[i][j] = NULL;
      texref_size[i][j] = 0;
    }
    // Bind texture
    if (xyz_texref_pointer[i][j] == NULL) {
      xyz_texref_pointer[i][j] = (int2 *)xyzp[j];
      texref_size[i][j] = xyz.size();
      get_xyz_texref(i,j).normalized = 0;
      get_xyz_texref(i,j).filterMode = cudaFilterModePoint;
      get_xyz_texref(i,j).addressMode[0] = cudaAddressModeClamp;
      get_xyz_texref(i,j).channelDesc.x = 32;
      get_xyz_texref(i,j).channelDesc.y = 32;
      get_xyz_texref(i,j).channelDesc.z = 0;
      get_xyz_texref(i,j).channelDesc.w = 0;
      get_xyz_texref(i,j).channelDesc.f = cudaChannelFormatKindUnsigned;
      cudaCheck(cudaBindTexture(NULL, get_xyz_texref(i,j), xyz_texref_pointer[i][j],
				texref_size[i][j]*sizeof(int2)));
    }
  }
}

//
// Apply constraints: xyz0 = reference (input), xyz1 = constrained (input/output)
//
void HoloConst::apply(cudaXYZ<double>& xyz0, cudaXYZ<double>& xyz1, cudaStream_t stream) {
  assert(xyz0.match(xyz1));

  int ntot = nsolvent + npair + ntrip + nquad;

  if (ntot == 0) return;

  update_setup(xyz0, xyz1, xyz1);

  if (use_textures) {
    setup_textures(xyz0, 0);
    setup_textures(xyz1, 1);
  }

  int nthread = 128;
  int nblock = (ntot - 1)/nthread + 1;
  int shmem = 0;
  if (use_indexed) {
    shmem = ntot_type*sizeof(double);
    all_kernels<true> <<< nblock, nthread, shmem, stream >>>();
  } else {
    all_kernels<false> <<< nblock, nthread, shmem, stream >>>();
  }
  cudaCheck(cudaGetLastError());
}
