#include <iostream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "HoloConst.h"

//
// Runs the SETTLE algorithm on solvent molecules
// xyz0 = coordinates at time t
// xyz1 = coordinates at time t + delta t
//
__global__ void settle_solvent_kernel(const int nmol, const int3* __restrict__ solvent_ind,
				      const double mO_div_mH2O, const double mH_div_mH2O,
				      const double ra, const double rc, const double rb,
				      const double ra_inv, const double rc2,
				      const int stride,
				      const double* __restrict__ xyz0, double *xyz1) {

  // Index of the solvent molecule
  const int imol = threadIdx.x + blockDim.x*blockIdx.x;
  const int stride2 = stride*2;

  if (imol < nmol) {
    int3 ind = solvent_ind[imol];

    // Load coordinates
    double x0i = xyz0[ind.x];
    double y0i = xyz0[ind.x+stride];
    double z0i = xyz0[ind.x+stride2];
    double x0j = xyz0[ind.y];
    double y0j = xyz0[ind.y+stride];
    double z0j = xyz0[ind.y+stride2];
    double x0k = xyz0[ind.z];
    double y0k = xyz0[ind.z+stride];
    double z0k = xyz0[ind.z+stride2];

    double x1i = xyz1[ind.x];
    double y1i = xyz1[ind.x+stride];
    double z1i = xyz1[ind.x+stride2];
    double x1j = xyz1[ind.y];
    double y1j = xyz1[ind.y+stride];
    double z1j = xyz1[ind.y+stride2];
    double x1k = xyz1[ind.z];
    double y1k = xyz1[ind.z+stride];
    double z1k = xyz1[ind.z+stride2];

    //
    // Convert to primed coordinates
    //

    // Calculate center of mass for (x1, y1, z1)
    double xcm = x1i*mO_div_mH2O + (x1j + x1k)*mH_div_mH2O;
    double ycm = y1i*mO_div_mH2O + (y1j + y1k)*mH_div_mH2O;
    double zcm = z1i*mO_div_mH2O + (z1j + z1k)*mH_div_mH2O;

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

    double sinphi = za1p * ra_inv;
    double cosphi = sqrt(1.0 - sinphi*sinphi);
    double sinpsi = (zb1p - zc1p) / (rc2 * cosphi);
    double cospsi = sqrt(1.0 - sinpsi*sinpsi);

    double ya2p =   ra * cosphi;
    double xb2p = - rc * cospsi;
    double yb2p = - rb * cosphi - rc *sinpsi * sinphi;
    double yc2p = - rb * cosphi + rc *sinpsi * sinphi;

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

    xyz1[ind.x]         = xcm + trans11 * xa3p + trans12 * ya3p + trans13 * za3p;
    xyz1[ind.x+stride]  = ycm + trans21 * xa3p + trans22 * ya3p + trans23 * za3p;
    xyz1[ind.x+stride2] = zcm + trans31 * xa3p + trans32 * ya3p + trans33 * za3p;
    xyz1[ind.y]         = xcm + trans11 * xb3p + trans12 * yb3p + trans13 * zb3p;
    xyz1[ind.y+stride]  = ycm + trans21 * xb3p + trans22 * yb3p + trans23 * zb3p;
    xyz1[ind.y+stride2] = zcm + trans31 * xb3p + trans32 * yb3p + trans33 * zb3p;
    xyz1[ind.z]         = xcm + trans11 * xc3p + trans12 * yc3p + trans13 * zc3p;
    xyz1[ind.z+stride]  = ycm + trans21 * xc3p + trans22 * yc3p + trans23 * zc3p;
    xyz1[ind.z+stride2] = zcm + trans31 * xc3p + trans32 * yc3p + trans33 * zc3p;
  }

}

//
// Class creator
//
HoloConst::HoloConst() {
  nsolvent = 0;
  solvent_ind_len = 0;
  solvent_ind = NULL;
}

//
// Class destructor
//
HoloConst::~HoloConst() {
  if (solvent_ind != NULL) deallocate<int3>(&solvent_ind);
}

//
// Setup
//
void HoloConst::setup(double mO, double mH, double rOHsq, double rHHsq) {

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
// Setups solvent_ind -table
//
void HoloConst::set_solvent_ind(int nsolvent, int3 *h_solvent_ind) {

  this->nsolvent = nsolvent;

  reallocate<int3>(&solvent_ind, &solvent_ind_len, nsolvent, 1.5f);
  copy_HtoD<int3>(h_solvent_ind, solvent_ind, nsolvent);

}

//
// Apply constraints
//
void HoloConst::apply(double *xyz0, double *xyz1, int stride) {

  if (nsolvent == 0) return;

  int nthread = 512;
  int nblock = (nsolvent-1)/nthread+1;

  settle_solvent_kernel<<< nblock, nthread >>>(nsolvent, solvent_ind, mO_div_mH2O, mH_div_mH2O,
					       ra, rc, rb, ra_inv, rc2, stride, xyz0, xyz1);

  cudaCheck(cudaGetLastError());  

}
