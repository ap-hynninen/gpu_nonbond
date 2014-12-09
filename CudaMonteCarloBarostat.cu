#include <cuda.h>
#include "CudaMonteCarloBarostat.h"

//
// Scale coordinates independently in all 3 dimensions
// nmol               = number of molecules
// molStart[0...nmol] = molecule index start in molLoc
// molLoc[]           = Local index for atoms in the molecule
//
__global__ void scaleCoord3_kernel(const int nmol,
				   const int* __restrict__ molStart,
				   const int* __restrict__ molLoc,
				   const double scalex, const double scaley, const double scalez,
				   const double boxx, const double boxy, const double boxz,
				   const double inv_boxx, const double inv_boxy, const double inv_boxz,
				   double* __restrict__ x,
				   double* __restrict__ y,
				   double* __restrict__ z) {
  // Each thread takes care of a single molecule.
  // CHECK: Is this good enough for large molecules?
  for (int imol=threadIdx.x+blockIdx.x*blockDim.x;imol < nmol;imol+=gridDim.x*blockDim.x) {
    int istart = molStart[imol];
    int iend = molStart[imol+1];
    // Compute molecule center
    double xm = 0.0;
    double ym = 0.0;
    double zm = 0.0;
    for (int i=istart;i < iend;i++) {
      int j = molLoc[i];
      xm += x[j];
      ym += y[j];
      zm += z[j];
    }
    double inv_natom = 1.0/(double)(iend - istart);
    xm *= inv_natom;
    ym *= inv_natom;
    zm *= inv_natom;
    // Calculate distance from the center periodic box
    double dx = floor(xm*inv_boxx)*boxx;
    double dy = floor(ym*inv_boxy)*boxy;
    double dz = floor(zm*inv_boxz)*boxz;
    // Calculate shift (sx, sy, sz) for the molecule
    double sx = xm*(scalex - 1.0) - dx*scalex;
    double sy = ym*(scaley - 1.0) - dy*scaley;
    double sz = zm*(scalez - 1.0) - dz*scalez;
    // Store new shifted positions back to atom coordinates
    for (int i=istart;i < iend;i++) {
      int j = molLoc[i];
      x[j] += sx;
      y[j] += sy;
      z[j] += sz;
    }    
  }
}

//###################################################################################
//###################################################################################
//###################################################################################

//
// Class creator
//
CudaMonteCarloBarostat::CudaMonteCarloBarostat(const double Pref, const double Tref,
					       const int N) {
}

//
// Class desctructor
//
CudaMonteCarloBarostat::~CudaMonteCarloBarostat() {
}

//
// Scale coordinates
//
void CudaMonteCarloBarostat::scaleCoord(const int nmol, const int* molStart, const int* molLoc,
					cudaStream_t stream) {

  int nthread = 512;
  int nblock = (nmol-1)/nthread + 1;
  scaleCoord3_kernel<<< nblock, nthread, 0, stream >>>
    (nmol, molStart, molLoc, scalex, scaley, scalez,
     boxx, boxy, boxz, inv_boxx, inv_boxy, inv_boxz,
     double* __restrict__ x,
     double* __restrict__ y,
     double* __restrict__ z);
  cudaCheck(cudaGetLastError());
}
