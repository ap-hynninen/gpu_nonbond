#include <cuda.h>
#include "gpu_utils.h"
#include "XYZQ.h"
#include "Bspline.h"
#include "Grid.h"

int main(int argc, char *argv[]) {

  const double boxx = 62.23;
  const double boxy = 62.23;
  const double boxz = 62.23;
  const int nfftx = 64;
  const int nffty = 64;
  const int nfftz = 64;
  const int order = 4;
  int nnode = 1;
  int mynode = 0;

  double recip[9];

  int gpu_ind = 1;
  cudaCheck(cudaSetDevice(gpu_ind));

  cudaCheck(cudaThreadSynchronize());
  
  cudaDeviceProp gpu_prop;
  cudaCheck(cudaGetDeviceProperties(&gpu_prop, gpu_ind));

  printf("Using CUDA device (%d) %s\n",gpu_ind,gpu_prop.name);

  for (int i=0;i < 9;i++) recip[i] = 0.0;
  recip[0] = 1.0/boxx;
  recip[4] = 1.0/boxy;
  recip[8] = 1.0/boxz;

  XYZQ xyzq(argv[1]);

  Bspline<float, float3> bspline(23558, order, recip);
  Grid<long long int> grid(nfftx, nffty, nfftz, order, nnode, mynode);

  grid.make_fft_plans();
  grid.print_info();

  bspline.fill_bspline(xyzq.xyzq, xyzq.ncoord, nfftx, nffty, nfftz);

  grid.spread_charge(xyzq.ncoord, bspline);
  grid.real2complex_fft();

  return 0;
}

void read_xyzq() {
}
