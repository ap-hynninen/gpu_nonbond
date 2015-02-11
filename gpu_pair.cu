//
// Calculates the SPME forces and energies for a pair of charges
//
#include <cuda.h>
#include <iostream>
#include "XYZQ.h"
#include "Force.h"
#include "CudaPMERecip.h"

template <typename T, typename T2>
void calcPair(const double L, const double kappa, const int nfft, const int order);

int main(int argc, char *argv[]) {

  double L=0.0, kappa=0.0;
  int nfft=0, order=0;

  bool arg_ok = true;
  int iarg = 1;
  while (iarg < argc) {
    if (strcmp(argv[iarg],"-L")==0) {
      iarg++;
      if (iarg == argc) {
	arg_ok = false;
	break;
      }
      sscanf(argv[iarg],"%lf",&L);
      iarg++;
    } else if (strcmp(argv[iarg],"-kappa")==0) {
      iarg++;
      if (iarg == argc) {
	arg_ok = false;
	break;
      }
      sscanf(argv[iarg],"%lf",&kappa);
      iarg++;
    } else if (strcmp(argv[iarg],"-nfft")==0) {
      iarg++;
      if (iarg == argc) {
	arg_ok = false;
	break;
      }
      sscanf(argv[iarg],"%d",&nfft);
      iarg++;
    } else if (strcmp(argv[iarg],"-order")==0) {
      iarg++;
      if (iarg == argc) {
	arg_ok = false;
	break;
      }
      sscanf(argv[iarg],"%d",&order);
      iarg++;
    } else {
      std::cout << "Invalid input parameter " << argv[iarg] << std::endl;
      arg_ok = false;
      break;
    }
  }

  if (!arg_ok || L == 0.0 || kappa == 0.0 || nfft == 0 || order == 0) {
    std::cout << "Usage: gpu_pair -L L -kappa kappa -nfft nfft -order order"<< std::endl;
    return 1;
  }
  
  calcPair<float, float2>(L, kappa, nfft, order);
  
  return 1;
}


template <typename T, typename T2>
void calcPair(const double L, const double kappa, const int nfft, const int order) {
  const FFTtype fft_type = BOX;

  // Setup reciprocal vectors
  double recip[9];
  for (int i=0;i < 9;i++) recip[i] = 0;
  recip[0] = 1.0/L;
  recip[4] = 1.0/L;
  recip[8] = 1.0/L;

  CudaEnergyVirial energyVirial;
  
  XYZQ xyzq(2);
  CudaPMERecip<int, T, T2> grid(nfft, nfft, nfft, order, fft_type, 1, 0,
			energyVirial, "recip", "self");
  Force<T> force(2);

  // r = Distance along diagonal
  double r = 6.0;
  double a = r/(2.0*sqrt(3.0));
  float4 h_xyzq[2];
  T fx[2], fy[2], fz[2];
  //h_xyzq[0].x = -a + 0.5*L;
  //h_xyzq[0].y = -a + 0.5*L;
  //h_xyzq[0].z = -a + 0.5*L;
  //h_xyzq[1].x = a + 0.5*L;
  //h_xyzq[1].y = a + 0.5*L;
  //h_xyzq[1].z = a + 0.5*L;

  h_xyzq[0].x = -r/2.0 + 0.5*L;
  h_xyzq[0].y = 0.5*L;
  h_xyzq[0].z = 0.5*L;
  h_xyzq[1].x = r/2.0 + 0.5*L;
  h_xyzq[1].y = 0.5*L;
  h_xyzq[1].z = 0.5*L;  
  
  h_xyzq[0].w = -1.0;
  h_xyzq[1].w = 1.0;
  
  xyzq.set_xyzq(2, h_xyzq);

  energyVirial.clear();
  
  grid.spread_charge(xyzq.xyzq, xyzq.ncoord, recip);
  grid.r2c_fft();
  grid.scalar_sum(recip, kappa, true, true);
  grid.c2r_fft();
  grid.gather_force(xyzq.xyzq, xyzq.ncoord, recip, force.stride(), force.xyz());
  force.getXYZ(fx, fy, fz);
  
  double energy, energy_self, virial[9];
  //grid.get_energy_virial(kappa, true, true, energy, energy_self, virial);
  energyVirial.copyToHost();
  cudaCheck(cudaDeviceSynchronize());
  energy = energyVirial.getEnergy("recip");
  energy_self = energyVirial.getEnergy("self");
  energyVirial.getVirial(virial);

  printf("%lf %e %e %e %e %e %e %e %e\n",r,energy,energy_self,fx[0],fy[0],fz[0],fx[1],fy[1],fz[1]);
  
}
