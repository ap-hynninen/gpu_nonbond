#ifndef VIRIALPRESSURE_H
#define VIRIALPRESSURE_H
#include "cudaXYZ.h"
#include "Force.h"

class VirialPressure {

private:

  // vpress in device memory (9 doubles)
  double *vpress;

  // vpress in host memory (9 doubles)
  double *h_vpress;

public:

  VirialPressure();
  ~VirialPressure();

  void calc_virial(cudaXYZ<double> *coord,
		   cudaXYZ<double> *force,
		   float3 *xyz_shift,
		   float boxx, float boxy, float boxz,
		   double *vpress_out, cudaStream_t stream=0);

};

#endif // VIRIALPRESSURE_H
