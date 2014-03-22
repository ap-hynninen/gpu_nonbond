#ifndef VIRIALPRESSURE_H
#define VIRIALPRESSURE_H
#include "cudaXYZ.h"
#include "Force.h"

class VirialPressure {

private:

  // vpress in device memory (9 doubles)
  //double *vpress;

  // vpress in host memory (9 doubles)
  //double *h_vpress;

  cudaEvent_t copy_virial_done_event;

public:

  VirialPressure();
  ~VirialPressure();

  void calc_virial(cudaXYZ<double> *coord,
		   cudaXYZ<double> *force,
		   float3 *xyz_shift,
		   float boxx, float boxy, float boxz,
		   double *d_vpress, double *h_vpress,
		   cudaStream_t stream=0);

  void wait_virial();
  void read_virial(double *h_vpress, double *vpress_out);
};

#endif // VIRIALPRESSURE_H
