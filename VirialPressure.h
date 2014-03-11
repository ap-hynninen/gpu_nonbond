#ifndef VIRIALPRESSURE_H
#define VIRIALPRESSURE_H
#include "cudaXYZ.h"
#include "Force.h"

class VirialPressure {

private:

  // 9 doubles
  double *global_buffer;

public:

  VirialPressure();
  ~VirialPressure();

  void calc_virial(cudaXYZ<double> *coord,
		   cudaXYZ<double> *force,
		   float3 *xyz_shift,
		   float boxx, float boxy, float boxz,
		   double *vpress);

};

#endif // VIRIALPRESSURE_H
