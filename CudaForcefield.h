#ifndef CUDAFORCEFIELD_H
#define CUDAFORCEFIELD_H
#include "cudaXYZ.h"
#include "Force.h"

//
// Abstract base class for CUDA force fields
//
class CudaForcefield {

public:

  virtual void calc(const cudaXYZ<double> *coord, const bool calc_energy, const bool calc_virial,
		    Force<long long int> *force)=0;

};

#endif // CUDAFORCEFIELD_H
