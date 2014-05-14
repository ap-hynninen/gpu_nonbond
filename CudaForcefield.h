#ifndef CUDAFORCEFIELD_H
#define CUDAFORCEFIELD_H

#include "Forcefield.h"
#include "cudaXYZ.h"
#include "Force.h"

//
// Abstract base class for CUDA force fields
//
class CudaForcefield : public Forcefield {

public:

  virtual void calc(cudaXYZ<double> *coord, const bool calc_energy, const bool calc_virial,
		    Force<long long int> *force)=0;

  virtual void init_coord(cudaXYZ<double> *coord)=0;

};

#endif // CUDAFORCEFIELD_H
