#ifndef CUDAFORCEFIELD_H
#define CUDAFORCEFIELD_H

#include "Forcefield.h"
#include "cudaXYZ.h"
#include "Force.h"
#include "HoloConst.h"

//
// Abstract base class for CUDA force fields
//
class CudaForcefield : public Forcefield {

public:

  virtual void pre_calc(cudaXYZ<double>& coord, cudaXYZ<double>& prev_step, cudaStream_t stream)=0;
  virtual void calc(const bool calc_energy, const bool calc_virial, Force<long long int>& force,
		    cudaStream_t stream)=0;
  virtual void post_calc(const float *global_mass, float *mass, HoloConst* holoconst,
			 cudaStream_t stream)=0;
  virtual void stop_calc(cudaStream_t stream)=0;

  virtual void constComm(const int dir, cudaXYZ<double>& coord, cudaStream_t stream)=0;

  virtual void assignCoordToNodes(hostXYZ<double>& coord, std::vector<int>& h_loc2glo)=0;

  virtual void get_restart_data(cudaXYZ<double>& coord, cudaXYZ<double>& step,
				Force<long long int>& force,
				double *x, double *y, double *z,
				double *dx, double *dy, double *dz,
				double *fx, double *fy, double *fz)=0;
};

#endif // CUDAFORCEFIELD_H
