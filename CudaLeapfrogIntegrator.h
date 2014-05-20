#ifndef CUDALEAPFROGINTEGRATOR_H
#define CUDALEAPFROGINTEGRATOR_H
#include <cuda.h>
#include "LeapfrogIntegrator.h"
#include "cudaXYZ.h"
#include "Force.h"
#include "CudaPMEForcefield.h"
#include "HoloConst.h"

class CudaLeapfrogIntegrator : public LeapfrogIntegrator {

  //friend class LangevinPiston;

private:

  // Coordinates
  cudaXYZ<double> coord;

  // Previous step coordinates
  cudaXYZ<double> prev_coord;

  // Step vector
  cudaXYZ<double> step;

  // Previous step vector 
  cudaXYZ<double> prev_step;

  // Mass
  float *mass;

  // Force array
  Force<long long int> force;

  // Host memory versions of coordinates, step, and force arrays
  hostXYZ<double> h_coord;
  hostXYZ<double> h_step;
  hostXYZ<double> h_force;

  // Holonomic constraints
  HoloConst *holoconst;

  cudaEvent_t copy_rms_work_done_event;
  cudaEvent_t copy_temp_ekin_done_event;

  cudaStream_t stream;

  void swap_step();
  void take_step();
  void calc_step();
  void calc_force(const bool calc_energy, const bool calc_virial);
  void do_holoconst();
  void do_pressure();
  void do_temperature();
  bool const_pressure();
  void do_print_energy(int step);
  void get_restart_data(double *x, double *y, double *z,
			double *dx, double *dy, double *dz,
			double *fx, double *fy, double *fz);

public:

  CudaLeapfrogIntegrator(HoloConst *holoconst, cudaStream_t stream=0);
  ~CudaLeapfrogIntegrator();

  void init(const int ncoord,
	    const double *x, const double *y, const double *z,
	    const double *dx, const double *dy, const double *dz);

};

#endif // CUDALEAPFROGINTEGRATOR_H
