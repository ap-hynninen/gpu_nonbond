#ifndef CUDALEAPFROGINTEGRATOR_H
#define CUDALEAPFROGINTEGRATOR_H
#include <cuda.h>
#include "LeapfrogIntegrator.h"
#include "cudaXYZ.h"
#include "Force.h"
#include "CudaPMEForcefield.h"
#include "HoloConst.h"

struct CudaLeapfrogIntegrator_storage_t {
  // Kinetic energy
  double kine;
};

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
  int mass_len;
  float *mass;

  // Force array
  Force<long long int> force;

  // Host memory versions of coordinates, step, and force arrays
  hostXYZ<double> h_coord;
  hostXYZ<double> h_step;
  hostXYZ<double> h_force;

  // Holonomic constraints
  HoloConst *holoconst;

  // Host version of storage
  CudaLeapfrogIntegrator_storage_t *h_CudaLeapfrogIntegrator_storage;

  cudaEvent_t copy_rms_work_done_event;
  cudaEvent_t copy_temp_ekin_done_event;

  cudaEvent_t done_integrate_event;

  cudaStream_t stream;

  // private functions
  void add_coord(cudaXYZ<double> &b, cudaXYZ<double> &c, cudaXYZ<double> &a);
  void sub_coord(cudaXYZ<double> &b, cudaXYZ<double> &c, cudaXYZ<double> &a);

  // Pure virtual function overriders
  void swap_step();
  void swap_coord();
  void take_step();
  void calc_step();
  void calc_force(const bool calc_energy, const bool calc_virial);
  void calc_temperature();
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

  void spec_init(const int ncoord,
		 const double *x, const double *y, const double *z,
		 const double *dx, const double *dy, const double *dz,
		 const double *h_mass);

};

#endif // CUDALEAPFROGINTEGRATOR_H
