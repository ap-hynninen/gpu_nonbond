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
  
  // Number of coordinates in the homezone
  int ncoord;

  // Total number of coordinates, including the import volume
  int ncoord_tot;

  // Coordinates (size ncoord_tot)
  cudaXYZ<double> coord;

  // Previous step coordinates (size ncoord)
  cudaXYZ<double> prev_coord;

  // Step vector (size ncoord)
  cudaXYZ<double> step;

  // Previous step vector (size ncoord)
  cudaXYZ<double> prev_step;

  // Global masses (size ncoord_glo)
  //int global_mass_len;
  float *global_mass;

  // Local masses (size ncoord)
  int mass_len;
  float *mass;

  // Force array (size ncoord_tot)
  Force<long long int> force;

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
  void pre_calc_force();
  void calc_force(const bool calc_energy, const bool calc_virial);
  void post_calc_force();
  void stop_calc_force();
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

  void spec_init(const double *x, const double *y, const double *z,
		 const double *dx, const double *dy, const double *dz,
		 const double *h_mass);

};

#endif // CUDALEAPFROGINTEGRATOR_H
