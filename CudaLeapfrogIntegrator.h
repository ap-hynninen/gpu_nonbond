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

  // Masses
  //int global_mass_len;
  float *global_mass;

  int mass_len;
  float *mass;

  // Holonomic constraint global arrays:
  //
  // pair_ind[npair]
  // pair_constr[npair]
  // pair_mass[npair*2]
  //
  // trip_ind[ntrip]
  // trip_constr[ntrip*2]
  // trip_mass[ntrip*5]
  //
  // quad_ind[nquad]
  // quad_constr[nquad*3]
  // quad_mass[nquad*7]
  //
  int npair;
  int2 *pair_ind;
  double *pair_constr;
  double *pair_mass;

  int ntrip;
  int3 *trip_ind;
  double *trip_constr;
  double *trip_mass;

  int nquad;
  int4 *quad_ind;
  double *quad_constr;
  double *quad_mass;

  int nsolvent;
  int3 *solvent_ind;

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

  CudaLeapfrogIntegrator(HoloConst *holoconst,
			 const int npair, const int2 *h_pair_ind,
			 const double *h_pair_constr, const double *h_pair_mass,
			 const int ntrip, const int3 *h_trip_ind,
			 const double *h_trip_constr, const double *h_trip_mass,
			 const int nquad, const int4 *h_quad_ind,
			 const double *h_quad_constr, const double *h_quad_mass,
			 const int nsolvent, const int3 *h_solvent_ind,
			 cudaStream_t stream=0);
  ~CudaLeapfrogIntegrator();

  void spec_init(const double *x, const double *y, const double *z,
		 const double *dx, const double *dy, const double *dz,
		 const double *h_mass);

};

#endif // CUDALEAPFROGINTEGRATOR_H
