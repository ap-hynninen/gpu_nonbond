#include <iostream>
#include <cassert>
#include "CudaLeapfrogIntegrator.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

//
// Calculates: coord = prev_coord + step
//
__global__ void take_step_kernel(const int stride3,
				 const double* __restrict__ prev_coord,
				 const double* __restrict__ step,
				 double* __restrict__ coord) {

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;

  if (tid < stride3) {
    coord[tid] = prev_coord[tid] + 0*step[tid];
  }

}

//
// Calculates the next step vector using forces:
// step = prev_step - force*dt^2/mass
// gamma_val = dt^2/mass
//
__global__ void calc_step_kernel(const int ncoord, const int stride, const int stride_force, 
				 const double dtsq,
				 const double* __restrict__ force,
				 const double* __restrict__ prev_step,
				 const float* __restrict__ mass,
				 double* __restrict__ step) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int stride2 = stride*2;
  const int stride_force2 = stride_force*2;

  if (tid < ncoord) {
    double gamma_val = dtsq/(double)mass[tid];
    step[tid]         = prev_step[tid]         - force[tid]*gamma_val;
    step[tid+stride]  = prev_step[tid+stride]  - force[tid+stride_force]*gamma_val;
    step[tid+stride2] = prev_step[tid+stride2] - force[tid+stride_force2]*gamma_val;
  }
}

//##################################################################################################

//
// Class creator
//
CudaLeapfrogIntegrator::CudaLeapfrogIntegrator(HoloConst *holoconst, cudaStream_t stream) {
  this->holoconst = holoconst;
  this->stream = stream;
  cudaCheck(cudaEventCreate(&copy_rms_work_done_event));
  cudaCheck(cudaEventCreate(&copy_temp_ekin_done_event));
  mass = NULL;
}

//
// Class destructor
//
CudaLeapfrogIntegrator::~CudaLeapfrogIntegrator() {
  cudaCheck(cudaEventDestroy(copy_rms_work_done_event));
  cudaCheck(cudaEventDestroy(copy_temp_ekin_done_event));
  if (mass != NULL) deallocate<float>(&mass);
}

//
// Initialize integrator
//
void CudaLeapfrogIntegrator::init(const int ncoord,
				  const double *x, const double *y, const double *z,
				  const double *dx, const double *dy, const double *dz) {
  prev_coord.set_data_sync(ncoord, x, y, z);
  prev_step.set_data_sync(ncoord, dx, dy, dz);
  step.resize(ncoord);
  coord.resize(ncoord);
  force.set_ncoord(ncoord);

  // Host versions of coordinate, step, and force arrays
  // NOTE: These are used for copying coordinates
  h_coord.resize(ncoord);
  h_step.resize(ncoord);
  h_force.resize(ncoord);

  if (forcefield != NULL) {
    CudaForcefield *p = static_cast<CudaForcefield*>(forcefield);
    p->init_coord(&prev_coord);
  }
}

//
// Swaps step <=> prev_step
//
void CudaLeapfrogIntegrator::swap_step() {
  assert(step.match(prev_step));

  // Wait here until work on stream has stopped
  cudaCheck(cudaStreamSynchronize(stream));

  step.swap(prev_step);

}


//
// Calculates new current coordinate positions (cur) using 
// the previous coordinates (prev) and the step vector (step)
// coord = prev_coord + step
//
void CudaLeapfrogIntegrator::take_step() {
  assert(prev_coord.match(step));
  assert(prev_coord.match(coord));

  int stride = coord.stride;
  int nthread = 512;
  int nblock = (3*stride - 1)/nthread + 1;

  take_step_kernel<<< nblock, nthread, 0, stream >>>
    (3*stride, prev_coord.data, prev_step.data, coord.data);

  cudaCheck(cudaGetLastError());
}

//
// Calculate step
//
void CudaLeapfrogIntegrator::calc_step() {
  assert(prev_coord.match(step));

  return;

  int ncoord = step.n;
  int stride = step.stride;
  int stride_force = stride;
  int nthread = 512;
  int nblock = (ncoord - 1)/nthread + 1;

  double dtsq = timestep*timestep;

  calc_step_kernel<<< nblock, nthread, 0, stream >>>
    (ncoord, stride, stride_force, dtsq,
     (double *)force.xyz.data, 
     prev_step.data, mass, step.data);
  
  cudaCheck(cudaGetLastError());
}

//
// Calculate forces
//
void CudaLeapfrogIntegrator::calc_force(const bool calc_energy, const bool calc_virial) {

  if (forcefield != NULL) {
    CudaForcefield *p = static_cast<CudaForcefield*>(forcefield);
    p->calc(&coord, &prev_step, calc_energy, calc_virial, &force);
  }

}

//
// Do holonomic constraints
//
void CudaLeapfrogIntegrator::do_holoconst() {
  if (holoconst != NULL)
    holoconst->apply(&prev_coord, &coord);
}

//
// Do constant pressure
//
void CudaLeapfrogIntegrator::do_pressure() {
}

//
// Returns true if constant pressure is ON
//
bool CudaLeapfrogIntegrator::const_pressure() {
  return false;
}

//
// Do constant temperature
//
void CudaLeapfrogIntegrator::do_temperature() {
}

//
// Print energy & other info on screen
//
void CudaLeapfrogIntegrator::do_print_energy(int step) {
  if (forcefield != NULL) {
    CudaForcefield *p = static_cast<CudaForcefield*>(forcefield);
    p->print_energy_virial(step);
  }
}

//
// Get coordinates to host memory
//
void CudaLeapfrogIntegrator::get_restart_data(double *x, double *y, double *z,
					      double *dx, double *dy, double *dz,
					      double *fx, double *fy, double *fz) {

  if (forcefield != NULL) {
    h_coord.set_data_sync(coord);
    h_step.set_data_sync(step);
    h_force.set_data_sync(force.xyz);
    CudaForcefield *p = static_cast<CudaForcefield*>(forcefield);
    p->get_restart_data(&h_coord, &h_step, &h_force, x, y, z, dx, dy, dz, fx, fy, fz);
  }
  
}

