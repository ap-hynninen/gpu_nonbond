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
    coord[tid] = prev_coord[tid] + step[tid];
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
CudaLeapfrogIntegrator::CudaLeapfrogIntegrator(cudaStream_t stream) {
  this->stream = stream;
  cudaCheck(cudaEventCreate(&copy_rms_work_done_event));
  cudaCheck(cudaEventCreate(&copy_temp_ekin_done_event));
  mass = NULL;
  forcefield = NULL;
}

//
// Class destructor
//
CudaLeapfrogIntegrator::~CudaLeapfrogIntegrator() {
  cudaCheck(cudaEventDestroy(copy_rms_work_done_event));
  cudaCheck(cudaEventDestroy(copy_temp_ekin_done_event));
  if (mass != NULL) deallocate<float>(&mass);
  if (forcefield != NULL) delete forcefield;
}

//
// Initialize integrator
//
void CudaLeapfrogIntegrator::init(const int ncoord,
				  const double *x, const double *y, const double *z,
				  const double *dx, const double *dy, const double *dz) {
  prev_coord.set_data_sync(ncoord, x, y, z);
  prev_step.set_data_sync(ncoord, dx, dy, dz);
}

//
// Swaps step <=> prev_step
//
void CudaLeapfrogIntegrator::swap_step() {
  assert(step.match(prev_step));

  // Wait here until work on stream has stopped
  cudaCheck(cudaStreamSynchronize(stream));

  //step.swap(prev_step);

  //double *tmp = prev_step.data;
  //prev_step.data = step.data;
  //step.data = tmp;
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
    (3*stride, prev_coord.data, step.data, coord.data);

  cudaCheck(cudaGetLastError());
}

//
// Calculate step
//
void CudaLeapfrogIntegrator::calc_step() {
  assert(prev_coord.match(step));

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
    forcefield->calc(&coord, calc_energy, calc_virial, &force);
  }

}
