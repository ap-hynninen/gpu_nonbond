#include <iostream>
#include <cassert>
#include "CudaLeapfrogIntegrator.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

//
// Storage
//
static __device__ CudaLeapfrogIntegrator_storage_t d_CudaLeapfrogIntegrator_storage;

//
// Calculates: coord = prev_coord + step
//
__global__ void take_step_kernel(const int stride3,
				 double* __restrict__ prev_coord,
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

//
// Calculates kinetic energy
//
__global__ void calc_kine_kernel(const int ncoord, const int stride,
				 const double fac,
				 const float* __restrict__ mass,
				 const double* __restrict__ prev_step,
				 const double* __restrict__ step) {

  // Required shared memory:
  // blockDim.x*sizeof(double)
  extern __shared__ double sh_kine[];

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;

  double kine = 0.0;
  if (tid < ncoord) {
    double vx = (prev_step[tid]          + step[tid])*fac;
    double vy = (prev_step[tid+stride]   + step[tid+stride])*fac;
    double vz = (prev_step[tid+stride*2] + step[tid+stride*2])*fac;
    kine = ((double)mass[tid])*(vx*vx + vy*vy + vz*vz);
  }

  sh_kine[threadIdx.x] = kine;
  __syncthreads();
  for (int d=1;d < blockDim.x;d*=2) {
    int t = threadIdx.x + d;
    double kine_val = (t < blockDim.x) ? sh_kine[t] : 0.0;
    __syncthreads();
    sh_kine[threadIdx.x] += kine_val;
    __syncthreads();
  }
  
  if (threadIdx.x == 0) {
    atomicAdd(&d_CudaLeapfrogIntegrator_storage.kine, sh_kine[0]);
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
  mass_len = 0;
  mass = NULL;
  allocate_host<CudaLeapfrogIntegrator_storage_t>(&h_CudaLeapfrogIntegrator_storage, 1);
}

//
// Class destructor
//
CudaLeapfrogIntegrator::~CudaLeapfrogIntegrator() {
  cudaCheck(cudaEventDestroy(copy_rms_work_done_event));
  cudaCheck(cudaEventDestroy(copy_temp_ekin_done_event));
  if (mass != NULL) deallocate<float>(&mass);
  deallocate_host<CudaLeapfrogIntegrator_storage_t>(&h_CudaLeapfrogIntegrator_storage);
}

//
// Initialize integrator
//
void CudaLeapfrogIntegrator::spec_init(const int ncoord,
				       const double *x, const double *y, const double *z,
				       const double *dx, const double *dy, const double *dz,
				       const double *h_mass) {

  // Resize arrays
  step.resize(ncoord);
  prev_step.resize(ncoord);

  coord.resize(ncoord);
  prev_coord.resize(ncoord);

  force.set_ncoord(ncoord);

  reallocate<float>(&mass, &mass_len, ncoord);

  // Copy array data
  prev_coord.set_data_sync(ncoord, x, y, z);
  prev_step.set_data_sync(ncoord, dx, dy, dz);

  float *h_mass_f = new float[ncoord];
  for (int i=0;i < ncoord;i++) {
    h_mass_f[i] = (float)h_mass[i];
  }
  copy_HtoD<float>(h_mass_f, mass, ncoord);
  delete [] h_mass_f;

  step.clear();
  coord.clear();

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
// Swaps coord <=> prev_coord
//
void CudaLeapfrogIntegrator::swap_coord() {
  assert(coord.match(prev_coord));

  // Wait here until work on stream has stopped
  cudaCheck(cudaStreamSynchronize(stream));

  coord.swap(prev_coord);

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

  int ncoord = step.n;
  int stride = step.stride;
  int stride_force = force.xyz.stride;
  int nthread = 512;
  int nblock = (ncoord - 1)/nthread + 1;

  double dtsq = timestep_akma*timestep_akma;

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
    p->calc(&coord, &prev_step, mass, calc_energy, calc_virial, &force);
  }

}

//
// Calculate temperature
//
void CudaLeapfrogIntegrator::calc_temperature() {
  // Clear kinetic energy accumulator
  h_CudaLeapfrogIntegrator_storage->kine = 0.0;
  cudaCheck(cudaMemcpyToSymbolAsync(d_CudaLeapfrogIntegrator_storage,
				    h_CudaLeapfrogIntegrator_storage,
				    sizeof(CudaLeapfrogIntegrator_storage_t),
				    0, cudaMemcpyHostToDevice, stream));
  // Calculate kinetic energy
  int ncoord = step.n;
  int stride = step.stride;
  int nthread = 512;
  int nblock = (ncoord - 1)/nthread + 1;
  int shmem_size = nthread*sizeof(double);
  double fac = 0.5/timestep_akma;
  calc_kine_kernel<<< nblock, nthread, shmem_size, stream >>>
    (ncoord, stride, fac, mass, prev_step.data, step.data);
  cudaCheck(cudaGetLastError());
  // Retrieve result
  cudaCheck(cudaMemcpyFromSymbol(h_CudaLeapfrogIntegrator_storage,
				 d_CudaLeapfrogIntegrator_storage,
				 sizeof(CudaLeapfrogIntegrator_storage_t),
				 0, cudaMemcpyDeviceToHost));
  //std::cout << "kinetic energy = " << 0.5*h_CudaLeapfrogIntegrator_storage->kine << std::endl;
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

