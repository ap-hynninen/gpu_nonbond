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
// Calculates: a = b + c
//
__global__ void add_coord_kernel(const int stride3,
				 const double* __restrict__ b,
				 const double* __restrict__ c,
				 double* __restrict__ a) {

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;

  if (tid < stride3) {
    a[tid] = b[tid] + c[tid];
  }

}

//
// Calculates: a = b - c
//
__global__ void sub_coord_kernel(const int stride3,
				 const double* __restrict__ b,
				 const double* __restrict__ c,
				 double* __restrict__ a) {

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;

  if (tid < stride3) {
    a[tid] = b[tid] - c[tid];
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
CudaLeapfrogIntegrator::CudaLeapfrogIntegrator(HoloConst *holoconst,
					       const int npair, const int2 *h_pair_ind,
					       const double *h_pair_constr, const double *h_pair_mass,
					       const int ntrip, const int3 *h_trip_ind,
					       const double *h_trip_constr, const double *h_trip_mass,
					       const int nquad, const int4 *h_quad_ind,
					       const double *h_quad_constr, const double *h_quad_mass,
					       const int nsolvent, const int3 *h_solvent_ind,
					       cudaStream_t stream) {
  this->holoconst = holoconst;

  this->npair = 0;
  this->pair_ind = NULL;
  this->pair_constr = NULL;
  this->pair_mass = NULL;
  
  this->ntrip = 0;
  this->trip_ind = NULL;
  this->trip_constr = NULL;
  this->trip_mass = NULL;
  
  this->nquad = 0;
  this->quad_ind = NULL;
  this->quad_constr = NULL;
  this->quad_mass = NULL;
  
  this->nsolvent = 0;
  this->solvent_ind = NULL;
  
  if (holoconst != NULL) {
    this->npair = npair;
    if (npair > 0) {
      allocate<int2>(&pair_ind, npair);
      allocate<double>(&pair_constr, npair);
      allocate<double>(&pair_mass, npair*2);
      copy_HtoD<int2>(h_pair_ind, pair_ind, npair);
      copy_HtoD<double>(h_pair_constr, pair_constr, npair);
      copy_HtoD<double>(h_pair_mass, pair_mass, npair*2);
    }
    this->ntrip = ntrip;
    if (ntrip > 0) {
      allocate<int3>(&trip_ind, ntrip);
      allocate<double>(&trip_constr, ntrip*2);
      allocate<double>(&trip_mass, ntrip*5);
      copy_HtoD<int3>(h_trip_ind, trip_ind, ntrip);
      copy_HtoD<double>(h_trip_constr, trip_constr, ntrip*2);
      copy_HtoD<double>(h_trip_mass, trip_mass, ntrip*5);
    }
    this->nquad = nquad;
    if (nquad > 0) {
      allocate<int4>(&quad_ind, nquad);
      allocate<double>(&quad_constr, nquad*3);
      allocate<double>(&quad_mass, nquad*7);
      copy_HtoD<int4>(h_quad_ind, quad_ind, nquad);
      copy_HtoD<double>(h_quad_constr, quad_constr, nquad*3);
      copy_HtoD<double>(h_quad_mass, quad_mass, nquad*7);
    }
    this->nsolvent = nsolvent;
    if (nsolvent > 0) {
      allocate<int3>(&solvent_ind, nsolvent);
      copy_HtoD<int3>(h_solvent_ind, solvent_ind, nsolvent);
    }
  }
  this->stream = stream;
  cudaCheck(cudaEventCreate(&copy_rms_work_done_event));
  cudaCheck(cudaEventCreate(&copy_temp_ekin_done_event));
  cudaCheck(cudaEventCreate(&done_integrate_event));
  global_mass = NULL;
  mass_len = 0;
  mass = NULL;
  allocate_host<CudaLeapfrogIntegrator_storage_t>(&h_CudaLeapfrogIntegrator_storage, 1);
}

//
// Class destructor
//
CudaLeapfrogIntegrator::~CudaLeapfrogIntegrator() {

  if (pair_ind != NULL) deallocate<int2>(&pair_ind);
  if (pair_constr != NULL) deallocate<double>(&pair_constr);
  if (pair_mass != NULL) deallocate<double>(&pair_mass);

  if (trip_ind != NULL) deallocate<int3>(&trip_ind);
  if (trip_constr != NULL) deallocate<double>(&trip_constr);
  if (trip_mass != NULL) deallocate<double>(&trip_mass);

  if (quad_ind != NULL) deallocate<int4>(&quad_ind);
  if (quad_constr != NULL) deallocate<double>(&quad_constr);
  if (quad_mass != NULL) deallocate<double>(&quad_mass);

  if (solvent_ind != NULL) deallocate<int3>(&solvent_ind);

  cudaCheck(cudaEventDestroy(copy_rms_work_done_event));
  cudaCheck(cudaEventDestroy(copy_temp_ekin_done_event));
  cudaCheck(cudaEventDestroy(done_integrate_event));
  if (global_mass != NULL) deallocate<float>(&global_mass);
  if (mass != NULL) deallocate<float>(&mass);
  deallocate_host<CudaLeapfrogIntegrator_storage_t>(&h_CudaLeapfrogIntegrator_storage);
}

//
// Initialize integrator
//
void CudaLeapfrogIntegrator::spec_init(const int ncoord_glo,
				       const double *x, const double *y, const double *z,
				       const double *dx, const double *dy, const double *dz,
				       const double *h_mass) {
  if (forcefield == NULL) {
    std::cerr << "CudaLeapfrogIntegrator::spec_init, no forcefield set!" << std::endl;
    exit(1);
  }

  // Create host array for coordinates
  hostXYZ<double> h_prev_coord(ncoord, NON_PINNED);
  h_prev_coord.set_data_fromhost(x, y, z);

  CudaForcefield *p = static_cast<CudaForcefield*>(forcefield);
  int ncoord_loc = p->init_coord(h_prev_coord);

  //std::cerr << "CudaLeapfrogIntegrator::spec_init, exit(1)" << std::endl;
  //exit(1);

  // Resize and set arrays
  step.resize(ncoord);
  step.clear();

  prev_step.resize(ncoord);
  prev_step.set_data_sync(ncoord, dx, dy, dz);

  coord.resize(ncoord);
  coord.clear();

  prev_coord.resize(ncoord);
  prev_coord.set_data_sync(ncoord, x, y, z);

  force.set_ncoord(ncoord);

  // Make global mass array
  float *h_mass_f = new float[ncoord];
  for (int i=0;i < ncoord;i++) {
    h_mass_f[i] = (float)h_mass[i];
  }
  allocate<float>(&global_mass, ncoord);
  copy_HtoD<float>(h_mass_f, global_mass, ncoord);
  delete [] h_mass_f;

  // Host versions of coordinate, step, and force arrays
  // NOTE: These are used for copying coordinates
  h_coord.resize(ncoord);
  h_step.resize(ncoord);
  h_force.resize(ncoord);

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
// coord = prev_coord + prev_step
//
void CudaLeapfrogIntegrator::take_step() {

  add_coord(prev_coord, prev_step, coord);

  cudaCheck(cudaEventRecord(done_integrate_event, stream));
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

void CudaLeapfrogIntegrator::pre_calc_force() {
  if (forcefield != NULL) {
    CudaForcefield *p = static_cast<CudaForcefield*>(forcefield);
    cudaCheck(cudaStreamWaitEvent(stream, done_integrate_event, 0));
    p->pre_calc(&coord, &prev_step);
  }
}

void CudaLeapfrogIntegrator::calc_force(const bool calc_energy, const bool calc_virial) {
  if (forcefield != NULL) {
    CudaForcefield *p = static_cast<CudaForcefield*>(forcefield);
    p->calc(calc_energy, calc_virial, force);
  }
}

void CudaLeapfrogIntegrator::post_calc_force() {
  if (forcefield != NULL) {
    CudaForcefield *p = static_cast<CudaForcefield*>(forcefield);
    if (holoconst != NULL) {
      
    }
    reallocate<float>(&mass, &mass_len, coord.n);
    p->post_calc(global_mass, mass);
    p->wait_calc(stream);
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
  if (holoconst != NULL) {
    // prev_coord = coord + step
    add_coord(coord, step, prev_coord);
    // holonomic constraint, result in prev_coord
    holoconst->apply(&coord, &prev_coord, stream);
    // step = prev_coord - coord
    sub_coord(prev_coord, coord, step);
  }
}

//
// Calculates: a = b + c
//
void CudaLeapfrogIntegrator::add_coord(cudaXYZ<double> &b, cudaXYZ<double> &c,
				       cudaXYZ<double> &a) {
  assert(b.match(c));
  assert(b.match(a));

  int stride = a.stride;
  int nthread = 512;
  int nblock = (3*stride - 1)/nthread + 1;

  add_coord_kernel<<< nblock, nthread, 0, stream >>>
    (3*stride, b.data, c.data, a.data);

  cudaCheck(cudaGetLastError());
}

//
// Calculates: a = b - c
//
void CudaLeapfrogIntegrator::sub_coord(cudaXYZ<double> &b, cudaXYZ<double> &c,
				       cudaXYZ<double> &a) {
  assert(b.match(c));
  assert(b.match(a));

  int stride = a.stride;
  int nthread = 512;
  int nblock = (3*stride - 1)/nthread + 1;

  sub_coord_kernel<<< nblock, nthread, 0, stream >>>
    (3*stride, b.data, c.data, a.data);

  cudaCheck(cudaGetLastError());
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

