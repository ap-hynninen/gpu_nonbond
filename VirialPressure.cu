#include <iostream>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "VirialPressure.h"

__global__ void calc_virial_kernel(const int ncoord, const int stride, const int stride_force,
				   const double *coord, const double *force,
				   const float3 *xyz_shift,
				   const float boxx, const float boxy, const float boxz,
				   double *global_buffer) {
  // Shared memory:
  // Requires 9*sizeof(double)*blockDim.x
  extern __shared__ double sh_buffer[];

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int stride2 = stride*2;
  const int stride_force2 = stride_force*2;

  double x, y, z;
  double forcex, forcey, forcez;
  x = 0.0;
  y = 0.0;
  z = 0.0;
  if (tid < ncoord) {
    x = coord[tid];
    y = coord[tid + stride];
    z = coord[tid + stride2];
    float3 xyzsh = xyz_shift[tid];
    x += (double)(xyzsh.x*boxx);
    y += (double)(xyzsh.y*boxy);
    z += (double)(xyzsh.z*boxz);
    forcex = -force[tid];
    forcey = -force[tid + stride_force];
    forcez = -force[tid + stride_force2];
  }

  double vpress[9];
  vpress[0] = x*forcex;
  vpress[1] = x*forcey;
  vpress[2] = x*forcez;

  vpress[3] = y*forcex;
  vpress[4] = y*forcey;
  vpress[5] = y*forcez;

  vpress[6] = z*forcex;
  vpress[7] = z*forcey;
  vpress[8] = z*forcez;

  // Reduce
#pragma unroll
  for (int i=0;i < 9;i++)
    sh_buffer[threadIdx.x + blockDim.x*i] = vpress[i];
  __syncthreads();
  for (int d=1;d < blockDim.x;d *= 2) {
    int t = threadIdx.x + d;
    double vals[9];
#pragma unroll
    for (int i=0;i < 9;i++)
      vals[i] = (t < blockDim.x) ? sh_buffer[t + blockDim.x*i] : 0.0;
    __syncthreads();
#pragma unroll
    for (int i=0;i < 9;i++)
      sh_buffer[threadIdx.x + blockDim.x*i] += vals[i];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double vals[9];
#pragma unroll
    for (int i=0;i < 9;i++)
      vals[i] = sh_buffer[blockDim.x*i];
#pragma unroll
    for (int i=0;i < 9;i++)
      atomicAdd(&global_buffer[i], vals[i]);
  }

}

//###########################################################################################

//
// Class creator
//
VirialPressure::VirialPressure() {
  allocate<double>(&global_buffer, 9);
}

//
// Class destructor
//
VirialPressure::~VirialPressure() {
  if (global_buffer != NULL) deallocate<double>(&global_buffer);
}

//
// Calculates the virial tensor due to non-bonded interactions
// coord = coordinates
// shift = coordinate shift
// force = forces
//
void VirialPressure::calc_virial(cudaXYZ<double> *coord,
				 cudaXYZ<double> *force,
				 float3 *xyz_shift,
				 float boxx, float boxy, float boxz,
				 double *vpress) {
  assert(coord->match(force));

  int ncoord = coord->n;
  int stride = coord->stride;
  int stride_force = force->stride;
  int nthread = 512;
  int nblock = (ncoord - 1)/nthread + 1;

  int shmem_size = 9*sizeof(double)*nthread;

  clear_gpu_array<double>(global_buffer, 9);
  
  calc_virial_kernel<<< nblock, nthread, shmem_size >>>
    (ncoord, stride, stride_force, coord->data, force->data, xyz_shift,
     boxx, boxy, boxz, global_buffer);

  copy_DtoH<double>(global_buffer, vpress, 9);

  cudaCheck(cudaGetLastError());
}
