
#ifndef REDUCE_H
#define REDUCE_H

#include "gpu_utils.h"

template <typename AT, typename CT>
__global__ static void reduce_data(const int nfft_tot,
				   const AT *data_in,
				   CT *data_out) {
  // The generic version can not be used
}

// Convert "long long int" -> "float"
template <>
__global__ static void reduce_data<long long int, float>(const int nfft_tot,
							 const long long int *data_in,
							 float *data_out) {
  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (pos < nfft_tot) {
    long long int val = data_in[pos];
    data_out[pos] = ((float)val)*INV_FORCE_SCALE;
    pos += blockDim.x*gridDim.x;
  }

}

// Convert "int" -> "float"
template <>
__global__ static void reduce_data<int, float>(const int nfft_tot,
					       const int *data_in,
					       float *data_out) {
  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (pos < nfft_tot) {
    int val = data_in[pos];
    data_out[pos] = ((float)val)*INV_FORCE_SCALE_I;
    pos += blockDim.x*gridDim.x;
  }

}

// Convert "long long int" -> "double"
template <>
__global__ static void reduce_data<long long int, double>(const int nfft_tot,
							  const long long int *data_in,
							  double *data_out) {
  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (pos < nfft_tot) {
    long long int val = data_in[pos];
    data_out[pos] = ((double)val)*INV_FORCE_SCALE;
    pos += blockDim.x*gridDim.x;
  }

}

// Convert "float" -> "double"
template <>
__global__ static void reduce_data<float, double>(const int nfft_tot,
						  const float *data_in,
						  double *data_out) {
  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (pos < nfft_tot) {
    float val = data_in[pos];
    data_out[pos] = ((double)val);
    pos += blockDim.x*gridDim.x;
  }

}

template <typename AT, typename CT>
__global__ static void reduce_data(const int nfft_tot,
				   AT *data_in) {
  // The generic version can not be used
}

// Convert "long long int" -> "double"
template <>
__global__ static void reduce_data<long long int, double>(const int nfft_tot,
							  long long int *data_in) {
  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  double *data_out = (double *)data_in;
  
  while (pos < nfft_tot) {
    long long int val = data_in[pos];
    data_out[pos] = ((double)val)*INV_FORCE_SCALE;
    pos += blockDim.x*gridDim.x;
  }

}

template <typename AT, typename CT1, typename CT2>
__global__ static void reduce_add_data(const int nfft_tot,
				       const CT2* __restrict__ data_add,
				       AT *data_inout) {
  // The generic version can not be used
}

// Convert "long long int" -> "double" and adds "float"
template <>
__global__ static 
void reduce_add_data<long long int, double, float>(const int nfft_tot,
						   const float* __restrict__ data_add,
						   long long int *data_inout) {

  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  double *data_out = (double *)data_inout;
  
  while (pos < nfft_tot) {
    long long int val = data_inout[pos];
    double val_add = (double)data_add[pos];
    data_out[pos] = ((double)val)*INV_FORCE_SCALE + val_add;
    pos += blockDim.x*gridDim.x;
  }

}
#endif // REDUCE_H
