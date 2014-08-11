
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>

#define cudaCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
	  printf("Error running %s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
	  printf("Error string: %s\n",cudaGetErrorString(err)); \
	  exit(1);							\
        }                                                  \
    } while(0)

// Returns stride that aligns with 256 byte boundaries
template <typename T>
inline int calc_stride(int ncoord) {
  //const int sizeof_T = 4;
  return (( (ncoord-1+32)*sizeof(T) - 1)/256 + 1)*256/sizeof(T);
  //return ((ncoord*sizeof_T - 1)/256 + 1)*256/sizeof_T;
}

void deallocate_host_T(void **pp);
void allocate_host_T(void **pp, const int len, const size_t sizeofT);
void reallocate_host_T(void **pp, int *curlen, const int newlen, const float fac, const size_t sizeofT);

void deallocate_T(void **pp);
void allocate_T(void **pp, const int len, const size_t sizeofT);
void reallocate_T(void **pp, int *curlen, const int newlen, const float fac, const size_t sizeofT);

#ifdef __CUDACC__
void copy_HtoD_async_T(const void *h_array, void *d_array, int array_len, cudaStream_t stream,
		       const size_t sizeofT);
#endif
void copy_HtoD_T(const void *h_array, void *d_array, int array_len,
		 const size_t sizeofT);

#ifdef __CUDACC__
void copy_DtoH_async_T(const void *d_array, void *h_array, const int array_len, cudaStream_t stream,
		       const size_t sizeofT);
#endif
void copy_DtoH_T(const void *d_array, void *h_array, const int array_len, const size_t sizeofT);

#ifdef __CUDACC__
void copy_DtoD_async_T(const void *d_src, void *d_dst, const int array_len, cudaStream_t stream,
		       const size_t sizeofT);
#endif
void copy_DtoD_T(const void *d_src, void *d_dst, const int array_len, const size_t sizeofT);

#ifdef __CUDACC__
void clear_gpu_array_async_T(void *data, const int ndata, cudaStream_t stream, const size_t sizeofT);
#endif
void clear_gpu_array_T(void *data, const int ndata, const size_t sizeofT);

#ifdef __CUDACC__
void set_gpu_array_async_T(void *data, const int ndata, const int value,
			   cudaStream_t stream, const size_t sizeofT);
#endif
void set_gpu_array_T(void *data, const int ndata, const int value, const size_t sizeofT);

void copy3D_HtoD_T(void* src_data, void* dst_data,
		   int src_x0, int src_y0, int src_z0,
		   size_t src_xsize, size_t src_ysize,
		   int dst_x0, int dst_y0, int dst_z0,
		   size_t width, size_t height, size_t depth,
		   size_t dst_xsize, size_t dst_ysize,
		   size_t sizeofT);

void copy3D_DtoH_T(void* src_data, void* dst_data,
		   int src_x0, int src_y0, int src_z0,
		   size_t src_xsize, size_t src_ysize,
		   int dst_x0, int dst_y0, int dst_z0,
		   size_t width, size_t height, size_t depth,
		   size_t dst_xsize, size_t dst_ysize,
		   size_t sizeofT);

//----------------------------------------------------------------------------------------
//
// Deallocate page-locked host memory
// pp = memory pointer
//
#ifdef __cplusplus
template <class T>
void deallocate_host(T **pp) {
  deallocate_host_T((void **)pp);
}
#endif
//----------------------------------------------------------------------------------------
//
// Allocate page-locked host memory
// pp = memory pointer
// len = length of the array
//
#ifdef __cplusplus
template <class T>
void allocate_host(T **pp, const int len) {
  allocate_host_T((void **)pp, len, sizeof(T));
}
#endif

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate host memory
// pp = memory pointer
// curlen = current length of the array
// newlen = new required length of the array
// fac = extra space allocation factor: in case of re-allocation new length will be fac*newlen
//
#ifdef __cplusplus
template <class T>
void reallocate_host(T **pp, int *curlen, const int newlen, const float fac=1.0f) {
  reallocate_host_T((void **)pp, curlen, newlen, fac, sizeof(T));
}
#endif

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
#ifdef __cplusplus
template <class T>
void deallocate(T **pp) {
  deallocate_T((void **)pp);
}
#endif
//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
#ifdef __cplusplus
template <class T>
void allocate(T **pp, const int len) {
  allocate_T((void **)pp, len, sizeof(T));
}
#endif

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate gpu memory
// pp = memory pointer
// curlen = current length of the array
// newlen = new required length of the array
// fac = extra space allocation factor: in case of re-allocation new length will be fac*newlen
//
#ifdef __cplusplus
template <class T>
void reallocate(T **pp, int *curlen, const int newlen, const float fac=1.0f) {
  reallocate_T((void **)pp, curlen, newlen, fac, sizeof(T));
}
#endif

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device
//
#ifdef __cplusplus
template <class T>
void copy_HtoD(const T *h_array, T *d_array, int array_len
#ifdef __CUDACC__
	       , cudaStream_t stream=0
#endif
	       ) {

#ifdef __CUDACC__
  copy_HtoD_async_T(h_array, d_array, array_len, stream, sizeof(T));
#else
  copy_HtoD_T(h_array, d_array, array_len, sizeof(T));
#endif
}
#endif

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device using synchronous calls
//
#ifdef __cplusplus
template <class T>
void copy_HtoD_sync(const T *h_array, T *d_array, int array_len) {
  copy_HtoD_T(h_array, d_array, array_len, sizeof(T));
}
#endif

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
#ifdef __cplusplus
template <class T>
void copy_DtoH(const T *d_array, T *h_array, const int array_len
#ifdef __CUDACC__
	       , cudaStream_t stream=0
#endif
	       ) {
#ifdef __CUDACC__
  copy_DtoH_async_T(d_array, h_array, array_len, stream, sizeof(T));
#else
  copy_DtoH_T(d_array, h_array, array_len, sizeof(T));
#endif
}
#endif
//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host using synchronous calls
//
#ifdef __cplusplus
template <class T>
void copy_DtoH_sync(const T *d_array, T *h_array, const int array_len) {
  copy_DtoH_T(d_array, h_array, array_len, sizeof(T));
}
#endif

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Device
//
#ifdef __cplusplus
template <class T>
void copy_DtoD(const T *d_src, T *h_dst, const int array_len
#ifdef __CUDACC__
	       , cudaStream_t stream=0
#endif
	       ) {
#ifdef __CUDACC__
  copy_DtoD_async_T(d_src, h_dst, array_len, stream, sizeof(T));
#else
  copy_DtoD_T(d_src, h_dst, array_len, sizeof(T));
#endif
}
#endif
//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Device using synchronous calls
//
#ifdef __cplusplus
template <class T>
void copy_DtoD_sync(const T *d_src, T *h_dst, const int array_len) {
  copy_DtoD_T(d_src, h_dst, array_len, sizeof(T));
}
#endif

//----------------------------------------------------------------------------------------

#ifdef __cplusplus
template <class T>
void clear_gpu_array(T *data, const int ndata
#ifdef __CUDACC__
		     , cudaStream_t stream=0
#endif
		     ) {
#ifdef __CUDACC__
  clear_gpu_array_async_T(data, ndata, stream, sizeof(T));
#else
  clear_gpu_array_T(data, ndata, sizeof(T));
#endif
}
#endif

//----------------------------------------------------------------------------------------

#ifdef __cplusplus
template <class T>
void set_gpu_array(T *data, const int ndata, const int value
#ifdef __CUDACC__
		   , cudaStream_t stream=0
#endif
		   ) {
#ifdef __CUDACC__
  set_gpu_array_async_T(data, ndata, value, stream, sizeof(T));
#else
  set_gpu_array_T(data, ndata, value, sizeof(T));
#endif
}
#endif

#ifdef __cplusplus
template <class T>
void set_gpu_array_sync(T *data, const int ndata, const int value) {
  set_gpu_array_T(data, ndata, value, sizeof(T));
}
#endif

//----------------------------------------------------------------------------------------

#ifdef __cplusplus
template <class T>
void copy3D_HtoD(T* src_data, T* dst_data,
		 int src_x0, int src_y0, int src_z0,
		 size_t src_xsize, size_t src_ysize,
		 int dst_x0, int dst_y0, int dst_z0,
		 size_t width, size_t height, size_t depth,
		 size_t dst_xsize, size_t dst_ysize) {
  copy3D_HtoD_T(src_data, dst_data, src_x0, src_y0, src_z0,
		src_xsize, src_ysize,
		dst_x0, dst_y0, dst_z0,
		width, height, depth,
		dst_xsize, dst_ysize, sizeof(T));
}
#endif

//----------------------------------------------------------------------------------------

#ifdef __cplusplus
template <class T>
void copy3D_DtoH(T* src_data, T* dst_data,
		 int src_x0, int src_y0, int src_z0,
		 size_t src_xsize, size_t src_ysize,
		 int dst_x0, int dst_y0, int dst_z0,
		 size_t width, size_t height, size_t depth,
		 size_t dst_xsize, size_t dst_ysize) {
  copy3D_DtoH_T(src_data, dst_data, src_x0, src_y0, src_z0,
		src_xsize, src_ysize,
		dst_x0, dst_y0, dst_z0,
		width, height, depth,
		dst_xsize, dst_ysize, sizeof(T));
}
#endif

//----------------------------------------------------------------------------------------
#ifdef __CUDACC__
#ifdef __cplusplus
template <class T>
__global__ void map_to_local_array_kernel(const int narray, const int *loc2glo, const T* global_array,
					  T* local_array) {
  const int i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i < narray) {
    local_array[i] = global_array[loc2glo[i]];
  }
}

template <class T>
void map_to_local_array(const int narray, const int *loc2glo, const T *global_array, T *local_array,
			cudaStream_t stream=0) {
  int nthread = 256;
  int nblock = (narray-1)/nthread + 1;
  map_to_local_array_kernel<T> <<< nblock, nthread, 0, stream >>>
    (narray, loc2glo, global_array, local_array);
  cudaCheck(cudaGetLastError());
}
#endif
#endif

//----------------------------------------------------------------------------------------
void gpu_range_start(const char *range_name);
void gpu_range_stop();
//----------------------------------------------------------------------------------------

void start_gpu(int numnode, int mynode);
void stop_gpu();
int get_gpu_ind();
int get_cuda_arch();

#ifdef __CUDACC__
int3 get_max_nblock();
int get_max_nthread();
int get_major();
#endif

#endif // CUDA_UTILS_H
