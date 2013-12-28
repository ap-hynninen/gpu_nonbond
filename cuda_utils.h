
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

void deallocate_host_T(void **pp);
void allocate_host_T(void **pp, const int len, const size_t sizeofT);

void deallocate_T(void **pp);
void allocate_T(void **pp, const int len, const size_t sizeofT);
void reallocate_T(void **pp, int *curlen, const int newlen, const float fac, const size_t sizeofT);

#ifdef __CUDACC__
void copy_HtoD_async_T(void *h_array, void *d_array, int array_len, cudaStream_t stream,
		       const size_t sizeofT);
#endif
void copy_HtoD_T(void *h_array, void *d_array, int array_len,
		 const size_t sizeofT);

#ifdef __CUDACC__
void copy_DtoH_async_T(void *d_array, void *h_array, const int array_len, cudaStream_t stream,
		       const size_t sizeofT);
#endif
void copy_DtoH_T(void *d_array, void *h_array, const int array_len, const size_t sizeofT);

void clear_gpu_array_T(void *data, const int ndata, /*cudaStream_t stream, */ const size_t sizeofT);

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
template <class T>
void deallocate_host(T **pp) {
  deallocate_host_T((void **)pp);
}
//----------------------------------------------------------------------------------------
//
// Allocate page-locked host memory
// pp = memory pointer
// len = length of the array
//
template <class T>
void allocate_host(T **pp, const int len) {
  allocate_host_T((void **)pp, len, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
template <class T>
void deallocate(T **pp) {
  deallocate_T((void **)pp);
}
//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
template <class T>
void allocate(T **pp, const int len) {
  allocate_T((void **)pp, len, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate gpu memory
// pp = memory pointer
// curlen = current length of the array
// newlen = new required length of the array
// fac = extra space allocation factor: in case of re-allocation new length will be fac*newlen
//
template <class T>
void reallocate(T **pp, int *curlen, const int newlen, const float fac=1.0f) {
  reallocate_T((void **)pp, curlen, newlen, fac, sizeof(T));
}
//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device
//
template <class T>
void copy_HtoD(T *h_array, T *d_array, int array_len
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

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
template <class T>
void copy_DtoH(T *d_array, T *h_array, const int array_len
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

//----------------------------------------------------------------------------------------

template <class T>
void clear_gpu_array(T *data, const int ndata /*, cudaStream_t stream=0*/) {
  clear_gpu_array_T(data, ndata, /*stream, */ sizeof(T));
}

//----------------------------------------------------------------------------------------

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

//----------------------------------------------------------------------------------------

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

//----------------------------------------------------------------------------------------

static void print_gpu_float(float *data, const int ndata) {
  float *h_data = new float[ndata];

  copy_DtoH<float>(data, h_data, ndata);
			   
  for (int i=0;i < ndata;i++)
    std::cout << h_data[i] << std::endl;

  delete [] h_data;
}

//----------------------------------------------------------------------------------------
void range_start(char *range_name);
void range_stop();
//----------------------------------------------------------------------------------------

void start_gpu(int numnode, int mynode, bool use_streams=false);

#ifdef __CUDACC__
cudaStream_t get_direct_nonbond_stream();
#endif

#endif // CUDA_UTILS_H
