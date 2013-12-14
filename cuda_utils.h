
void deallocate_T(void **pp);
void allocate_T(void **pp, const int len, const size_t sizeofT);
void reallocate_T(void **pp, int *curlen, const int newlen, const float fac, const size_t sizeofT);
void copy_HtoD_T(void *h_array, void *d_array, int array_len, /*cudaStream_t stream, */
		 const size_t sizeofT);
void copy_DtoH_T(void *d_array, void *h_array, const int array_len, const size_t sizeofT);
void clear_gpu_array_T(void *data, const int ndata, /*cudaStream_t stream, */ const size_t sizeofT);
void copy3D_HtoD_T(void* h_data, void* d_data, int x0, int x1, int y0, int y1, int z0, int z1,
		   size_t sizeofT);

void copy3D_DtoH_T(void* src_data, void* dst_data,
		   int src_x0, int src_y0, int src_z0,
		   size_t src_xsize, size_t src_ysize,
		   int dst_x0, int dst_x1, int dst_y0, int dst_y1, int dst_z0, int dst_z1,
		   size_t dst_xsize, size_t dst_ysize,
		   size_t sizeofT);

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
// curlen = current length of the array
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
void copy_HtoD(T *h_array, T *d_array, int array_len /*, cudaStream_t stream=0*/) {
  copy_HtoD_T(h_array, d_array, array_len, /*stream, */ sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
template <class T>
void copy_DtoH(T *d_array, T *h_array, const int array_len) {
  copy_DtoH_T(d_array, h_array, array_len, sizeof(T));
}

//----------------------------------------------------------------------------------------

template <class T>
void clear_gpu_array(T *data, const int ndata /*, cudaStream_t stream=0*/) {
  clear_gpu_array_T(data, ndata, /*stream, */ sizeof(T));
}

//----------------------------------------------------------------------------------------

template <class T>
void copy3D_HtoD(T* h_data, T* d_data, int x0, int x1, int y0, int y1, int z0, int z1) {
  copy3D_HtoD_T(h_data, d_data, x0, x1, y0, y1, z0, z1, sizeof(T));
}

//----------------------------------------------------------------------------------------

template <class T>
void copy3D_DtoH(T* src_data, T* dst_data,
		 int src_x0, int src_y0, int src_z0,
		 size_t src_xsize, size_t src_ysize,
		 int dst_x0, int dst_x1, int dst_y0, int dst_y1, int dst_z0, int dst_z1,
		 size_t dst_xsize, size_t dst_ysize) {
  copy3D_DtoH_T(src_data, dst_data, src_x0, src_y0, src_z0,
		src_xsize, src_ysize,
		dst_x0, dst_x1, dst_y0, dst_y1, dst_z0, dst_z1,
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

void start_gpu(int numnode, int mynode);
