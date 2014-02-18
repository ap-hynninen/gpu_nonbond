#include <iostream>
#include <cuda.h>
#include <nvToolsExtCuda.h>
#include "gpu_utils.h"
#include "cuda_utils.h"

//----------------------------------------------------------------------------------------
//
// Deallocate page-locked host memory
// pp = memory pointer
//
void deallocate_host_T(void **pp) {
  
  if (*pp != NULL) {
    cudaCheck(cudaFreeHost((void *)(*pp)));
    *pp = NULL;
  }

}
//----------------------------------------------------------------------------------------
//
// Allocate page-locked host memory
// pp = memory pointer
// len = length of the array
//
void allocate_host_T(void **pp, const int len, const size_t sizeofT) {
  cudaCheck(cudaMallocHost(pp, sizeofT*len));
}

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate page-locked host memory
// pp = memory pointer
// curlen = current length of the array
// newlen = new required length of the array
// fac = extra space allocation factor: in case of re-allocation new length will be fac*newlen
//
void reallocate_host_T(void **pp, int *curlen, const int newlen, const float fac, const size_t sizeofT) {

  if (*pp != NULL && *curlen < newlen) {
    cudaCheck(cudaFreeHost((void *)(*pp)));
    *pp = NULL;
  }

  if (*pp == NULL) {
    *curlen = (int)(float(newlen)*fac);
    allocate_host_T(pp, *curlen, sizeofT);
  }

}

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
void deallocate_T(void **pp) {
  
  if (*pp != NULL) {
    cudaCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }

}
//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
void allocate_T(void **pp, const int len, const size_t sizeofT) {
  cudaCheck(cudaMalloc(pp, sizeofT*len));
}

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate gpu memory
// pp = memory pointer
// curlen = current length of the array
// newlen = new required length of the array
// fac = extra space allocation factor: in case of re-allocation new length will be fac*newlen
//
void reallocate_T(void **pp, int *curlen, const int newlen, const float fac, const size_t sizeofT) {

  if (*pp != NULL && *curlen < newlen) {
    cudaCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }

  if (*pp == NULL) {
    *curlen = (int)(float(newlen)*fac);
    allocate_T(pp, *curlen, sizeofT);
  }

}
//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device
//
void copy_HtoD_async_T(void *h_array, void *d_array, int array_len, cudaStream_t stream,
		       const size_t sizeofT) {
  cudaCheck(cudaMemcpyAsync(d_array, h_array, sizeofT*array_len,
			    cudaMemcpyHostToDevice, stream));
}

void copy_HtoD_T(void *h_array, void *d_array, int array_len,
		 const size_t sizeofT) {
  cudaCheck(cudaMemcpy(d_array, h_array, sizeofT*array_len,
		       cudaMemcpyHostToDevice));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
void copy_DtoH_async_T(void *d_array, void *h_array, const int array_len, cudaStream_t stream,
		       const size_t sizeofT) {
  cudaCheck(cudaMemcpyAsync(h_array, d_array, sizeofT*array_len, cudaMemcpyDeviceToHost, stream));
}

void copy_DtoH_T(void *d_array, void *h_array, const int array_len, const size_t sizeofT) {
  cudaCheck(cudaMemcpy(h_array, d_array, sizeofT*array_len, cudaMemcpyDeviceToHost));
}

//----------------------------------------------------------------------------------------

void clear_gpu_array_async_T(void *data, const int ndata, cudaStream_t stream, const size_t sizeofT) {
  cudaCheck(cudaMemsetAsync(data, 0, sizeofT*ndata, stream));
}

void clear_gpu_array_T(void *data, const int ndata, const size_t sizeofT) {
  cudaCheck(cudaMemset(data, 0, sizeofT*ndata));
}

//----------------------------------------------------------------------------------------

void copy3D_HtoD_T(void* src_data, void* dst_data,
		   int src_x0, int src_y0, int src_z0,
		   size_t src_xsize, size_t src_ysize,
		   int dst_x0, int dst_y0, int dst_z0,
		   size_t width, size_t height, size_t depth,
		   size_t dst_xsize, size_t dst_ysize,
		   size_t sizeofT) {
  cudaMemcpy3DParms parms = {0};

  parms.srcPos = make_cudaPos(sizeofT*src_x0, src_y0, src_z0);
  parms.srcPtr = make_cudaPitchedPtr(src_data, sizeofT*src_xsize, src_xsize, src_ysize);

  parms.dstPos = make_cudaPos(sizeofT*dst_x0, dst_y0, dst_z0);
  parms.dstPtr = make_cudaPitchedPtr(dst_data, sizeofT*dst_xsize, dst_xsize, dst_ysize);

  parms.extent = make_cudaExtent(sizeofT*width, height, depth);
  parms.kind = cudaMemcpyHostToDevice;

  //  cudaCheck(cudaMemcpy3D(&parms));

  if (cudaMemcpy3D(&parms) != cudaSuccess) {
    std::cerr << "copy3D_HtoD_T" << std::endl;
    std::cerr << "source: " << std::endl;
    std::cerr << parms.srcPos.x << " " << parms.srcPos.y << " " << parms.srcPos.z << std::endl;
    std::cerr << parms.srcPtr.pitch << " " << parms.srcPtr.xsize << " "<< parms.srcPtr.ysize << std::endl;
    std::cerr << "destination: " << std::endl;
    std::cerr << parms.dstPos.x << " " << parms.dstPos.y << " " << parms.dstPos.z << std::endl;
    std::cerr << parms.dstPtr.pitch << " " << parms.dstPtr.xsize << " "<< parms.dstPtr.ysize << std::endl;
    std::cerr << "extent: " << std::endl;
    std::cerr << parms.extent.width << " "<< parms.extent.height << " "<< parms.extent.depth << std::endl;
    exit(1);
  }
}

//----------------------------------------------------------------------------------------

void copy3D_DtoH_T(void* src_data, void* dst_data,
		   int src_x0, int src_y0, int src_z0,
		   size_t src_xsize, size_t src_ysize,
		   int dst_x0, int dst_y0, int dst_z0,
		   size_t width, size_t height, size_t depth,
		   size_t dst_xsize, size_t dst_ysize,
		   size_t sizeofT) {
  cudaMemcpy3DParms parms = {0};

  parms.srcPos = make_cudaPos(sizeofT*src_x0, src_y0, src_z0);
  parms.srcPtr = make_cudaPitchedPtr(src_data, sizeofT*src_xsize, src_xsize, src_ysize);

  parms.dstPos = make_cudaPos(sizeofT*dst_x0, dst_y0, dst_z0);
  parms.dstPtr = make_cudaPitchedPtr(dst_data, sizeofT*dst_xsize, dst_xsize, dst_ysize);

  parms.extent = make_cudaExtent(sizeofT*width, height, depth);
  parms.kind = cudaMemcpyDeviceToHost;

  //  cudaCheck(cudaMemcpy3D(&parms));
  if (cudaMemcpy3D(&parms) != cudaSuccess) {
    std::cerr << "copy3D_DtoH_T" << std::endl;
    std::cerr << "source: " << std::endl;
    std::cerr << parms.srcPos.x << " " << parms.srcPos.y << " " << parms.srcPos.z << std::endl;
    std::cerr << parms.srcPtr.pitch << " " << parms.srcPtr.xsize << " "<< parms.srcPtr.ysize << std::endl;
    std::cerr << "destination: " << std::endl;
    std::cerr << parms.dstPos.x << " " << parms.dstPos.y << " " << parms.dstPos.z << std::endl;
    std::cerr << parms.dstPtr.pitch << " " << parms.dstPtr.xsize << " "<< parms.dstPtr.ysize << std::endl;
    std::cerr << "extent: " << std::endl;
    std::cerr << parms.extent.width << " "<< parms.extent.height << " "<< parms.extent.depth << std::endl;
    exit(1);
  }
}

//----------------------------------------------------------------------------------------

void gpu_range_start(char *range_name) {
  static int color_id=0;
  nvtxEventAttributes_t att;
  att.version = NVTX_VERSION;
  att.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  att.colorType = NVTX_COLOR_ARGB;
  if (color_id == 0) {
    att.color = 0xFFFF0000;
  } else if (color_id == 1) {
    att.color = 0xFF00FF00;
  } else if (color_id == 2) {
    att.color = 0xFF0000FF;
  } else if (color_id == 3) {
    att.color = 0xFFFF00FF;
  }
  color_id++;
  if (color_id > 3) color_id = 0;
  att.messageType = NVTX_MESSAGE_TYPE_ASCII;
  att.message.ascii = range_name;
  nvtxRangePushEx(&att);
}

void gpu_range_stop() {
  nvtxRangePop();
}

//----------------------------------------------------------------------------------------

static int gpu_ind = -1;
static cudaDeviceProp gpu_prop;

void start_gpu(int numnode, int mynode) {
  int devices[4] = {2, 3, 0, 1};

  int device_count;
  cudaCheck(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    std::cout << "No CUDA device found" << std::endl;
    exit(1);
  }

  gpu_ind = devices[mynode % 4];
  cudaCheck(cudaSetDevice(gpu_ind));

  cudaCheck(cudaThreadSynchronize());
  
  cudaCheck(cudaGetDeviceProperties(&gpu_prop, gpu_ind));

  if (gpu_prop.major < 2) {
    std::cout << "CUDA device(s) must have compute capability 2.0 or higher" << std::endl;
    exit(1);
  }

  int cuda_driver_version;
  cudaCheck(cudaDriverGetVersion(&cuda_driver_version));

  int cuda_rt_version;
  cudaCheck(cudaRuntimeGetVersion(&cuda_rt_version));

  if (mynode == 0) {
    std::cout << "Number of CUDA devices found " << device_count << std::endl;
    std::cout << "Using CUDA driver version " << cuda_driver_version << std::endl;
    std::cout << "Using CUDA runtime version " << cuda_rt_version << std::endl;
  }

  std::cout << "Node " << mynode << " uses CUDA device " << gpu_ind << 
    " " << gpu_prop.name << std::endl;
  
}

void stop_gpu() {
  cudaCheck(cudaDeviceReset());
  gpu_ind = -1;
}

int3 get_max_nblock() {
  int3 max_nblock;
  max_nblock.x = gpu_prop.maxGridSize[0];
  max_nblock.y = gpu_prop.maxGridSize[1];
  max_nblock.z = gpu_prop.maxGridSize[2];
  return max_nblock;
}

int get_major() {
  return gpu_prop.major;
}

int get_gpu_ind() {
  return gpu_ind;
}
