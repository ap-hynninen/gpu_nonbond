#include <iostream>
#include <fstream>
#include <cassert>
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "XYZQ.h"

//
// XYZQ class method definitions
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//
//

//
// Copies x, y, z coordinates into xyzq -array
//
__global__ void set_xyz_kernel(const int ncoord, const int stride,
			       const double* __restrict__ xyz,
			       float4* __restrict__ xyzq) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < ncoord) {
    float x = xyz[tid];
    float y = xyz[tid + stride];
    float z = xyz[tid + stride*2];
    xyzq[tid].x = x;
    xyzq[tid].y = y;
    xyzq[tid].z = z;
  }
}

//##########################################################################################
//##########################################################################################
//##########################################################################################

int XYZQ::get_xyzq_len() {
  return ((ncoord-1)/align+1)*align;
}

//
// Class creator
//
XYZQ::XYZQ() {
  ncoord = 0;
  xyzq_len = 0;
  align = 32;
  xyzq = NULL;
}

//
// Class creator
//
XYZQ::XYZQ(int ncoord, int align) : ncoord(ncoord), align(align) {
  xyzq_len = get_xyzq_len();
  allocate<float4>(&xyzq, xyzq_len);
}

//
// Class creator
//
XYZQ::XYZQ(const char *filename, int align) : align(align) {
  
  std::ifstream file(filename);
  if (file.is_open()) {
    
    float x, y, z, q;
    
    // Count number of coordinates
    ncoord = 0;
    while (file >> x >> y >> z >> q) ncoord++;

    // Rewind
    file.clear();
    file.seekg(0, std::ios::beg);
    
    // Allocate CPU memory
    float4 *xyzq_cpu = new float4[ncoord];
    
    // Read coordinates
    int i=0;
    while (file >> xyzq_cpu[i].x >> xyzq_cpu[i].y >> xyzq_cpu[i].z >> xyzq_cpu[i].w) i++;
    
    // Allocate GPU memory
    xyzq_len = get_xyzq_len();
    allocate<float4>(&xyzq, xyzq_len);

    // Copy coordinates from CPU to GPU
    copy_HtoD<float4>(xyzq_cpu, xyzq, ncoord);

    // Deallocate CPU memory
    delete [] xyzq_cpu;
    
  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }
  
}

//
// Class destructor
//
XYZQ::~XYZQ() {
  if (xyzq != NULL) deallocate<float4>(&xyzq);
}

//
// Set ncoord
//
void XYZQ::set_ncoord(int ncoord, float fac) {
  this->ncoord = ncoord;
  int req_xyzq_len = get_xyzq_len();
  
  reallocate<float4>(&xyzq, &xyzq_len, req_xyzq_len, fac);
}

//
// Copies xyzq from host
// NOTE: Does not reallocate xyzq
//
void XYZQ::set_xyzq(int ncopy, float4 *h_xyzq, size_t offset, cudaStream_t stream) {
  copy_HtoD<float4>(&h_xyzq[offset], &xyzq[offset], ncopy, stream);
}

//
// Copies xyz (on device) into the coordinate slots
//
void XYZQ::set_xyz(const double *xyz, const int stride, cudaStream_t stream) {
  int nthread = 512;
  int nblock = (ncoord-1)/nthread+1;

  set_xyz_kernel<<< nblock, nthread, 0, stream >>>(ncoord, stride, xyz, xyzq);

  cudaCheck(cudaGetLastError());
}

//
// Compares two XYZQ arrays
//
bool XYZQ::compare(XYZQ& xyzq_in, const double tol, double& max_diff) {
  assert(xyzq_in.ncoord == ncoord);

  float4 *h_xyzq = new float4[ncoord];
  float4 *h_xyzq_in = new float4[ncoord];
  copy_DtoH<float4>(xyzq, h_xyzq, ncoord);
  copy_DtoH<float4>(xyzq_in.xyzq, h_xyzq_in, ncoord);

  bool ok = true;

  max_diff = 0.0;
  int i;
  double dx, dy, dz, dq;
  double diff;
  try {
    for (i=0;i < ncoord;i++) {
      dx = fabs(h_xyzq[i].x - h_xyzq_in[i].x);
      dy = fabs(h_xyzq[i].y - h_xyzq_in[i].y);
      dz = fabs(h_xyzq[i].z - h_xyzq_in[i].z);
      dq = fabs(h_xyzq[i].w - h_xyzq_in[i].w);
      diff = max(dx, max(dy, dz));
      max_diff = max(max_diff, diff);
      if (diff > tol || dq > 0.0) throw 1;
    }
  }
  catch (int a) {
    std::cout << "i = " << i << std::endl;
    std::cout << "this: x,y,z,q = " << h_xyzq[i].x << " " << h_xyzq[i].y
	      << " " << h_xyzq[i].z << " " << h_xyzq[i].w << std::endl;
    std::cout << "in  : x,y,z,q = " << h_xyzq_in[i].x << " " << h_xyzq_in[i].y
	      << " " << h_xyzq_in[i].z << " " << h_xyzq_in[i].w << std::endl;
    ok = false;
  }

  delete [] h_xyzq;
  delete [] h_xyzq_in;

  return ok;
}
