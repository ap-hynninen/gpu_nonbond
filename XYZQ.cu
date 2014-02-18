#include <iostream>
#include <fstream>
#include "cuda_utils.h"
#include "XYZQ.h"

//
// XYZQ class method definitions
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//
//

int XYZQ::get_xyzq_len() {
  return ((ncoord-1)/align+1)*align;
}

XYZQ::XYZQ() {
  ncoord = 0;
  xyzq_len = 0;
  align = 32;
  xyzq = NULL;
}

XYZQ::XYZQ(int ncoord, int align) : ncoord(ncoord), align(align) {
  xyzq_len = get_xyzq_len();
  allocate<float4>(&xyzq, xyzq_len);
}

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
