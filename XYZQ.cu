#include <iostream>
#include <fstream>
#include <cuda.h>
#include "gpu_utils.h"
#include "XYZQ.h"

//
// XYZQ class method definitions
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//
//

XYZQ::XYZQ(int ncoord) : ncoord(ncoord) {
  allocate<float4>(&xyzq, ncoord);
}

XYZQ::XYZQ(const char *filename) {
  
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
    allocate<float4>(&xyzq, ncoord);

    // Copy coordinates from CPU to GPU
    copy_HtoD<float4>(xyzq_cpu, xyzq, ncoord);

    // Deallocate CPU memory
    delete [] xyzq_cpu;
    
  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }
  
}

XYZQ::~XYZQ() {
  deallocate<float4>(&xyzq);
}
