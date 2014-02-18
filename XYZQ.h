#ifndef XYZQ_H
#define XYZQ_H

//
// XYZQ class
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//
//

#include <cuda.h>

class XYZQ {

private:
  int get_xyzq_len();

public:
  int align;
  int ncoord;
  int xyzq_len;
  float4 *xyzq;

  XYZQ();
  XYZQ(int ncoord, int align=1);
  XYZQ(const char *filename, int align=1);
  ~XYZQ();

  void set_ncoord(int ncoord, float fac=1.0f);
  void set_xyzq(int ncopy, float4 *h_xyzq, size_t offset=0, cudaStream_t stream=0);
};

#endif // XYZQ_H
