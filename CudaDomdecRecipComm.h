#ifndef CUDADOMDECRECIPCOMM_H
#define CUDADOMDECRECIPCOMM_H

#include <cassert>
#include "DomdecRecipComm.h"
#include "CudaMPI.h"
#include "Force.h"

class CudaDomdecRecipComm : public DomdecRecipComm {

 private:
  CudaMPI cudaMPI;

  // Coordinate buffer
  int coord_len;
  float4 *coord;

  // Host send buffer (only used when MPI implementation is not CUDA Aware)
  int h_sendbuf_len;
  char *h_sendbuf;

  int h_recvbuf_len;
  char *h_recvbuf;

 public:
  CudaDomdecRecipComm(MPI_Comm comm_recip, MPI_Comm comm_direct_recip,
		      int mynode, std::vector<int>& direct_nodes, std::vector<int>& recip_nodes,
		      const bool cudaAware);
  ~CudaDomdecRecipComm();

  void comm_coord(float4* coord_in);
  void recv_force(Force<long long int>& force_in);

  float4* get_coord() {
    assert(isRecip);
    return coord;
  }

  int get_ncoord() {
    assert(isRecip);
    return ncoord;
  }

};

#endif // CUDADOMDECRECIPCOMM_H
