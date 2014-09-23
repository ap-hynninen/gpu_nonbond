#ifndef CUDADOMDECRECIPCOMM_H
#define CUDADOMDECRECIPCOMM_H

#include <cassert>
#include "DomdecRecipComm.h"
#include "CudaMPI.h"
#include "Force.h"

class CudaDomdecRecipComm : public DomdecRecipComm {

 private:
  CudaMPI cudaMPI;

  // Host communication buffer (only used when MPI implementation is not CUDA Aware)
  int h_commbuf_len;
  char *h_commbuf;

  // Pure storage pointers, these are not allocated or deallocated by this class
  // pointer for copy_DtoD()
  float4 *coord_copy_ptr;
  // pointer for where coordinates are found after recv_coord() -call
  float4 *coord_ptr;
  // pointer to where recip forces are
  float3 *force_ptr;

 public:
  CudaDomdecRecipComm(MPI_Comm comm_recip, MPI_Comm comm_direct_recip,
		      int mynode, std::vector<int>& direct_nodes, std::vector<int>& recip_nodes,
		      const bool cudaAware);
  ~CudaDomdecRecipComm();

  float4 *get_coord_ptr() {assert(coord_ptr != NULL);return coord_ptr;}
  float3 *get_force_ptr() {assert(force_ptr != NULL);return force_ptr;}

  void send_coord(float4* coord);
  void recv_coord(float4* coord);
  void send_force(float3* force);
  void recv_force(float3* force);

};

#endif // CUDADOMDECRECIPCOMM_H
