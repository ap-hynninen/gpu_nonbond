
#include <iostream>
#include "mpi_utils.h"
#include "cuda_utils.h"
#include "CudaDomdecRecipComm.h"

CudaDomdecRecipComm::CudaDomdecRecipComm(MPI_Comm comm_recip, MPI_Comm comm_direct_recip,
		     int mynode, std::vector<int>& direct_nodes, std::vector<int>& recip_nodes,
		     const bool cudaAware) : 
  DomdecRecipComm(comm_recip, comm_direct_recip, mynode, direct_nodes, recip_nodes),
    cudaMPI(cudaAware, comm_direct_recip) {
  coord_len = 0;
  coord = NULL;
  h_sendbuf_len = 0;
  h_sendbuf = NULL;
  h_recvbuf_len = 0;
  h_recvbuf = NULL;
}

CudaDomdecRecipComm::~CudaDomdecRecipComm() {
  if (coord_len > 0) deallocate<float4>(&coord);
  if (h_sendbuf != NULL) deallocate_host<char>(&h_sendbuf);
  if (h_recvbuf != NULL) deallocate_host<char>(&h_recvbuf);
}

//
// Communicate coordinates Direct -> Recip
//
void CudaDomdecRecipComm::comm_coord(float4* coord_in) {

  const int TAG = 1;
  
  if (isDirect && !isRecip) {
    //------------------------------------------------
    // Pure Direct node => Send coordinates to Recip
    //------------------------------------------------

    // Re-allocate h_sendbuf if needed
    if (!cudaMPI.isCudaAware()) {
      reallocate_host<char>(&h_sendbuf, &h_sendbuf_len, ncomm.at(0)*sizeof(float4), 1.2f);
    }

    MPICheck(cudaMPI.Send((void *)coord_in, ncomm.at(0)*sizeof(float4),
			  recip_nodes.at(0), TAG, h_sendbuf));
  }

  if (isDirect && isRecip) {
    //------------------------------------------------
    // Direct+Recip node => Copy coordinates or set pointer
    //------------------------------------------------
    if (direct_nodes.size() == 1) {
      // Only a single Direct node => set pointer and we are done!
      coord = coord_in;
      return;
    } else {
      float fac = (recip_nodes.size() == 1) ? 1.0f : 1.2f;
      // Allocate memory for receiving coordinates
      reallocate<float4>(&coord, &coord_len, ncoord*sizeof(float4), fac);
      // Re-allocate h_recvbuf if needed
      if (!cudaMPI.isCudaAware()) {
	reallocate_host<char>(&h_recvbuf, &h_recvbuf_len, ncoord*sizeof(float4), fac);
      }
    }
  }

  if (isRecip) {
    //------------------------------------------------
    // Recip node => Receive coordinates from Direct
    //------------------------------------------------
    for (int i=0;i < direct_nodes.size();i++) {
      if (mynode != direct_nodes.at(i)) {
	// Receive via MPI
	MPICheck(cudaMPI.Recv(&coord[pcomm.at(i)], ncomm.at(i)*sizeof(float4),
			      direct_nodes.at(i), TAG, MPI_STATUS_IGNORE, &h_recvbuf[pcomm.at(i)]));
      } else {
	// Copy
	copy_DtoD<float4>(coord_in, &coord[pcomm.at(i)], ncomm.at(i));
      }
    }
  }

}

//
// Communicate forces Recip -> Direct
//
void CudaDomdecRecipComm::recv_force(Force<long long int>& force_in) {

  const int TAG = 1;
  /*
  if (isRecip) {
    //---------------------------------------------------
    // Recip node => Send forces to Direct nodes
    //---------------------------------------------------
    for (int i=0;i < direct_nodes.size();i++) {
      if (mynode != direct_nodes.at(i)) {
	// Send via MPI
	MPICheck(cudaMPI.Send(force_in, ncomm.at(0)*sizeof(float4),
			      recip_nodes.at(0), TAG, h_sendbuf));
      } else {
	// Copy
	copy_DtoD<>();
      }
    }
  }

  if (isDirect) {
    //---------------------------------------------------
    // Direct node => Receive forces from Recip node
    //---------------------------------------------------
    MPICheck(cudaMPI.Recv(force_in, ncomm.at(0)*sizeof(double),
			  recip_nodes.at(0), TAG, MPI_STATUS_IGNORE, h_sendbuf));

  }
  */

}
