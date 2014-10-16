
#include <iostream>
#include "mpi_utils.h"
#include "cuda_utils.h"
#include "CudaDomdecRecipComm.h"

CudaDomdecRecipComm::CudaDomdecRecipComm(MPI_Comm comm_recip, MPI_Comm comm_direct_recip,
		     int mynode, std::vector<int>& direct_nodes, std::vector<int>& recip_nodes,
		     const bool cudaAware) : 
  DomdecRecipComm(comm_recip, comm_direct_recip, mynode, direct_nodes, recip_nodes),
    cudaMPI(cudaAware, comm_direct_recip) {
  h_commbuf_len = 0;
  h_commbuf = NULL;
  coord_copy_ptr = NULL;
  coord_ptr = NULL;
}

CudaDomdecRecipComm::~CudaDomdecRecipComm() {
  if (h_commbuf != NULL) deallocate_host<char>(&h_commbuf);
}

//
// Send coordinates to Recip from coord[]
//
void CudaDomdecRecipComm::send_coord(float4* coord) {

  const int TAG = 1;

  if (!isDirect) {
    std::cout << "CudaDomdecRecipComm::send_coord, only direct nodes are allowed here" << std::endl;
    exit(1);
  }

  coord_ptr = NULL;
  coord_copy_ptr = NULL;

  /*
  if (isDirect && isRecip && direct_nodes.size() == 1) {
    //-----------------------------------------------------------
    // Only a single Direct node => set pointer and we are done!
    //-----------------------------------------------------------
    coord_ptr = coord;
    return;
  }
  */

  if (isRecip) {
    //------------------------------------------------
    // Mixed Direct+Recip node => Copy coordinates
    //------------------------------------------------
    coord_copy_ptr = coord;
  } else {
    //------------------------------------------------
    // Pure Direct node => Send coordinates via MPI
    //------------------------------------------------
    // Re-allocate h_commbuf if needed
    if (!cudaMPI.isCudaAware()) {
      reallocate_host<char>(&h_commbuf, &h_commbuf_len, ncomm.at(0)*sizeof(float4), 1.2f);
    }
    // Send
    MPICheck(cudaMPI.Send((void *)coord, ncomm.at(0)*sizeof(float4),
			  recip_nodes.at(0), TAG, h_commbuf));
  }

}

//
// Recv coordinates from Direct to coord[]
//
void CudaDomdecRecipComm::recv_coord(float4* coord) {

  const int TAG = 1;

  if (!isRecip) {
    std::cout << "CudaDomdecRecipComm::recv_coord, only recip nodes are allowed here" << std::endl;
    exit(1);
  }

  /*
  if (isDirect && isRecip && direct_nodes.size() == 1) {
    //-----------------------------------------------------------
    // Only a single Direct node => set pointer and we are done!
    // NOTE: coord_ptr was already set in the call to send_coord()
    //-----------------------------------------------------------
    if (coord_ptr == NULL) {
      std::cout << "CudaDomdecRecipComm::recv_coord, coord_ptr should have been set by send_coord" << std::endl;
      exit(1);
    }
    return;
  }
  */
  
  coord_ptr = NULL;

  if (isRecip) {
    //------------------------------------------------
    // Recip node => Receive coordinates from Direct
    //------------------------------------------------
    // Re-allocate h_commbuf if needed
    if (!cudaMPI.isCudaAware()) {
      // Count the required h_commbuf size
      int ncoord_buf = 0;
      for (int i=0;i < direct_nodes.size();i++) {
	if (mynode != direct_nodes.at(i)) ncoord_buf += ncomm.at(i);
      }
      float fac = (recip_nodes.size() == 1) ? 1.0f : 1.2f;
      reallocate_host<char>(&h_commbuf, &h_commbuf_len, ncoord_buf*sizeof(float4), fac);
    }
    float4* h_coordbuf = (float4 *)h_commbuf;
    int h_coordbuf_pos = 0;
    for (int i=0;i < direct_nodes.size();i++) {
      if (mynode != direct_nodes.at(i)) {
	// Receive via MPI
	MPICheck(cudaMPI.Recv(&coord[pcomm.at(i)], ncomm.at(i)*sizeof(float4),
			      direct_nodes.at(i), TAG, MPI_STATUS_IGNORE,
			      &h_coordbuf[h_coordbuf_pos]));
	h_coordbuf_pos += ncomm.at(i);
      } else {
	// Copy device buffer
	assert(coord_copy_ptr != NULL);
	copy_DtoD_sync<float4>(coord_copy_ptr, &coord[pcomm.at(i)], ncomm.at(i));
      }
    }
    // Store pointer to where coordinates are found
    coord_ptr = coord;
  }

}

//
// Send forces to Direct
//
void CudaDomdecRecipComm::send_force(float3* force) {

  const int TAG = 1;

  if (!isRecip) {
    std::cout << "CudaDomdecRecipComm::send_force, only recip nodes are allowed here" << std::endl;
    exit(1);
  }

  // Re-allocate h_commbuf if needed
  if (!cudaMPI.isCudaAware()) {
    float fac = (recip_nodes.size() == 1) ? 1.0f : 1.2f;
    reallocate_host<char>(&h_commbuf, &h_commbuf_len, pcomm.at(direct_nodes.size())*sizeof(float3), fac);
  }

  //---------------------------------------------------
  // Recip node => Send forces to Direct nodes
  //---------------------------------------------------
  float3* h_coordbuf = (float3 *)h_commbuf;
  for (int i=0;i < direct_nodes.size();i++) {
    if (mynode != direct_nodes.at(i)) {
      // Send via MPI
      MPICheck(cudaMPI.Send(&force[pcomm.at(i)], ncomm.at(i)*sizeof(float3),
			    direct_nodes.at(i), TAG, &h_coordbuf[pcomm.at(i)]));
    }
  }

}

//
// Receive forces from Direct
//
void CudaDomdecRecipComm::recv_force(float3* force) {

  const int TAG = 1;

  if (!isDirect) {
    std::cout << "CudaDomdecRecipComm::recv_force, only direct nodes are allowed here" << std::endl;
    exit(1);
  }

  force_ptr = NULL;

  //---------------------------------------------------
  // Direct node => Receive forces from Recip node
  //---------------------------------------------------
  if (mynode != recip_nodes.at(0)) {
    if (isRecip) {
      std::cout << "CudaDomdecRecipComm::recv_force, must be pure direct node to be here" << std::endl;
      exit(1);
    }
    // Re-allocate h_commbuf if needed
    if (!cudaMPI.isCudaAware()) {
      reallocate_host<char>(&h_commbuf, &h_commbuf_len, ncomm.at(0)*sizeof(float3), 1.2f);
    }
    // Reveice via MPI
    MPICheck(cudaMPI.Recv(force, ncomm.at(0)*sizeof(float3),
			  recip_nodes.at(0), TAG, MPI_STATUS_IGNORE, h_commbuf));

    force_ptr = force;
  } else {
    if (!isRecip) {
      std::cout << "CudaDomdecRecipComm::recv_force, must be direct+recip node to be here" << std::endl;
      exit(1);
    }
    // This is a Direct+Recip node. No need to receive forces via MPI since they are
    // already in the force buffer, just need to get the address:
    // Get a pointer where the forces are stored on a Direct+Recip node
    force_ptr = &force[pcomm.at(imynode)];
  }

}
