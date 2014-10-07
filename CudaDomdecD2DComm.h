#ifndef CUDADOMDECD2DCOMM_H
#define CUDADOMDECD2DCOMM_H

#include <vector>
#include <thrust/device_vector.h>
#include "cudaXYZ.h"
#include "Domdec.h"
#include "CudaMPI.h"
#include "Force.h"
#include "DomdecD2DComm.h"

class CudaDomdecD2DComm : public DomdecD2DComm {

 private:
  
  // Cuda MPI
  CudaMPI& cudaMPI;

  thrust::device_vector<unsigned char> atom_pick;
  thrust::device_vector<int> atom_pos;
  
  // Local indices that we send
  // Initial version
  std::vector< thrust::device_vector<int> > z_send_loc0;
  // Final version, built after neighborlist has re-ordered coordinates
  thrust::device_vector<int> z_send_loc;  

  // Local indices that we receive from +z direction
  thrust::device_vector<int> z_recv_loc;

  // Global coordinate indices we receive from +z direction
  thrust::device_vector<int> z_recv_glo;

  // Sending buffer
  int sendbuf_len;
  char *sendbuf;
  
  // Host sending buffer, neccessary for non-Cuda Aware MPI setups
  int h_sendbuf_len;
  char *h_sendbuf;

  // Receiving buffer
  int recvbuf_len;
  char *recvbuf;
  
  // Host receiving buffer, neccessary for non-Cuda Aware MPI setups
  int h_recvbuf_len;
  char *h_recvbuf;

  //------------
  //std::vector<MPI_Request> request;

  // Number of items to send and position in bytes
  std::vector<int> nsend;
  std::vector<int> psend;

  // Number of items to recv and position in bytes
  std::vector<int> nrecv;
  std::vector<int> precv;

  void computeByteNumPos(const int nc_comm, std::vector<int>& c_nsend,
			 std::vector<int>& nsend, std::vector<int>& psend,
			 const bool update);

 public:

  CudaDomdecD2DComm(Domdec& domdec, CudaMPI& cudaMPI);
  ~CudaDomdecD2DComm();

  void comm_coord(cudaXYZ<double>& coord, thrust::device_vector<int>& loc2glo,
		  const bool update);
  void update(int* glo2loc, int* loc2loc);
  void comm_force(Force<long long int>& force);
};

#endif // CUDADOMDECD2DCOMM_H
