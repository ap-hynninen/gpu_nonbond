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
  
  // Local indices that we send to -z direction, initial version
  std::vector< thrust::device_vector<int> > z_send_loc0;

  // Local indices that we send to -z direction
  thrust::device_vector<int> z_send_loc;

  // Global indices that we send to -z direction
  thrust::device_vector<int> z_send_glo;

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
  std::vector<int> nsendByte;
  std::vector<int> psendByte;

  // Number of items to recv and position in bytes
  std::vector<int> nrecvByte;
  std::vector<int> precvByte;

  void computeByteNumPos(const int num_comm, std::vector<int>& ncomm,
			 std::vector<int>& ncommByte, std::vector<int>& pcommByte,
			 const bool update);

  void packXYZ(double* x, double* y, double* z, int* ind, int n, double *outbuf,
	       cudaStream_t stream);
  void unpackXYZ(double* inbuf, int n, double* x, double* y, double* z,
		 cudaStream_t stream, int* ind=NULL);
  void unpackForce(double* inbuf, int n, double* x, double* y, double* z, int* ind,
		   cudaStream_t stream);

 public:

  CudaDomdecD2DComm(Domdec& domdec, CudaMPI& cudaMPI);
  ~CudaDomdecD2DComm();

  void comm_coord(cudaXYZ<double>& coord, thrust::device_vector<int>& loc2glo,
		  const bool update, cudaStream_t stream=0);
  void comm_update(int* glo2loc, cudaStream_t stream=0);
  void comm_force(Force<long long int>& force, cudaStream_t stream=0);

  void test_comm_coord(const int* glo2loc, cudaXYZ<double>& coord);
};

#endif // CUDADOMDECD2DCOMM_H
