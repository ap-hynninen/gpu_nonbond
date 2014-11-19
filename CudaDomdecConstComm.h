//
// CUDA Domdec constraint communicator class
// (c) Antti-Pekka Hynninen, 2014
//
#include <vector>
#include <mpi.h>
#include "cudaXYZ.h"
#include "CudaMPI.h"
#include "Domdec.h"

class CudaDomdecConstComm {
private:

  // CUDA MPI
  CudaMPI& cudaMPI;

  // Number of neighbors
  int numNeigh;

  // Neighboring node index (0...6)
  std::vector<int> neighInd;

  // Neighboring node MPI index
  std::vector<int> rightMPInode;
  std::vector<int> leftMPInode;

  // MPI requests
  std::vector<MPI_Request> request;

  // Number of coordinates we are receiving from neighbors
  std::vector<int> numRecv;
  std::vector<int> posRecv;

  // Number of coordinates we are sending to neighbors
  std::vector<int> numSend;
  std::vector<int> posSend;

  // Total number of coordiantes we are sending/receiving
  int numSendTot;
  int numRecvTot;

  // Received coordinates' indices
  int recvCoordIndLen;
  int* recvCoordInd;

  // Sent coordinates' indices
  int sendCoordIndLen;
  int* sendCoordInd;

  // Coordinate buffers for MPI communications
  int sendCoordBufLen;
  double* sendCoordBuf;

  int recvCoordBufLen;
  double* recvCoordBuf;

  // Buffers for GPU-GPU communication (needed if MPI is not CUDA-aware)
  int h_sendBufLen;
  char* h_sendBuf;
  int h_recvBufLen;
  char* h_recvBuf;

  int mynode;

  void packCoordBuf(const int n, const int* coordInd,
		    cudaXYZ<double>& coord, double* coordBuf,
		    cudaStream_t stream);
  void unpackCoordBuf(const int n, const int* coordInd,
		      const double* coordBuf, cudaXYZ<double>& coord,
		      cudaStream_t stream);

  void commCoordBuf(std::vector<int>& sendMPInodeLoc, std::vector<int>& recvMPInodeLoc,
		    std::vector<int>& numSendLoc, std::vector<int>& posSendLoc,
		    std::vector<int>& numRecvLoc, std::vector<int>& posRecvLoc,
		    double* sendCoordBufLoc, double* recvCoordBufLoc,
		    double* h_sendBufLoc, double* h_recvBufLoc);

public:
  CudaDomdecConstComm(const Domdec& domdec, CudaMPI& cudaMPI);
  ~CudaDomdecConstComm();

  void setup(const int* neighPos, int* coordInd, const int* glo2loc,
	     cudaStream_t stream);

  void communicate(const int dir, cudaXYZ<double>& coord, cudaStream_t stream);
};
