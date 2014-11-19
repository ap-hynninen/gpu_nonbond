#include <iostream>
#include <cassert>
#include <cuda.h>
#include "cuda_utils.h"
#include "CudaMPI.h"
#include "mpi_utils.h"
#include "CudaDomdecConstComm.h"

//
// Map coordInd (global) -> recvCoordInd (local)
//
__global__ void buildCoordInd_kernel(const int numCoord,
				     const int* __restrict__ coordInd,
				     const int* __restrict__ glo2loc,
				     int* __restrict__ recvCoordInd) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < numCoord) {
    int iglo = coordInd[i];
    int iloc = glo2loc[iglo];
    recvCoordInd[i] = iloc;
  }
}

//
// Pack coordinates from (x, y, z) array.
//
__global__ void packCoordBuf_kernel(const int numCoord,
				    const int* __restrict__ indCoord,
				    const double* __restrict__ x,
				    const double* __restrict__ y,
				    const double* __restrict__ z,
				    double* __restrict__ coordBuf) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < numCoord) {
    const int tid3 = tid*3;
    const int i = indCoord[tid];
    coordBuf[tid3+0] = x[i];
    coordBuf[tid3+1] = y[i];
    coordBuf[tid3+2] = z[i];
  }
}

//
// Unpack coordinates to (x, y, z) array.
//
__global__ void unpackCoordBuf_kernel(const int numCoord,
				      const int* __restrict__ indCoord,
				      const double* __restrict__ coordBuf,
				      double* __restrict__ x,
				      double* __restrict__ y,
				      double* __restrict__ z) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < numCoord) {
    const int tid3 = tid*3;
    const int i = indCoord[tid];
    x[i] = coordBuf[tid3+0];
    y[i] = coordBuf[tid3+1];
    z[i] = coordBuf[tid3+2];
  }
}

//#####################################################################################################
//#####################################################################################################
//#####################################################################################################

//
// Class creator
//
CudaDomdecConstComm::CudaDomdecConstComm(const Domdec& domdec, CudaMPI& cudaMPI) : cudaMPI(cudaMPI) {
  
  recvCoordIndLen = 0;
  recvCoordInd = NULL;

  sendCoordIndLen = 0;
  sendCoordInd = NULL;

  recvCoordBufLen = 0;
  recvCoordBuf = NULL;

  sendCoordBufLen = 0;
  sendCoordBuf = NULL;

  h_sendBufLen = 0;
  h_sendBuf = NULL;

  h_recvBufLen = 0;
  h_recvBuf = NULL;

  int nnx = (domdec.get_nx() == 1) ? 1 : 2;
  int nny = (domdec.get_ny() == 1) ? 1 : 2;
  int nnz = (domdec.get_nz() == 1) ? 1 : 2;
  numNeigh = nnx*nny*nnz - 1;

  if (numNeigh > 0) {
    rightMPInode.resize(numNeigh);
    leftMPInode.resize(numNeigh);
    neighInd.resize(numNeigh);
    request.resize(2*numNeigh);

    numRecv.resize(numNeigh);
    posRecv.resize(numNeigh);
    numSend.resize(numNeigh);
    posSend.resize(numNeigh);

    int i = 0;
    for (int ineigh=0;ineigh < 7;ineigh++) {
      int t = ineigh+1;
      int iz = t/4;
      t -= iz*4;
      int iy = t/2;
      int ix = t - iy*2;
      if (ix > 0 && domdec.get_nx() == 1) continue;
      if (iy > 0 && domdec.get_ny() == 1) continue;
      if (iz > 0 && domdec.get_nz() == 1) continue;
      neighInd.at(i) = ineigh;
      rightMPInode.at(i) = domdec.get_nodeind_pbc(domdec.get_homeix()+ix,
						 domdec.get_homeiy()+iy,
						 domdec.get_homeiz()+iz);
      leftMPInode.at(i) = domdec.get_nodeind_pbc(domdec.get_homeix()-ix,
						 domdec.get_homeiy()-iy,
						 domdec.get_homeiz()-iz);
      i++;
    }
  }

  mynode = domdec.get_mynode();
}

//
// Class destructor
//
CudaDomdecConstComm::~CudaDomdecConstComm() {
  if (recvCoordInd != NULL) deallocate<int>(&recvCoordInd);
  if (sendCoordInd != NULL) deallocate<int>(&sendCoordInd);
  if (recvCoordBuf != NULL) deallocate<double>(&recvCoordBuf);
  if (sendCoordBuf != NULL) deallocate<double>(&sendCoordBuf);
  if (h_sendBuf != NULL) deallocate_host<char>(&h_sendBuf);
  if (h_recvBuf != NULL) deallocate_host<char>(&h_recvBuf);
}

//
// Setup communication after neighborlist update
//
void CudaDomdecConstComm::setup(const int* neighPos, int* coordInd, const int* glo2loc,
				cudaStream_t stream) {
  if (numNeigh == 0) return;

  // Send/Recv number of coordinates
  int nrequest = 0;
  for (int i=0;i < numNeigh;i++) {
    int ineigh = neighInd.at(i);
    numRecv.at(i) = neighPos[ineigh+1] - neighPos[ineigh];
    const int NUM_TAG = 1;
    MPICheck(MPI_Isend(&numRecv.at(i), 1, MPI_INT, rightMPInode.at(i), NUM_TAG,
		       cudaMPI.get_comm(), &request.at(nrequest++)));
    MPICheck(MPI_Irecv(&numSend.at(i), 1, MPI_INT, leftMPInode.at(i), NUM_TAG,
		       cudaMPI.get_comm(), &request.at(nrequest++)));
  }

  // Wait for communication to finish
  MPICheck(MPI_Waitall(nrequest, request.data(), MPI_STATUSES_IGNORE));

  posSend.at(0) = 0;
  posRecv.at(0) = 0;
  for (int i=1;i < numNeigh;i++) {
    posSend.at(i) = posSend.at(i-1) + numSend.at(i-1);
    posRecv.at(i) = posRecv.at(i-1) + numRecv.at(i-1);
  }
  numSendTot = posSend.at(numNeigh-1) + numSend.at(numNeigh-1);
  numRecvTot = posRecv.at(numNeigh-1) + numRecv.at(numNeigh-1);

  reallocate<int>(&recvCoordInd, &recvCoordIndLen, numRecvTot, 1.2f);
  reallocate<int>(&sendCoordInd, &sendCoordIndLen, numSendTot, 1.2f);
  reallocate<double>(&recvCoordBuf, &recvCoordBufLen, 3*numRecvTot, 1.2f);
  reallocate<double>(&sendCoordBuf, &sendCoordBufLen, 3*numSendTot, 1.2f);

  // Make temporary integer array from sendCoordBuf
  int* sendCoordTmp = (int *)sendCoordBuf;

  // Build recvCoordInd from coordInd (by mapping it to local indices)
  int nthread = 512;
  int nblock = (numRecvTot-1)/nthread + 1;
  buildCoordInd_kernel<<< nblock, nthread, 0, stream >>>
    (numRecvTot, coordInd, glo2loc, recvCoordInd);
  cudaCheck(cudaGetLastError());

  // Re-allocate h_sendBuf and h_recvBuf if needed
  if (!cudaMPI.isCudaAware()) {
    reallocate_host<char>(&h_sendBuf, &h_sendBufLen, 3*numSendTot*sizeof(double), 1.2f);
    reallocate_host<char>(&h_recvBuf, &h_recvBufLen, 3*numRecvTot*sizeof(double), 1.2f);
  }

  // Send/Recv coordinate indices
  for (int i=0;i < numNeigh;i++) {
    int* h_sendBufInt = (int *)h_sendBuf;
    int* h_recvBufInt = (int *)h_recvBuf;
    const int DATA_TAG = 2;
    if (numRecv.at(i) > 0 && numSend.at(i) > 0) {
      MPICheck(cudaMPI.Sendrecv(coordInd+posRecv.at(i), numRecv.at(i)*sizeof(int),
				rightMPInode.at(i), DATA_TAG,
				sendCoordTmp+posSend.at(i), numSend.at(i)*sizeof(int),
				leftMPInode.at(i), DATA_TAG, MPI_STATUS_IGNORE,
				h_recvBufInt+posRecv.at(i), h_sendBufInt+posSend.at(i)));
    } else if (numRecv.at(i) > 0) {
      MPICheck(cudaMPI.Send(coordInd+posRecv.at(i), numRecv.at(i)*sizeof(int),
			    rightMPInode.at(i), DATA_TAG,
			    h_recvBufInt+posRecv.at(i)));
    } else if (numSend.at(i) > 0) {
      MPICheck(cudaMPI.Recv(sendCoordTmp+posSend.at(i), numSend.at(i)*sizeof(int),
			    leftMPInode.at(i), DATA_TAG, MPI_STATUS_IGNORE,
			    h_sendBufInt+posSend.at(i)));
    }
  }

  // Build sendCoordInd (by mapping it to local indices)
  nthread = 512;
  nblock = (numRecvTot-1)/nthread + 1;
  buildCoordInd_kernel<<< nblock, nthread, 0, stream >>>
    (numSendTot, sendCoordTmp, glo2loc, sendCoordInd);
  cudaCheck(cudaGetLastError());

  /*
  cudaCheck(cudaStreamSynchronize(stream));
  int* h_sendCoordTmp = new int[numSendTot];
  copy_DtoH_sync<int>(sendCoordTmp, h_sendCoordTmp, numSendTot);
  fprintf(stderr,"%d sendCoord(Global):",mynode);
  for (int i=0;i < numSendTot;i++) {
    fprintf(stderr," %d",h_sendCoordTmp[i]);
  }
  fprintf(stderr,"\n");
  delete [] h_sendCoordTmp;

  cudaCheck(cudaStreamSynchronize(stream));
  int* h_coordInd = new int[numRecvTot];
  copy_DtoH_sync<int>(coordInd, h_coordInd, numRecvTot);
  fprintf(stderr,"%d recvCoord(Global):",mynode);
  for (int i=0;i < numRecvTot;i++) {
    fprintf(stderr," %d",h_coordInd[i]);
  }
  fprintf(stderr,"\n");
  delete [] h_coordInd;

  cudaCheck(cudaStreamSynchronize(stream));
  int* h_sendCoordInd = new int[numSendTot];
  copy_DtoH_sync<int>(sendCoordInd, h_sendCoordInd, numSendTot);
  fprintf(stderr,"%d sendCoord(Local):",mynode);
  for (int i=0;i < numSendTot;i++) {
    fprintf(stderr," %d",h_sendCoordInd[i]);
  }
  fprintf(stderr,"\n");
  delete [] h_sendCoordInd;

  cudaCheck(cudaStreamSynchronize(stream));
  int* h_recvCoordInd = new int[numRecvTot];
  copy_DtoH_sync<int>(recvCoordInd, h_recvCoordInd, numRecvTot);
  fprintf(stderr,"%d recvCoord(Local):",mynode);
  for (int i=0;i < numRecvTot;i++) {
    fprintf(stderr," %d",h_recvCoordInd[i]);
  }
  fprintf(stderr,"\n");
  delete [] h_recvCoordInd;
  */

}

//
// Packs (x,y,z) => coordBuf
//
void CudaDomdecConstComm::packCoordBuf(const int n, const int* coordInd,
				       cudaXYZ<double>& coord, double* coordBuf,
				       cudaStream_t stream) {
  int nthread = 512;
  int nblock = (n-1)/nthread+1;
  packCoordBuf_kernel<<< nblock, nthread, 0, stream >>>
    (n, coordInd, coord.x(), coord.y(), coord.z(), coordBuf);
  cudaCheck(cudaGetLastError());
}

//
// Unpacks coordBuf => (x, y, z)
//
void CudaDomdecConstComm::unpackCoordBuf(const int n, const int* coordInd,
					 const double* coordBuf, cudaXYZ<double>& coord,
					 cudaStream_t stream) {
  // Unpack coordinates from coordBuf to (x, y, z)
  int nthread = 512;
  int nblock = (n-1)/nthread+1;
  unpackCoordBuf_kernel<<< nblock, nthread, 0, stream >>>
  (n, coordInd, coordBuf, coord.x(), coord.y(), coord.z());
  cudaCheck(cudaGetLastError());
}

//
// Communicate coordinate buffers
//
void CudaDomdecConstComm::commCoordBuf(std::vector<int>& sendMPInodeLoc, std::vector<int>& recvMPInodeLoc,
				       std::vector<int>& numSendLoc, std::vector<int>& posSendLoc,
				       std::vector<int>& numRecvLoc, std::vector<int>& posRecvLoc,
				       double* sendCoordBufLoc, double* recvCoordBufLoc,
				       double* h_sendBufLoc, double* h_recvBufLoc) {
  for (int i=0;i < numNeigh;i++) {
    const int DATA_TAG = 2;
    if (numRecvLoc.at(i) > 0 && numSendLoc.at(i) > 0) {
      MPICheck(cudaMPI.Sendrecv(sendCoordBufLoc+posSendLoc.at(i), 3*numSendLoc.at(i)*sizeof(double),
				sendMPInodeLoc.at(i), DATA_TAG,
				recvCoordBufLoc+posRecvLoc.at(i), 3*numRecvLoc.at(i)*sizeof(double),
				recvMPInodeLoc.at(i), DATA_TAG, MPI_STATUS_IGNORE,
				h_sendBufLoc+posSendLoc.at(i), h_recvBufLoc+posRecvLoc.at(i)));
    } else if (numSendLoc.at(i) > 0) {
      MPICheck(cudaMPI.Send(sendCoordBufLoc+posSendLoc.at(i), 3*numSendLoc.at(i)*sizeof(double),
			    sendMPInodeLoc.at(i), DATA_TAG,
			    h_sendBufLoc+posSendLoc.at(i)));
    } else if (numRecvLoc.at(i) > 0) {
      MPICheck(cudaMPI.Recv(recvCoordBufLoc+posRecvLoc.at(i), 3*numRecvLoc.at(i)*sizeof(double),
			    recvMPInodeLoc.at(i), DATA_TAG, MPI_STATUS_IGNORE,
			    h_recvBufLoc+posRecvLoc.at(i)));
    }
  }
}

//
// Sends coordinates to "Left" or "-1" -neighbor
//                  OR
// Sends coordinates to the "Right" or "+1" -neighbor
//
void CudaDomdecConstComm::communicate(const int dir, cudaXYZ<double>& coord, cudaStream_t stream) {
  assert(dir == -1 || dir == 1);

  if (numNeigh == 0) return;

  if (dir == -1) {
    // Pack the coordinates form (x, y, z) to sendCoordBuf
    packCoordBuf(numSendTot, sendCoordInd, coord, sendCoordBuf, stream);
  } else {
    // Pack the coordinates form (x, y, z) to recvCoordBuf
    packCoordBuf(numRecvTot, recvCoordInd, coord, recvCoordBuf, stream);
  }

  cudaCheck(cudaStreamSynchronize(stream));
  if (dir == -1) {
    commCoordBuf(leftMPInode, rightMPInode, numSend, posSend, numRecv, posRecv,
		 sendCoordBuf, recvCoordBuf, (double *)h_sendBuf, (double *)h_recvBuf);
  } else {
    commCoordBuf(rightMPInode, leftMPInode, numRecv, posRecv, numSend, posSend,
		 recvCoordBuf, sendCoordBuf, (double *)h_recvBuf, (double *)h_sendBuf);
  }

  if (dir == -1) {
    // Unpack recvCoordBuf to (x, y, z)
    unpackCoordBuf(numRecvTot, recvCoordInd, recvCoordBuf, coord, stream);
  } else {
    // Unpack sendCoordBuf to (x, y, z)
    unpackCoordBuf(numSendTot, sendCoordInd, sendCoordBuf, coord, stream);
  }

  /*
  cudaCheck(cudaStreamSynchronize(stream));
  int* h_sendCoordInd = new int[numSendTot];
  copy_DtoH_sync<int>(sendCoordInd, h_sendCoordInd, numSendTot);
  if (mynode == 1) {
    fprintf(stderr,"%d sendCoordInd:",mynode);
    for (int i=0;i < numSendTot;i++) {
      fprintf(stderr," %d",h_sendCoordInd[i]);
    }
    fprintf(stderr,"\n");
  }

  cudaCheck(cudaStreamSynchronize(stream));
  int* h_recvCoordInd = new int[numRecvTot];
  copy_DtoH_sync<int>(recvCoordInd, h_recvCoordInd, numRecvTot);
  if (mynode == 0) {
    fprintf(stderr,"%d recvCoordInd:",mynode);
    for (int i=0;i < numRecvTot;i++) {
      fprintf(stderr," %d",h_recvCoordInd[i]);
    }
    fprintf(stderr,"\n");
  }

  cudaCheck(cudaDeviceSynchronize());
  double h_x;
  double h_y;
  double h_z;
  int ind = (mynode == 1) ? h_sendCoordInd[0] : h_recvCoordInd[0];
  copy_DtoH_sync<double>(coord.x()+ind, &h_x, 1);
  copy_DtoH_sync<double>(coord.y()+ind, &h_y, 1);
  copy_DtoH_sync<double>(coord.z()+ind, &h_z, 1);
  fprintf(stderr,"%d xyz: %lf %lf %lf\n",mynode, h_x, h_y, h_z);

  delete [] h_sendCoordInd;
  delete [] h_recvCoordInd;
  */

}
