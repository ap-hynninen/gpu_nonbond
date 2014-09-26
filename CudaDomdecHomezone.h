#ifndef CUDADOMDECHOMEZONE_H
#define CUDADOMDECHOMEZONE_H

#include <thrust/device_vector.h>
#include "cudaXYZ.h"
#include "hostXYZ.h"
#include "Domdec.h"
#include "CudaMPI.h"

class CudaDomdecHomezone {
 public:
  //
  // Neighbor communication data structure
  //
  struct neighcomm_t {
    int gloind;
    double x1, y1, z1;
    double x2, y2, z2;
  };

 private:

  // Domdec definition
  Domdec& domdec;

  // Cuda MPI
  CudaMPI& cudaMPI;

  // Number of neighbors, including the current node
  int nneigh;

  // ------------
  // Device memory
  // ------------

  // Local -> global mapping
  // NOTE: also serves as a list of atom on this node
  thrust::device_vector<int> loc2glo;

  // num_send[nneigh] = number of atoms that are going to send each box
  int *num_send;

  // Exclusive cumulative sum of num_send
  int *pos_send;

  // destind[ncoord] = destination node index for each coordinate
  int destind_len;
  int *destind;

  // Neighbor communication buffers
  int send_len;
  neighcomm_t *send;

  int recv_len;
  neighcomm_t *recv;

  // ------------
  // Host memory
  // ------------
  // Neighbor node list (i.e. these are the MPI node numbers)
  // neighnode[nneigh]
  std::vector<int> neighnode;

  // Index of mynode in neighnode (i.e. neighnode[imynode] = domdec.get_mynode())
  int imynode;

  // Host version of num_send and pos_send
  int *h_num_send;
  int *h_pos_send;

  // Host version, only used when no cuda-aware MPI is available
  int h_send_len;
  neighcomm_t *h_send;

  int h_recv_len;
  neighcomm_t *h_recv;

  // Number of coordinates we receive from each neighbor node
  std::vector<int> num_recv;
  std::vector<int> pos_recv;

  // MPI requests
  std::vector<MPI_Request> request;

 public:
  CudaDomdecHomezone(Domdec& domdec, CudaMPI& cudaMPI);
  ~CudaDomdecHomezone();

  int build(hostXYZ<double>& h_coord);
  int update(cudaXYZ<double>& coord, cudaXYZ<double>& coord2, cudaStream_t stream=0);

  int* get_loc2glo_ptr() {return thrust::raw_pointer_cast(loc2glo.data());}
  thrust::device_vector<int>& get_loc2glo() {return loc2glo;}
};

#endif // CUDADOMDECHOMEZONE_H
