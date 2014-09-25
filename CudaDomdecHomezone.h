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

  // num_neighind[nneigh] = number of atoms that are going to each box
  int *num_neighind;

  // Exclusive cumulative sum of num_neighind
  int *pos_neighind;

  // destind[ncoord] = destination node index for each coordinate
  int destind_len;
  int *destind;

  // Neighbor communication buffers
  int neighsend_len;
  neighcomm_t *neighsend;

  int neighrecv_len;
  neighcomm_t *neighrecv;

  // ------------
  // Host memory
  // ------------
  // Neighbor node list (i.e. these are the MPI node numbers)
  // neighnode[nneigh]
  int *neighnode;

  // Host version of pos_neighind
  int *h_pos_neighind;

  // Host version, only used when no cuda-aware MPI is available
  int h_neighsend_len;
  neighcomm_t *h_neighsend;

  int h_neighrecv_len;
  neighcomm_t *h_neighrecv;

  // MPI requests
  MPI_Request *send_request;

 public:
  CudaDomdecHomezone(Domdec& domdec, CudaMPI& cudaMPI);
  ~CudaDomdecHomezone();

  int build(hostXYZ<double>& coord);
  int update(cudaXYZ<double>& coord, cudaXYZ<double>& coord2, cudaStream_t stream=0);

  int* get_loc2glo_ptr() {return thrust::raw_pointer_cast(loc2glo.data());}
  thrust::device_vector<int>& get_loc2glo() {return loc2glo;}
};

#endif // CUDADOMDECHOMEZONE_H
