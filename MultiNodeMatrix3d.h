#ifndef MULTINODEMATRIX3D_H
#define MULTINODEMATRIX3D_H

#include "Matrix3d.h"

template <typename T>
class node_t {

public:

  int node;
  int x0, x1;
  int y0, y1;
  int z0, z1;
  T *h_data;

  node_t() {
    h_data = NULL;
  }

  ~node_t() {
    if (h_data != NULL) {
      delete [] h_data;
    }
  }

};

template <typename T>
class MultiNodeMatrix3d : Matrix3d<T> {

private:

  // Number of nodes
  int nnode;

  // Number of nodes in each coordinate direction
  // NOTE: nnode = nnodex * nnodey * nnodez
  int nnodex;
  int nnodey;
  int nnodez;

  // Node (MPI) ID list
  int *nodeID;

  // My node index
  int mynode;

  // Coordinate limits for each node
  int *x0, *x1;
  int *y0, *y1;
  int *z0, *z1;

  int nsend;
  node_t<T> *send;

  int nrecv;
  node_t<T> *recv;

  int nxtot;
  int nytot;
  int nztot;

  int xsizetot;
  int ysizetot;
  int zsizetot;

public:

  MultiNodeMatrix3d(const int nxtot, const int nytot, const int nztot,
		    const int nnodex, const int nnodey, const int nnodez,
		    const int mynode);
  ~MultiNodeMatrix3d();

  void print_info();
  void transpose_xyz_yzx(MultiNodeMatrix3d<T>* mat);

};

#endif // MULTINODEMATRIX3D_H
