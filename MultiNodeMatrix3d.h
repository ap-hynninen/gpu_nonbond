#ifndef MULTINODEMATRIX3D_H
#define MULTINODEMATRIX3D_H

#include "cuda_utils.h"
#include "Matrix3d.h"

template <typename T>
class node_t {

public:

  int node;

  int x0, x1;
  int y0, y1;
  int z0, z1;

  int xsize;
  int ysize;

  int len;

  T *h_data;
  T *d_data;

  node_t() {
    len = 0;
    h_data = NULL;
    d_data = NULL;
  }

  ~node_t() {
    if (h_data != NULL) deallocate_host<T>(&h_data); //delete [] h_data;
    if (d_data != NULL) deallocate<T>(&d_data);
  }

};

template <typename T>
class MultiNodeMatrix3d : public Matrix3d<T> {

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

  int nxtot;
  int nytot;
  int nztot;

  int xsizetot;
  int ysizetot;
  int zsizetot;

  // Stuff for matrix transpose
  MultiNodeMatrix3d<T>* mat_yzx;
  int nsend;
  node_t<T> *send;

  int nrecv;
  node_t<T> *recv;

  bool loc_transpose;
  int loc_x0, loc_x1;
  int loc_y0, loc_y1;
  int loc_z0, loc_z1;

  void *recv_req;
  void *send_req;

  void *recv_stat;
  void *send_stat;

public:

  MultiNodeMatrix3d(const int nxtot, const int nytot, const int nztot,
		    const int nnodex, const int nnodey, const int nnodez,
		    const int mynode,
		    const char *filename = NULL);
  ~MultiNodeMatrix3d();

  bool compare(Matrix3d<T>* mat, const double tol, double& max_diff);

  void print_info();
  void setup_transpose_xyz_yzx(MultiNodeMatrix3d<T>* mat);
  void transpose_xyz_yzx();
  void transpose_xyz_yzx(MultiNodeMatrix3d<T>* mat);

};

#endif // MULTINODEMATRIX3D_H
