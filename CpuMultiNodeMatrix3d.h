#ifndef CPUMULTINODEMATRIX3D_H
#define CPUMULTINODEMATRIX3D_H

#include <mpi.h>
#include "CpuMatrix3d.h"

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

  T *data;

  node_t() {
    len = 0;
    data = NULL;
  }

  ~node_t() {
    if (data != NULL) delete [] data;
  }

};

template <typename T>
class CpuMultiNodeMatrix3d : public CpuMatrix3d<T> {

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
  CpuMultiNodeMatrix3d<T>* mat_yzx;
  int nsend;
  node_t<T> *send;

  int nrecv;
  node_t<T> *recv;

  bool loc_transpose;
  int loc_x0, loc_x1;
  int loc_y0, loc_y1;
  int loc_z0, loc_z1;

  MPI_Request *recv_req;
  MPI_Request *send_req;

  MPI_Status *recv_stat;
  MPI_Status *send_stat;

public:

  CpuMultiNodeMatrix3d(const int nxtot, const int nytot, const int nztot,
		    const int nnodex, const int nnodey, const int nnodez,
		    const int mynode,
		    const char *filename = NULL);
  ~CpuMultiNodeMatrix3d();

  bool compare(CpuMatrix3d<T>* mat, const double tol, double& max_diff);

  void print_info();
  void setup_transpose_xyz_yzx(CpuMultiNodeMatrix3d<T>* mat);
  void transpose_xyz_yzx();
  void transpose_xyz_yzx(CpuMultiNodeMatrix3d<T>* mat);

};

#endif // CPUMULTINODEMATRIX3D_H
