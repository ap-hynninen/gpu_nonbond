#include <mpi.h>
#include <iostream>
#include <cassert>
#include <stdlib.h>
#include "mpi_utils.h"
#include "cpu_utils.h"
//#define use_onesided
#include "CpuMultiNodeMatrix3d.h"

#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))

void get_inode_xyz(const int nnodex, const int nnodey, const int nnodez,
		  const int mynode, int &inodex, int &inodey, int &inodez) {
  int mynode_tmp = mynode;
  inodez = mynode_tmp/(nnodex*nnodey);
  mynode_tmp -= inodez*nnodex*nnodey;
  inodey = mynode_tmp/nnodex;
  mynode_tmp -= inodey*nnodex;
  inodex = mynode_tmp;
}

int get_xyz0(const int ntot, const int nnode, const int inode) {
  return ntot*inode/nnode;
}

int get_xyz1(const int ntot, const int nnode, const int inode) {
  return ntot*(inode+1)/nnode - 1;
}

int get_nx(const int nxtot, const int nnodex, const int nnodey, const int nnodez, const int mynode) {
  int inodex, inodey, inodez;
  get_inode_xyz(nnodex, nnodey, nnodez, mynode, inodex, inodey, inodez);
  return get_xyz1(nxtot, nnodex, inodex) - get_xyz0(nxtot, nnodex, inodex) + 1;
}

int get_ny(const int nytot, const int nnodex, const int nnodey, const int nnodez, const int mynode) {
  int inodex, inodey, inodez;
  get_inode_xyz(nnodex, nnodey, nnodez, mynode, inodex, inodey, inodez);
  return get_xyz1(nytot, nnodey, inodey) - get_xyz0(nytot, nnodey, inodey) + 1;
}

int get_nz(const int nztot, const int nnodex, const int nnodey, const int nnodez, const int mynode) {
  int inodex, inodey, inodez;
  get_inode_xyz(nnodex, nnodey, nnodez, mynode, inodex, inodey, inodez);
  return get_xyz1(nztot, nnodez, inodez) - get_xyz0(nztot, nnodez, inodez) + 1;
}

template <typename T>
CpuMultiNodeMatrix3d<T>::CpuMultiNodeMatrix3d(const int nxtot, const int nytot, const int nztot,
					      const int nnodex, const int nnodey, const int nnodez,
					      const int mynode, const int tiledim,
					      const char *filename) : 
  nxtot(nxtot), nytot(nytot), nztot(nztot), 
  xsizetot(nxtot), ysizetot(nytot), zsizetot(nztot),
  nnodex(nnodex), nnodey(nnodey), nnodez(nnodez), mynode(mynode), nnode(nnodex*nnodey*nnodez),
  CpuMatrix3d<T>(get_nx(nxtot, nnodex, nnodey, nnodez, mynode), 
		 get_ny(nytot, nnodex, nnodey, nnodez, mynode),
		 get_nz(nztot, nnodex, nnodey, nnodez, mynode), tiledim) {

  x0 = new int[nnode];
  x1 = new int[nnode];
  y0 = new int[nnode];
  y1 = new int[nnode];
  z0 = new int[nnode];
  z1 = new int[nnode];

  nodeID = new int[nnode];

  for (int order=0;order < 2;order++) {
    mat_t[order] = NULL;

    send[order] = NULL;
    recv[order] = NULL;

#ifdef use_onesided
    win_set[order] = false;
#else
    send_req[order] = NULL;
    recv_req[order] = NULL;
#endif
  }

  for (int inode=0;inode < nnode;inode++) {
    int inodex, inodey, inodez;
    get_inode_xyz(nnodex, nnodey, nnodez, inode, inodex, inodey, inodez);
    x0[inode] = get_xyz0(nxtot, nnodex, inodex);
    x1[inode] = get_xyz1(nxtot, nnodex, inodex);
    y0[inode] = get_xyz0(nytot, nnodey, inodey);
    y1[inode] = get_xyz1(nytot, nnodey, inodey);
    z0[inode] = get_xyz0(nztot, nnodez, inodez);
    z1[inode] = get_xyz1(nztot, nnodez, inodez);
    nodeID[inode] = inode;
  }

  if (filename != NULL) {
    this->load(x0[mynode], x1[mynode], nxtot,
	       y0[mynode], y1[mynode], nytot,
	       z0[mynode], z1[mynode], nztot,
	       filename);
  }

}

//
// Class destructor
//
template <typename T>
CpuMultiNodeMatrix3d<T>::~CpuMultiNodeMatrix3d() {
  delete [] x0;
  delete [] x1;
  delete [] y0;
  delete [] y1;
  delete [] z0;
  delete [] z1;
  delete [] nodeID;
  for (int order=0;order < 2;order++) {
    this->deallocate_transpose(order);
  }
}

//
// De-allocates memory buffers used for tranpose
//
template <typename T>
void CpuMultiNodeMatrix3d<T>::deallocate_transpose(const int order) {
  if (send[order] != NULL) delete [] send[order];
  if (recv[order] != NULL) delete [] recv[order];
#ifdef use_onesided
  if (win_set) MPICheck(MPI_Win_free(&win[order]));
#else
  if (send_req[order] != NULL) delete [] send_req[order];
  if (recv_req[order] != NULL) delete [] recv_req[order];
#endif
}

//
// Prints matrix size on screen
//
template <typename T>
void CpuMultiNodeMatrix3d<T>::print_info() {
  std::cout << "mynode = " << mynode << std::endl;
  std::cout << "nxtot nytot nztot          = " << nxtot << " "<< nytot << " "<< nztot << std::endl;
  std::cout << "xsizetot ysizetot zsizetot = " << xsizetot << " "<< ysizetot << " "<<zsizetot<<std::endl;
  std::cout << "x0...x1 = " << x0[mynode] << " ... " << x1[mynode] << std::endl;
  std::cout << "y0...y1 = " << y0[mynode] << " ... " << y1[mynode] << std::endl;
  std::cout << "z0...z1 = " << z0[mynode] << " ... " << z1[mynode] << std::endl;
  CpuMatrix3d<T>::print_info();
}

//
// Compares two matrices, returns true if the difference is within tolerance
// NOTE: Comparison is done in double precision
//
template <typename T>
bool CpuMultiNodeMatrix3d<T>::compare(CpuMatrix3d<T>& mat, const double tol, double& max_diff) {
  assert(mat.get_nx() == nxtot);
  assert(mat.get_ny() == nytot);
  assert(mat.get_nz() == nztot);

  int res = (int)false;
  int loc_res = (int)false;

  CpuMatrix3d<T> loc(x1[mynode]-x0[mynode]+1,
		     y1[mynode]-y0[mynode]+1,
		     z1[mynode]-z0[mynode]+1);
  
  mat.copy(x0[mynode], y0[mynode], z0[mynode],
	   0, 0, 0,
	   x1[mynode]-x0[mynode]+1, y1[mynode]-y0[mynode]+1, z1[mynode]-z0[mynode]+1,
	   loc);

  loc_res = (int)CpuMatrix3d<T>::compare(loc, tol, max_diff);

  MPICheck(MPI_Allreduce(&loc_res, &res, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD));

  return (bool)res;
}

//
// Set data element in the node where it belongs
//
template <typename T>
void CpuMultiNodeMatrix3d<T>::setData(const int x, const int y, const int z, const T val) {
  assert(x >= 0 && x < nxtot);
  assert(y >= 0 && y < nytot);
  assert(z >= 0 && z < nztot);

  if (x >= x0[mynode] && x <= x1[mynode] &&
      y >= y0[mynode] && y <= y1[mynode] &&
      z >= z0[mynode] && z <= z1[mynode]) {
    this->data[x-x0[mynode] + (y-y0[mynode] + (z-z0[mynode])*this->ysize)*this->xsize] = val;
  }
}

//
// Returns data element from the correct node.
// Returns undetermined if element not found
//
template <typename T>
T CpuMultiNodeMatrix3d<T>::getData(const int x, const int y, const int z) {
  assert(x >= 0 && x < nxtot);
  assert(y >= 0 && y < nytot);
  assert(z >= 0 && z < nztot);

  if (x >= x0[mynode] && x <= x1[mynode] &&
      y >= y0[mynode] && y <= y1[mynode] &&
      z >= z0[mynode] && z <= z1[mynode]) {
    return this->data[x-x0[mynode] + (y-y0[mynode] + (z-z0[mynode])*this->ysize)*this->xsize];
  }

  return this->data[0];
}

//
// Returns true if this node had the element, false otherwise
//
template <typename T>
bool CpuMultiNodeMatrix3d<T>::hasData(const int x, const int y, const int z) {
  assert(x >= 0 && x < nxtot);
  assert(y >= 0 && y < nytot);
  assert(z >= 0 && z < nztot);

  if (x >= x0[mynode] && x <= x1[mynode] &&
      y >= y0[mynode] && y <= y1[mynode] &&
      z >= z0[mynode] && z <= z1[mynode]) {
    return true;
  }

  return false;
}

//
// Returns the overlap between two 3d volumes
// Returns true if overlap, false otherwise
//
bool get_vol_overlap(int x0_a, int x1_a, int y0_a, int y1_a, int z0_a, int z1_a,
		     int x0_b, int x1_b, int y0_b, int y1_b, int z0_b, int z1_b,
		     int &x0_ab, int &x1_ab, int &y0_ab, int &y1_ab, int &z0_ab, int &z1_ab) {
  
  x0_ab = max(x0_a, x0_b);
  x1_ab = min(x1_a, x1_b);
  y0_ab = max(y0_a, y0_b);
  y1_ab = min(y1_a, y1_b);
  z0_ab = max(z0_a, z0_b);
  z1_ab = min(z1_a, z1_b);

  return ((x0_ab < x1_ab) && (y0_ab < y1_ab) && (z0_ab < z1_ab));
}

//
// Setups a transpose for multi-node 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
//
template <typename T>
void CpuMultiNodeMatrix3d<T>::setup_transpose_yzx(CpuMultiNodeMatrix3d<T>& mat) {
  setup_transpose(mat, YZX);
}

//
// Setups a transpose for multi-node 3d matrix out-of-place: data(x, y, z) -> data(z, x, y)
//
template <typename T>
void CpuMultiNodeMatrix3d<T>::setup_transpose_zxy(CpuMultiNodeMatrix3d<T>& mat) {
  setup_transpose(mat, ZXY);
}

//
// Setups a transpose for multi-node 3d matrix out-of-place
//
template <typename T>
void CpuMultiNodeMatrix3d<T>::setup_transpose(CpuMultiNodeMatrix3d<T>& mat, const int order) {
  assert(order == YZX || order == ZXY);

  if (mat_t[order] != NULL) this->deallocate_transpose(order);

  mat_t[order] = &mat;

  assert(mat.nnode == nnode);
  if (order == YZX) {
    assert(mat.nxtot == nytot);
    assert(mat.nytot == nztot);
    assert(mat.nztot == nxtot);
    assert(mat.xsizetot == ysizetot);
    assert(mat.ysizetot == zsizetot);
    assert(mat.zsizetot == xsizetot);
  } else {
    assert(mat.nxtot == nztot);
    assert(mat.nytot == nxtot);
    assert(mat.nztot == nytot);
    assert(mat.xsizetot == zsizetot);
    assert(mat.ysizetot == xsizetot);
    assert(mat.zsizetot == ysizetot);
  }

  // Limits on the transposed matrix of node "mynode"
  int x0_t, y0_t, z0_t;
  int x1_t, y1_t, z1_t;
  if (order == YZX) {
    x0_t = mat.z0[mynode];
    x1_t = mat.z1[mynode];
    y0_t = mat.x0[mynode];
    y1_t = mat.x1[mynode];
    z0_t = mat.y0[mynode];
    z1_t = mat.y1[mynode];
  } else {
    x0_t = mat.y0[mynode];
    x1_t = mat.y1[mynode];
    y0_t = mat.z0[mynode];
    y1_t = mat.z1[mynode];
    z0_t = mat.x0[mynode];
    z1_t = mat.x1[mynode];
  }

  assert(nnodex == mat.nnodex);
  assert(nnodey == mat.nnodey);
  assert(nnodez == mat.nnodez);

  loc_transpose[order] = false;

  send[order] = new node_t<T>[nnode];
  recv[order] = new node_t<T>[nnode];

  nrecv[order] = 0;
  nsend[order] = 0;
  for (int inode=0;inode < nnode;inode++) {

    //--------------------------------
    // Data receiving
    //--------------------------------
    // Calculate overlap between:
    // node "mynode" on the transposed matrix and node "inode" on the original matrix

    int x0_ol, x1_ol;
    int y0_ol, y1_ol;
    int z0_ol, z1_ol;

    bool overlap;
    overlap = get_vol_overlap(x0_t, x1_t, y0_t, y1_t, z0_t, z1_t,
			      x0[inode], x1[inode], y0[inode], y1[inode], z0[inode], z1[inode],
			      x0_ol, x1_ol, y0_ol, y1_ol, z0_ol, z1_ol);

    if (inode != mynode && overlap) {
      recv[order][nrecv[order]].x0 = x0_ol;
      recv[order][nrecv[order]].x1 = x1_ol;
      recv[order][nrecv[order]].y0 = y0_ol;
      recv[order][nrecv[order]].y1 = y1_ol;
      recv[order][nrecv[order]].z0 = z0_ol;
      recv[order][nrecv[order]].z1 = z1_ol;
      recv[order][nrecv[order]].xsize = recv[order][nrecv[order]].x1-recv[order][nrecv[order]].x0+1;
      recv[order][nrecv[order]].ysize = recv[order][nrecv[order]].y1-recv[order][nrecv[order]].y0+1;
      recv[order][nrecv[order]].len = (recv[order][nrecv[order]].x1-recv[order][nrecv[order]].x0+1)*
	(recv[order][nrecv[order]].y1-recv[order][nrecv[order]].y0+1)*
	(recv[order][nrecv[order]].z1-recv[order][nrecv[order]].z0+1);
      recv[order][nrecv[order]].allocate_data(recv[order][nrecv[order]].len);
      recv[order][nrecv[order]].node = inode;
      nrecv[order]++;
    }

    //--------------------------------
    // Data sending & Local transpose
    //--------------------------------
    // Calculate overlap between:
    // node "mynode" on the original matrix and node "inode" on the transposed matrix
    int x0i_t, y0i_t, z0i_t;
    int x1i_t, y1i_t, z1i_t;
    if (order == YZX) {
      x0i_t = mat.z0[inode];
      x1i_t = mat.z1[inode];
      y0i_t = mat.x0[inode];
      y1i_t = mat.x1[inode];
      z0i_t = mat.y0[inode];
      z1i_t = mat.y1[inode];
    } else {
      x0i_t = mat.y0[inode];
      x1i_t = mat.y1[inode];
      y0i_t = mat.z0[inode];
      y1i_t = mat.z1[inode];
      z0i_t = mat.x0[inode];
      z1i_t = mat.x1[inode];
    }

    overlap = get_vol_overlap(x0i_t, x1i_t, y0i_t, y1i_t, z0i_t, z1i_t,
			      x0[mynode], x1[mynode], y0[mynode], y1[mynode], z0[mynode], z1[mynode],
			      x0_ol, x1_ol, y0_ol, y1_ol, z0_ol, z1_ol);

    /*
    if (overlap && mynode == 1)
      fprintf(stderr,"\n%d %d %d %d %d %d\n%d %d %d %d %d %d\n%d %d %d %d %d %d\n",
	      x0i_t, x1i_t, y0i_t, y1i_t, z0i_t, z1i_t,
	      x0[mynode], x1[mynode], y0[mynode], y1[mynode], z0[mynode], z1[mynode],
	      x0_ol, x1_ol, y0_ol, y1_ol, z0_ol, z1_ol);
    */

    if (overlap) {
      send[order][nsend[order]].x0 = x0_ol;
      send[order][nsend[order]].x1 = x1_ol;
      send[order][nsend[order]].y0 = y0_ol;
      send[order][nsend[order]].y1 = y1_ol;
      send[order][nsend[order]].z0 = z0_ol;
      send[order][nsend[order]].z1 = z1_ol;
      if (inode == mynode) {
	// Local transpose
	loc_x0[order] = send[order][nsend[order]].x0;
	loc_x1[order] = send[order][nsend[order]].x1;
	loc_y0[order] = send[order][nsend[order]].y0;
	loc_y1[order] = send[order][nsend[order]].y1;
	loc_z0[order] = send[order][nsend[order]].z0;
	loc_z1[order] = send[order][nsend[order]].z1;
	loc_transpose[order] = true;
      } else {
	send[order][nsend[order]].xsize = send[order][nsend[order]].x1-send[order][nsend[order]].x0+1;
	send[order][nsend[order]].ysize = send[order][nsend[order]].y1-send[order][nsend[order]].y0+1;
	send[order][nsend[order]].len = (send[order][nsend[order]].x1-send[order][nsend[order]].x0+1)*
	  (send[order][nsend[order]].y1-send[order][nsend[order]].y0+1)*
	  (send[order][nsend[order]].z1-send[order][nsend[order]].z0+1);
	send[order][nsend[order]].allocate_data(send[order][nsend[order]].len);
	send[order][nsend[order]].node = inode;
	nsend[order]++;
      }
    }
  }

#ifdef use_onesided
  MPI_Win_create(recv[order][0].data, recv[order][0].len*sizeof(T), sizeof(T),
		 MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  win_set[order] = true;
#else
  if (nrecv[order] > 0) {
    recv_req[order] = new MPI_Request[nrecv[order]];
  }

  if (nsend[order] > 0) {
    send_req[order] = new MPI_Request[nsend[order]];
  }
#endif
}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
//
template <typename T>
void CpuMultiNodeMatrix3d<T>::transpose_yzx(CpuMultiNodeMatrix3d<T>& mat) {
  transpose(mat, YZX);
}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(z, x, y)
//
template <typename T>
void CpuMultiNodeMatrix3d<T>::transpose_zxy(CpuMultiNodeMatrix3d<T>& mat) {
  transpose(mat, ZXY);
}

//
// Transposes a 3d matrix out-of-place
//
template <typename T>
void CpuMultiNodeMatrix3d<T>::transpose(CpuMultiNodeMatrix3d<T>& mat, const int order) {
  assert(order == YZX || order == ZXY);
  assert(&mat == mat_t[order]);

  const int MPI_tag = 1;

#ifndef use_onesided
  // Post receives
  if (nrecv[order] > 0) {
    for (int i=0;i < nrecv[order];i++) {
      MPICheck(MPI_Irecv(recv[order][i].data, sizeof(T)*recv[order][i].len, MPI_BYTE,
			 recv[order][i].node, MPI_tag, MPI_COMM_WORLD, &recv_req[order][i]));
    }
  }
#endif

  // Copy data to send buffer
  if (nsend[order] > 0) {
    for (int i=0;i < nsend[order];i++) {
      copy3D_HtoH<T>(this->data, send[order][i].data,
		     send[order][i].x0 - x0[mynode],
		     send[order][i].y0 - y0[mynode],
		     send[order][i].z0 - z0[mynode],
		     this->xsize, this->ysize,
		     0,0,0,
		     send[order][i].x1 - send[order][i].x0 + 1,
		     send[order][i].y1 - send[order][i].y0 + 1,
		     send[order][i].z1 - send[order][i].z0 + 1,
		     send[order][i].xsize, send[order][i].ysize);
    }
  }

#ifdef use_onesided
  MPICheck(MPI_Win_fence(0, win[order]));
  if (nsend[order] > 0) {
    for (int i=0;i < nsend[order];i++) {
      MPICheck(MPI_Put(send[order][i].data, send[order][i].len, MPI_FLOAT, send[order][i].node,
		       0, send[order][i].len, MPI_FLOAT, win[order]));
    }
  }
#else
  // Post sends
  if (nsend[order] > 0) {
    for (int i=0;i < nsend[order];i++) {
      // Send via MPI
      MPICheck(MPI_Isend(send[order][i].data, sizeof(T)*send[order][i].len, MPI_BYTE,
			 send[order][i].node, MPI_tag, MPI_COMM_WORLD, &send_req[order][i]));
    }
  }
#endif

  // Perform local matrix transpose on a sub block:
  // (loc_x0...loc_x1) x (loc_y0...loc_y1) x (loc_z0...loc_z1)

  if (loc_transpose[order]) {
    int dstloc_x0, dstloc_y0, dstloc_z0;
    if (order == YZX) {
      dstloc_x0 = loc_y0[order];
      dstloc_y0 = loc_z0[order];
      dstloc_z0 = loc_x0[order];
    } else {
      dstloc_x0 = loc_z0[order];
      dstloc_y0 = loc_x0[order];
      dstloc_z0 = loc_y0[order];
    }
    /*
    fprintf(stderr,"\n loc0 = %d %d %d\n loc1 = %d %d %d\n xyz0 = %d %d %d\n xyz0 = %d %d %d\n",
	    loc_x0[order],loc_y0[order],loc_z0[order],
	    loc_x1[order],loc_y1[order],loc_z1[order],
	    this->x0[mynode],this->y0[mynode],this->z0[mynode],
	    dstloc_x0, dstloc_y0, dstloc_z0);
    MPICheck(MPI_Barrier(MPI_COMM_WORLD));
    //exit(1);
    */
    CpuMatrix3d<T>::transpose(loc_x0[order] - this->x0[mynode],        // src_x0
			      loc_y0[order] - this->y0[mynode],        // src_y0
			      loc_z0[order] - this->z0[mynode],        // src_z0
			      dstloc_x0 - mat_t[order]->x0[mynode],    // dst_x0
			      dstloc_y0 - mat_t[order]->y0[mynode],    // dst_y0
			      dstloc_z0 - mat_t[order]->z0[mynode],    // dst_z0
			      loc_x1[order]-loc_x0[order]+1,           // xlen
			      loc_y1[order]-loc_y0[order]+1,           // ylen
			      loc_z1[order]-loc_z0[order]+1,           // zlen
			      *mat_t[order], order);
  }

#ifdef use_onesided
  MPICheck(MPI_Win_fence(0, win[order]));
#else
  // Wait for sends to finish
  if (nsend[order] > 0) {
    MPICheck(MPI_Waitall(nsend[order], send_req[order], MPI_STATUSES_IGNORE));
  }

  if (nrecv[order] > 0) {
    MPICheck(MPI_Waitall(nrecv[order], recv_req[order], MPI_STATUSES_IGNORE));
  }
#endif

  // Wait for receives
  if (nrecv[order] > 0) {
    for (int i=0;i < nrecv[order];i++) {
      int k=i;
      //MPICheck(MPI_Waitany(nrecv, (MPI_Request *)recv_req, &k, (MPI_Status *)recv_stat));

      CpuMatrix3d<T> loc(recv[order][k].x1-recv[order][k].x0+1,
			 recv[order][k].y1-recv[order][k].y0+1,
			 recv[order][k].z1-recv[order][k].z0+1,
			 recv[order][k].xsize, recv[order][k].ysize,
			 recv[order][k].z1-recv[order][k].z0+1, 
			 this->tiledim, recv[order][k].data);
      int srcloc_x0, srcloc_y0, srcloc_z0;
      if (order == YZX) {
	srcloc_x0 = recv[order][k].y0;
	srcloc_y0 = recv[order][k].z0;
	srcloc_z0 = recv[order][k].x0;
      } else {
	srcloc_x0 = recv[order][k].z0;
	srcloc_y0 = recv[order][k].x0;
	srcloc_z0 = recv[order][k].y0;
      }
      loc.transpose(0,0,0,
		    srcloc_x0 - mat_t[order]->x0[mynode],
		    srcloc_y0 - mat_t[order]->y0[mynode],
		    srcloc_z0 - mat_t[order]->z0[mynode],
		    recv[order][k].x1-recv[order][k].x0+1,
		    recv[order][k].y1-recv[order][k].y0+1,
		    recv[order][k].z1-recv[order][k].z0+1,
		    *mat_t[order], order);

    }
  }

}

//
// Explicit instances of CpuMultiNodeMatrix3d
//
template class CpuMultiNodeMatrix3d<float>;
template class CpuMultiNodeMatrix3d<double>;
template class CpuMultiNodeMatrix3d<long long int>;
