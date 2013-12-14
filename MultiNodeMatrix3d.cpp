#ifdef USE_MPI
#include <mpi.h>
#endif
#include <iostream>
#include <cassert>
#include <stdlib.h>
#include "mpi_utils.h"
#include "cuda_utils.h"
#include "MultiNodeMatrix3d.h"
//#include <cuda.h>

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
MultiNodeMatrix3d<T>::MultiNodeMatrix3d(const int nxtot, const int nytot, const int nztot,
					const int nnodex, const int nnodey, const int nnodez,
					const int mynode) : 
  nxtot(nxtot), nytot(nytot), nztot(nztot), 
  xsizetot(nxtot), ysizetot(nytot), zsizetot(nztot),
  nnodex(nnodex), nnodey(nnodey), nnodez(nnodez), mynode(mynode), nnode(nnodex*nnodey*nnodez),
  Matrix3d<T>(get_nx(nxtot, nnodex, nnodey, nnodez, mynode), 
	      get_ny(nytot, nnodex, nnodey, nnodez, mynode),
	      get_nz(nztot, nnodex, nnodey, nnodez, mynode)) {

  //  std::cout << get_nx(nxtot, nnodex, nnodey, nnodez, mynode) << std::endl;
  //  std::cout << get_ny(nytot, nnodex, nnodey, nnodez, mynode) << std::endl;
  //  std::cout << get_nz(nztot, nnodex, nnodey, nnodez, mynode) << std::endl;

  x0 = new int[nnode];
  x1 = new int[nnode];
  y0 = new int[nnode];
  y1 = new int[nnode];
  z0 = new int[nnode];
  z1 = new int[nnode];

  nodeID = new int[nnode];

  send = new node_t<T>[nnode];
  recv = new node_t<T>[nnode];

  for (int inode=0;inode < nnode;inode++) {
    int inodex, inodey, inodez;
    get_inode_xyz(nnodex, nnodey, nnodez, inode, inodex, inodey, inodez);
    //    std::cout << inode << " " << inodex << " " << inodey << " " << inodez << std::endl;
    x0[inode] = get_xyz0(nxtot, nnodex, inodex);
    x1[inode] = get_xyz1(nxtot, nnodex, inodex);
    y0[inode] = get_xyz0(nytot, nnodey, inodey);
    y1[inode] = get_xyz1(nytot, nnodey, inodey);
    z0[inode] = get_xyz0(nztot, nnodez, inodez);
    z1[inode] = get_xyz1(nztot, nnodez, inodez);
    nodeID[inode] = inode;
  }

  for (int inode=0;inode < nnode;inode++) {
    int inodex, inodey, inodez;
    get_inode_xyz(nnodex, nnodey, nnodez, inode, inodey, inodez, inodex);
    //    std::cout << inode << " " << inodex << " " << inodey << " " << inodez << std::endl;
  }
}

template <typename T>
MultiNodeMatrix3d<T>::~MultiNodeMatrix3d() {
  delete [] x0;
  delete [] x1;
  delete [] y0;
  delete [] y1;
  delete [] z0;
  delete [] z1;
  delete [] nodeID;
  delete [] send;
  delete [] recv;
  Matrix3d<T>::~Matrix3d();
}

//
// Prints matrix size on screen
//
template <typename T>
void MultiNodeMatrix3d<T>::print_info() {
  std::cout << "mynode = " << mynode << std::endl;
  std::cout << "nxtot nytot nztot          = " << nxtot << " "<< nytot << " "<< nztot << std::endl;
  std::cout << "xsizetot ysizetot zsizetot = " << xsizetot << " "<< ysizetot << " "<< zsizetot << std::endl;  
  std::cout << "x0...x1 = " << x0[mynode] << " ... " << x1[mynode] << std::endl;
  std::cout << "y0...y1 = " << y0[mynode] << " ... " << y1[mynode] << std::endl;
  std::cout << "z0...z1 = " << z0[mynode] << " ... " << z1[mynode] << std::endl;
  Matrix3d<T>::print_info();
}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
//
template <typename T>
void MultiNodeMatrix3d<T>::transpose_xyz_yzx(MultiNodeMatrix3d<T>* mat) {

  assert(mat->nnode == nnode);
  assert(mat->nxtot == nytot);
  assert(mat->nytot == nztot);
  assert(mat->nztot == nxtot);
  assert(mat->xsizetot == ysizetot);
  assert(mat->ysizetot == zsizetot);
  assert(mat->zsizetot == xsizetot);

  // Limits in the transposed matrix
  int x0_t = z0[mynode];
  int x1_t = z1[mynode];
  int y0_t = x0[mynode];
  int y1_t = x1[mynode];
  int z0_t = y0[mynode];
  int z1_t = y1[mynode];

  //  std::cout << "x0t...x1t = " << x0t << " ... " << x1t << std::endl;
  //  std::cout << "y0t...y1t = " << y0t << " ... " << y1t << std::endl;
  //  std::cout << "z0t...z1t = " << z0t << " ... " << z1t << std::endl;

  nrecv = 0;
  nsend = 0;
  for (int inode=0;inode < nnode;inode++) {
    int x0i_t = z0[inode];
    int x1i_t = z1[inode];
    int y0i_t = x0[inode];
    int y1i_t = x1[inode];
    int z0i_t = y0[inode];
    int z1i_t = y1[inode];
    if (((x0_t >= x0[inode] && x0_t <= x1[inode]) || (x1_t >= x0[inode] && x1_t <= x1[inode])) &&
	((y0_t >= y0[inode] && y0_t <= y1[inode]) || (y1_t >= y0[inode] && y1_t <= y1[inode])) &&
	((z0_t >= z0[inode] && z0_t <= z1[inode]) || (z1_t >= z0[inode] && z1_t <= z1[inode]))) {
      // This node needs the volume (x0_t...x1_t) x (y0_t...y1_t) x (z0_t...z1_t)
      recv[nrecv].x0 = max(x0_t, x0[inode]);
      recv[nrecv].x1 = min(x1_t, x1[inode]);
      recv[nrecv].y0 = max(y0_t, y0[inode]);
      recv[nrecv].y1 = min(y1_t, y1[inode]);
      recv[nrecv].z0 = max(z0_t, z0[inode]);
      recv[nrecv].z1 = min(z1_t, z1[inode]);
      recv[nrecv].h_data = new T[(recv[nrecv].x1-recv[nrecv].x0+1)*(recv[nrecv].y1-recv[nrecv].y0+1)*
				 (recv[nrecv].z1-recv[nrecv].z0+1)];
      recv[nrecv].node = inode;
      nrecv++;
    }
    if (((x0i_t >= x0[mynode] && x0i_t <= x1[mynode]) || (x1i_t >= x0[mynode] && x1i_t <= x1[mynode])) &&
	((y0i_t >= y0[mynode] && y0i_t <= y1[mynode]) || (y1i_t >= y0[mynode] && y1i_t <= y1[mynode])) &&
	((z0i_t >= z0[mynode] && z0i_t <= z1[mynode]) || (z1i_t >= z0[mynode] && z1i_t <= z1[mynode]))) {
      send[nsend].x0 = max(x0i_t, x0[mynode]);
      send[nsend].x1 = min(x1i_t, x1[mynode]);
      send[nsend].y0 = max(y0i_t, y0[mynode]);
      send[nsend].y1 = min(y1i_t, y1[mynode]);
      send[nsend].z0 = max(z0i_t, z0[mynode]);
      send[nsend].z1 = min(z1i_t, z1[mynode]);
      send[nsend].h_data = new T[(send[nsend].x1-send[nsend].x0+1)*(send[nsend].y1-send[nsend].y0+1)*
				 (send[nsend].z1-send[nsend].z0+1)];
      send[nsend].node = inode;
      nsend++;
    }
  }  

  /*
  std::cout << "mynode " << mynode << std::endl;
  std::cout << "recv:" << std::endl;
  for (int i=0;i < nrecv;i++) {
    std::cout << recv[i].node<<std::endl;
    std::cout << "x0...x1 = " << recv[i].x0 << " ... " << recv[i].x1 << std::endl;
    std::cout << "y0...y1 = " << recv[i].y0 << " ... " << recv[i].y1 << std::endl;
    std::cout << "z0...z1 = " << recv[i].z0 << " ... " << recv[i].z1 << std::endl;
  }
  std::cout << "send:" << std::endl;
  for (int i=0;i < nsend;i++) {
    std::cout << send[i].node<<std::endl;
    std::cout << "x0...x1 = " << send[i].x0 << " ... " << send[i].x1 << std::endl;
    std::cout << "y0...y1 = " << send[i].y0 << " ... " << send[i].y1 << std::endl;
    std::cout << "z0...z1 = " << send[i].z0 << " ... " << send[i].z1 << std::endl;
  }
  */

  // Post receives
  for (int i=0;i < nrecv;i++) {
    if (recv[i].node != mynode) {
#ifdef USE_MPI
      //      MPICheck(MPI_Irecv());
#else
      std::cerr << "MultiNodeMatrix3d::transpose_xyz_yzx, MPI required for now" << std::endl;
      exit(1);
#endif
    }
  }

  // Copy data to host
#ifdef USE_MPI
  for (int i=0;i < nsend;i++) {
    if (send[i].node != mynode) {
      copy3D_DtoH<T>(this->data, send[i].h_data, send[i].x0, send[i].y0, send[i].z0,
		     this->xsize, this->ysize,
		     0, 0, 0,
		     send[i].x1 - send[i].x0, send[i].y1 - send[i].y0, send[i].z1 - send[i].z0,
		     this->xsize, this->ysize);
    }
  }
#endif

  // Post sends
  for (int i=0;i < nsend;i++) {
    if (send[i].node != mynode) {
#ifdef USE_MPI
      // Send via MPI
      //      MPICheck(MPI_Isend(h_data, ));
#else
      std::cerr << "MultiNodeMatrix3d::transpose_xyz_yzx, MPI required for now" << std::endl;
      exit(1);
#endif
    }
  }

    //  Matrix3d<T>::transpose_xyz_yzx(mat);

}

//
// Explicit instances of MultiNodeMatrix3d
//
template class MultiNodeMatrix3d<float>;
//template class MultiNodeMatrix3d<float2>;
template class MultiNodeMatrix3d<long long int>;
