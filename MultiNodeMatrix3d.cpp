#ifdef USE_MPI
#include <mpi.h>
#endif
#include <iostream>
#include <cassert>
#include <stdlib.h>
#include "mpi_utils.h"
#include "cuda_utils.h"
#include "MultiNodeMatrix3d.h"

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
					const int mynode,
					const char *filename) : 
  nxtot(nxtot), nytot(nytot), nztot(nztot), 
  xsizetot(nxtot), ysizetot(nytot), zsizetot(nztot),
  nnodex(nnodex), nnodey(nnodey), nnodez(nnodez), mynode(mynode), nnode(nnodex*nnodey*nnodez),
  Matrix3d<T>(get_nx(nxtot, nnodex, nnodey, nnodez, mynode), 
	      get_ny(nytot, nnodex, nnodey, nnodez, mynode),
	      get_nz(nztot, nnodex, nnodey, nnodez, mynode)) {

  //  std::cout << get_nx(nxtot, nnodex, nnodey, nnodez, mynode) << " " << this->nx << std::endl;
  //  std::cout << get_ny(nytot, nnodex, nnodey, nnodez, mynode) << " " << this->ny << std::endl;
  //  std::cout << get_nz(nztot, nnodex, nnodey, nnodez, mynode) << " " << this->nz << std::endl;

  x0 = new int[nnode];
  x1 = new int[nnode];
  y0 = new int[nnode];
  y1 = new int[nnode];
  z0 = new int[nnode];
  z1 = new int[nnode];

  nodeID = new int[nnode];

  send = NULL;
  recv = NULL;

  send_req = NULL;
  recv_req = NULL;

  send_stat = NULL;
  recv_stat = NULL;

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

template <typename T>
MultiNodeMatrix3d<T>::~MultiNodeMatrix3d() {
  delete [] x0;
  delete [] x1;
  delete [] y0;
  delete [] y1;
  delete [] z0;
  delete [] z1;
  delete [] nodeID;
  if (send != NULL) delete [] send;
  if (recv != NULL) delete [] recv;
  if (send_req != NULL) delete [] send_req;
  if (recv_req != NULL) delete [] recv_req;
  if (send_stat != NULL) delete [] send_stat;
  if (recv_stat != NULL) delete [] recv_stat;
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
// Compares two matrices, returns true if the difference is within tolerance
// NOTE: Comparison is done in double precision
//
template <typename T>
bool MultiNodeMatrix3d<T>::compare(Matrix3d<T>* mat, const double tol, double& max_diff) {
  assert(mat->get_nx() == nxtot);
  assert(mat->get_ny() == nytot);
  assert(mat->get_nz() == nztot);

  int res = (int)false;
  int loc_res = (int)false;

  //  if (mynode == 1) {
    Matrix3d<T> loc(x1[mynode]-x0[mynode]+1,
		    y1[mynode]-y0[mynode]+1,
		    z1[mynode]-z0[mynode]+1);

    mat->copy_host(x0[mynode], y0[mynode], z0[mynode],
		   0, 0, 0,
		   x1[mynode]-x0[mynode]+1, y1[mynode]-y0[mynode]+1, z1[mynode]-z0[mynode]+1,
		   &loc);

    //    std::cout << "in multinode:" << std::endl;
    //    mat->print(0,0,0,0,1,1);
    //    loc.print(0,0,0,0,1,1);

    loc_res = (int)Matrix3d<T>::compare(&loc, tol, max_diff);
    //  }

#ifdef USE_MPI
  MPICheck(MPI_Allreduce(&loc_res, &res, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD));
#else
  std::cerr << "MultiNodeMatrix3d::transpose_xyz_yzx, MPI required for now" << std::endl;
  exit(1);
#endif

  return (bool)res;
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
void MultiNodeMatrix3d<T>::setup_transpose_xyz_yzx(MultiNodeMatrix3d<T>* mat) {

  mat_yzx = mat;

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

  assert(nnodex == mat->nnodex);
  assert(nnodey == mat->nnodey);
  assert(nnodez == mat->nnodez);

  assert(x0[mynode] == mat->x0[mynode]);
  assert(x1[mynode] == mat->x1[mynode]);
  assert(y0[mynode] == mat->y0[mynode]);
  assert(y1[mynode] == mat->y1[mynode]);
  assert(z0[mynode] == mat->z0[mynode]);
  assert(z1[mynode] == mat->z1[mynode]);

  loc_transpose = false;

  send = new node_t<T>[nnode];
  recv = new node_t<T>[nnode];

  nrecv = 0;
  nsend = 0;
  for (int inode=0;inode < nnode;inode++) {

    int x0_ol, x1_ol;
    int y0_ol, y1_ol;
    int z0_ol, z1_ol;

    bool overlap;

    overlap = get_vol_overlap(x0_t, x1_t, y0_t, y1_t, z0_t, z1_t,
			      x0[inode], x1[inode], y0[inode], y1[inode], z0[inode], z1[inode],
			      x0_ol, x1_ol, y0_ol, y1_ol, z0_ol, z1_ol);

    if (inode != mynode && overlap) {
      recv[nrecv].x0 = x0_ol;
      recv[nrecv].x1 = x1_ol;
      recv[nrecv].y0 = y0_ol;
      recv[nrecv].y1 = y1_ol;
      recv[nrecv].z0 = z0_ol;
      recv[nrecv].z1 = z1_ol;
      recv[nrecv].xsize = recv[nrecv].x1-recv[nrecv].x0+1;
      recv[nrecv].ysize = recv[nrecv].y1-recv[nrecv].y0+1;
      recv[nrecv].len = (recv[nrecv].x1-recv[nrecv].x0+1)*(recv[nrecv].y1-recv[nrecv].y0+1)*
	(recv[nrecv].z1-recv[nrecv].z0+1);
      //      recv[nrecv].h_data = new T[recv[nrecv].len];
      allocate_host<T>(&recv[nrecv].h_data, recv[nrecv].len);
      allocate<T>(&recv[nrecv].d_data, recv[nrecv].len);
      recv[nrecv].node = inode;
      nrecv++;
    }
    int x0i_t = z0[inode];
    int x1i_t = z1[inode];
    int y0i_t = x0[inode];
    int y1i_t = x1[inode];
    int z0i_t = y0[inode];
    int z1i_t = y1[inode];

    overlap = get_vol_overlap(x0i_t, x1i_t, y0i_t, y1i_t, z0i_t, z1i_t,
			      x0[mynode], x1[mynode], y0[mynode], y1[mynode], z0[mynode], z1[mynode],
			      x0_ol, x1_ol, y0_ol, y1_ol, z0_ol, z1_ol);
    
    if (overlap) {
      send[nsend].x0 = x0_ol;
      send[nsend].x1 = x1_ol;
      send[nsend].y0 = y0_ol;
      send[nsend].y1 = y1_ol;
      send[nsend].z0 = z0_ol;
      send[nsend].z1 = z1_ol;
      if (inode == mynode) {
	// Local transpose
	loc_x0 = send[nsend].x0;
	loc_x1 = send[nsend].x1;
	loc_y0 = send[nsend].y0;
	loc_y1 = send[nsend].y1;
	loc_z0 = send[nsend].z0;
	loc_z1 = send[nsend].z1;
	loc_transpose = true;
      } else {
	send[nsend].xsize = send[nsend].x1-send[nsend].x0+1;
	send[nsend].ysize = send[nsend].y1-send[nsend].y0+1;
	send[nsend].len = (send[nsend].x1-send[nsend].x0+1)*(send[nsend].y1-send[nsend].y0+1)*
	  (send[nsend].z1-send[nsend].z0+1);
	//	send[nsend].h_data = new T[send[nsend].len];
	allocate_host<T>(&send[nsend].h_data, send[nsend].len);
	send[nsend].d_data = NULL;
	send[nsend].node = inode;
	nsend++;
      }
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
  
  std::cout << "send: nsend = " << nsend << std::endl;
  for (int i=0;i < nsend;i++) {
    std::cout << send[i].node<<std::endl;
    std::cout << "x0...x1 = " << send[i].x0 << " ... " << send[i].x1 << std::endl;
    std::cout << "y0...y1 = " << send[i].y0 << " ... " << send[i].y1 << std::endl;
    std::cout << "z0...z1 = " << send[i].z0 << " ... " << send[i].z1 << std::endl;
  }
  */

#ifdef USE_MPI
  if (nrecv > 0) {
    recv_req = (void *)(new MPI_Request[nrecv]);
    recv_stat = (void *)(new MPI_Status[nrecv]);
  }

  if (nsend > 0) {
    send_req = (void *)(new MPI_Request[nsend]);
    send_stat = (void *)(new MPI_Status[nsend]);
  }
#endif

}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
//
template <typename T>
void MultiNodeMatrix3d<T>::transpose_xyz_yzx() {

  int MPI_tag=1;

  // Post receives
  if (nrecv > 0) {
    for (int i=0;i < nrecv;i++) {
#ifdef USE_MPI
      MPICheck(MPI_Irecv(recv[i].h_data, sizeof(T)*recv[i].len, MPI_BYTE,
			 recv[i].node, MPI_tag, MPI_COMM_WORLD, &((MPI_Request *)recv_req)[i]));
#else
      std::cerr << "MultiNodeMatrix3d::transpose_xyz_yzx, MPI required for now" << std::endl;
      exit(1);
#endif
    }
  }

  // Copy data to host
#ifdef USE_MPI
  if (nsend > 0) {
    for (int i=0;i < nsend;i++) {
      /*
      if (mynode == 0) {
	std::cout << "send src: "<< send[i].x0 - x0[mynode] << " " << send[i].y0 - y0[mynode] << " " << send[i].z0 - z0[mynode] << std::endl;
	std::cout << this->xsize << " " << this->ysize << std::endl;
	std::cout << send[i].x1 - send[i].x0 << " " << send[i].y1 - send[i].y0 << " " << send[i].z1 - send[i].z0 << std::endl;
	std::cout << send[i].xsize << " " << send[i].ysize << std::endl;
      }
      */
      copy3D_DtoH<T>(this->data, send[i].h_data,
		     send[i].x0 - x0[mynode],
		     send[i].y0 - y0[mynode],
		     send[i].z0 - z0[mynode],
		     this->xsize, this->ysize,
		     0,0,0,
		     send[i].x1 - send[i].x0 + 1,
		     send[i].y1 - send[i].y0 + 1,
		     send[i].z1 - send[i].z0 + 1,
		     send[i].xsize, send[i].ysize);
      /*
      if (mynode == 0) {
	std::cout << "this->data[8]:" << std::endl;
	this->print(8,8,0,0,0,0);
	std::cout << "h_data[0]:" << std::endl;
	std::cout << send[i].h_data[0] << std::endl;
      }
      */

    }
  }
#endif

  // Post sends
  if (nsend > 0) {
    for (int i=0;i < nsend;i++) {
#ifdef USE_MPI
      // Send via MPI
      MPICheck(MPI_Isend(send[i].h_data, sizeof(T)*send[i].len, MPI_BYTE,
			 send[i].node, MPI_tag, MPI_COMM_WORLD, &((MPI_Request *)send_req)[i]));
#else
      std::cerr << "MultiNodeMatrix3d::transpose_xyz_yzx, MPI required for now" << std::endl;
      exit(1);
#endif
    }
  }

  // Perform local matrix transpose on a sub block:
  // (loc_x0...loc_x1) x (loc_y0...loc_y1) x (loc_z0...loc_z1)

  if (loc_transpose) {
    /*
    std::cout << "mynode = " << mynode << std::endl;
    std::cout << "loc_x =  " << loc_x0 << " ... " << loc_x1 << std::endl;
    std::cout << "loc_y =  " << loc_y0 << " ... " << loc_y1 << std::endl;
    std::cout << "loc_z =  " << loc_z0 << " ... " << loc_z1 << std::endl;
    std::cout << "    x =  " << this->x0[mynode] << " ... " << this->x1[mynode] << std::endl;
    std::cout << "    y =  " << this->y0[mynode] << " ... " << this->y1[mynode] << std::endl;
    std::cout << "    z =  " << this->z0[mynode] << " ... " << this->z1[mynode] << std::endl;
    std::cout << "mat x = " << mat_yzx->x0[mynode] << " ... " << mat_yzx->x1[mynode] << std::endl;
    std::cout << "mat y = " << mat_yzx->y0[mynode] << " ... " << mat_yzx->y1[mynode] << std::endl;
    std::cout << "mat z = " << mat_yzx->z0[mynode] << " ... " << mat_yzx->z1[mynode] << std::endl;
    */

    Matrix3d<T>::transpose_xyz_yzx(loc_x0 - this->x0[mynode],
				   loc_y0 - this->y0[mynode],
				   loc_z0 - this->z0[mynode],
				   loc_y0 - mat_yzx->x0[mynode],
				   loc_z0 - mat_yzx->y0[mynode],
				   loc_x0 - mat_yzx->z0[mynode],
				   loc_x1-loc_x0+1,
				   loc_y1-loc_y0+1,
				   loc_z1-loc_z0+1,
				   mat_yzx);
  }

  //  MPICheck(MPI_Barrier(MPI_COMM_WORLD));
  //  return;

  // Wait for receives and copy to device memory
  if (nrecv > 0) {
    for (int i=0;i < nrecv;i++) {
      int k;
#ifdef USE_MPI
      range_start("MPI_Waitany");
      MPICheck(MPI_Waitany(nrecv, (MPI_Request *)recv_req, &k, (MPI_Status *)recv_stat));
      range_stop();
#else
      k = i;
#endif

      /*
      if (mynode == 1) {
      std::cout << "mynode " << mynode << std::endl;
      std::cout << "mat:  x0...x1 = " << mat_yzx->x0[mynode] << " ... " << mat_yzx->x1[mynode] << std::endl;
      std::cout << "mat:  y0...y1 = " << mat_yzx->y0[mynode] << " ... " << mat_yzx->y1[mynode] << std::endl;
      std::cout << "mat:  z0...z1 = " << mat_yzx->z0[mynode] << " ... " << mat_yzx->z1[mynode] << std::endl;
      std::cout << "recv: x0...x1 = " << recv[k].x0 << " ... " << recv[k].x1 << std::endl;
      std::cout << "recv: y0...y1 = " << recv[k].y0 << " ... " << recv[k].y1 << std::endl;
      std::cout << "recv: z0...z1 = " << recv[k].z0 << " ... " << recv[k].z1 << std::endl;
      }
      */

      copy3D_HtoD<T>(recv[k].h_data, recv[k].d_data, 0, 0, 0,
      		     recv[k].xsize, recv[k].ysize,
		     0,0,0,
      		     recv[k].x1 - recv[k].x0 + 1,
      		     recv[k].y1 - recv[k].y0 + 1,
      		     recv[k].z1 - recv[k].z0 + 1,
      		     recv[k].xsize, recv[k].ysize);

      Matrix3d<T> loc(recv[k].x1-recv[k].x0+1,
		      recv[k].y1-recv[k].y0+1,
		      recv[k].z1-recv[k].z0+1,
		      recv[k].xsize, recv[k].ysize, recv[k].z1-recv[k].z0+1, 
		      recv[k].d_data);

      /*
      if (mynode == 0) {
	std::cout << "node 0:" << std::endl;
	Matrix3d<T>::print(32,32,0,0,0,0);
      }

      if (mynode == 1) {
	mat_yzx->print(0,0,0,0,0,0);
	loc.print(0,0,0,0,0,0);
      }

      if (mynode == 1) {
	std::cout << "dst: " << recv[k].y0 - mat_yzx->x0[mynode] << " " << recv[k].z0 - mat_yzx->y0[mynode] << " " << recv[k].x0 - mat_yzx->z0[mynode] << std::endl;
	std::cout << "len: " << recv[k].x1-recv[k].x0+1 << " " << recv[k].y1-recv[k].y0+1 << " " << recv[k].z1-recv[k].z0+1 << std::endl;
      }
      */

      loc.transpose_xyz_yzx(0,0,0,
			    recv[k].y0 - mat_yzx->x0[mynode],
			    recv[k].z0 - mat_yzx->y0[mynode],
			    recv[k].x0 - mat_yzx->z0[mynode],
			    recv[k].x1-recv[k].x0+1,
			    recv[k].y1-recv[k].y0+1,
			    recv[k].z1-recv[k].z0+1,
			    mat_yzx);

      //      if (mynode == 1) mat_yzx->print(0,0,0,0,0,0);

    }
  }

  // Wait for sends to finish
  if (nsend > 0) {
#ifdef USE_MPI
    MPICheck(MPI_Waitall(nsend, (MPI_Request *)send_req, (MPI_Status *)send_stat));
#endif
  }

}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
//
template <typename T>
void MultiNodeMatrix3d<T>::transpose_xyz_yzx(MultiNodeMatrix3d<T>* mat) {
  setup_transpose_xyz_yzx(mat);
  transpose_xyz_yzx();
}

//
// Explicit instances of MultiNodeMatrix3d
//
template class MultiNodeMatrix3d<float>;
//template class MultiNodeMatrix3d<float2>;
template class MultiNodeMatrix3d<long long int>;
