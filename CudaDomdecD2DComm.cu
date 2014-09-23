#include <iostream>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/gather.h>
#include "CudaDomdecD2DComm.h"
#include "mpi_utils.h"

//################################################################################
//################################################################################
//################################################################################

CudaDomdecD2DComm::CudaDomdecD2DComm(Domdec& domdec, CudaMPI& cudaMPI) : 
  DomdecD2DComm(domdec), cudaMPI(cudaMPI) {

  // Send
  sendbuf_len = 0;
  sendbuf = NULL;

  h_sendbuf_len = 0;
  h_sendbuf = NULL;

  z_send_loc.resize(nz_comm);
  
  // Recv
  recvbuf_len = 0;
  recvbuf = NULL;

  h_recvbuf_len = 0;
  h_recvbuf = NULL;

  // MPI requests
  int max_n_comm = std::max(std::max(nx_comm,ny_comm), nz_comm);
  request.reserve(2*max_n_comm);

}

CudaDomdecD2DComm::~CudaDomdecD2DComm() {
  if (sendbuf != NULL) deallocate<char>(&sendbuf);
  if (h_sendbuf != NULL) deallocate_host<char>(&h_sendbuf);
  if (recvbuf != NULL) deallocate<char>(&recvbuf);
  if (h_recvbuf != NULL) deallocate_host<char>(&h_recvbuf);
}

struct z_pick_functor {
  const double zb, inv_boxz;

  z_pick_functor(double zb, double inv_boxz) : zb(zb), inv_boxz(inv_boxz) {}

  //
  // Returns 1 if coordinate is within the z boundary (zb)
  //
  __host__ __device__ unsigned char operator()(const double& z) const {
    double zf = z*inv_boxz + 0.5;
    zf -= floor(zf);
    // Now zf = (0.0 ... 1.0)
    return (zf < zb);
  }

};

//
// Communicate coordinates
//
void CudaDomdecD2DComm::comm_coord(cudaXYZ<double> *coord, thrust::device_vector<int>& loc2glo,
				   const bool update) {

  double rnl = domdec.get_rnl();
  double inv_boxx = domdec.get_inv_boxx();
  double inv_boxy = domdec.get_inv_boxy();
  double inv_boxz = domdec.get_inv_boxz();
  int homeix = domdec.get_homeix();
  int homeiy = domdec.get_homeiy();
  int homeiz = domdec.get_homeiz();

  const int COUNT_TAG = 1, DATA_TAG = 2;
  
  // Size of each buffer elements
  const int buf_elem_size = update ? (sizeof(int) + 3*sizeof(double)) : (3*sizeof(double));

  if (nz_comm > 0) {

    double rnl_grouped = rnl;
    int pos = 0;
    for (int i=0;i < nz_comm;i++) {
      
      if (update) {
	// Neighborlist has been updated => update communicated atoms
	double zf;
	get_fz_boundary(homeix, homeiy, homeiz-(i+1), rnl, rnl_grouped, zf);
	if (homeiz-(i+1) < 0) zf -= 1.0;

	// Get pointer to z coordinates
	thrust::device_ptr<double> z_ptr(&coord->data[coord->stride*2]);

	// Pick atoms that are in the communication region
	thrust::transform(z_ptr, z_ptr + coord->n, atom_pick.begin(),
			  z_pick_functor(zf + rnl*inv_boxz, inv_boxz));

	// atom_pick[] now contains atoms that are picked for z-communication
	// Exclusive cumulative sum to find picked atom positions
	thrust::exclusive_scan(atom_pick.begin(), atom_pick.end(), atom_pos.begin());
	
	// Count the number of atoms we are adding to the buffer
	z_nsend[i] = atom_pos[coord->n] + atom_pick[coord->n];
	z_psend[i] = (i == 0) ? 0 : (z_psend[i-1] + z_nsend[i-1]);

	// atom_pos[] now contains position to store each atom
	// Scatter to produce packed atom index table
	thrust::scatter_if(thrust::make_counting_iterator(0),
			   thrust::make_counting_iterator(coord->n),
			   atom_pos.begin(), atom_pick.begin(),
			   z_send_loc[i].begin());

	// z_send_loc[i][] now contains the local indices of atoms

	// Re-allocate sendbuf if needed
	int req_sendbuf_len = (z_psend[i] + z_nsend[i])*buf_elem_size;
	reallocate<char>(&sendbuf, &sendbuf_len, req_sendbuf_len, 1.5f);

	// Get int pointer to sendbuf
	thrust::device_ptr<int> sendbuf_ind_ptr((int *)&sendbuf[pos]);
	
	// Pack in atom global indices to sendbuf_ind_ptr
	thrust::gather(z_send_loc[i].begin(), z_send_loc[i].end(), loc2glo.begin(), sendbuf_ind_ptr);

	// Advance sendbuf position
	pos += z_nsend[i]*sizeof(int);
      }

      // Get double pointer to send buffer
      thrust::device_ptr<double> sendbuf_xyz_ptr((double *)&sendbuf[pos]);

      // Get pointer to coordinates
      thrust::device_ptr<double> xyz_ptr(&coord->data[0]);
      
      // Pack in coordinates
      thrust::gather(z_send_loc[i].begin(), z_send_loc[i].end(), xyz_ptr,
		     sendbuf_xyz_ptr);

      thrust::gather(z_send_loc[i].begin(), z_send_loc[i].end(), xyz_ptr + coord->stride,
		     sendbuf_xyz_ptr + z_nsend[i]);

      thrust::gather(z_send_loc[i].begin(), z_send_loc[i].end(), xyz_ptr + coord->stride*2,
		     sendbuf_xyz_ptr + 2*z_nsend[i]);

      pos += z_nsend[i]*3*sizeof(double);
    } // for (int i=1;i < nz_comm;i++)
    
    if (update) {
      // Total send size
      z_psend[nz_comm] = z_psend[nz_comm-1] + z_nsend[nz_comm-1];
      // Re-allocate h_sendbuf if needed
      if (!cudaMPI.isCudaAware()) {
	reallocate_host<char>(&h_sendbuf, &h_sendbuf_len, z_psend[nz_comm]*buf_elem_size, 1.2f);
      }
      // Send & receive data counts
      int nrequest = 0;
      for (int i=0;i < nz_comm;i++) {
	MPICheck(MPI_Isend(&z_nsend[i], 1, MPI_INT, z_send_node[i], COUNT_TAG, cudaMPI.get_comm(),
			   &request[nrequest++]));
	MPICheck(MPI_Irecv(&z_nrecv[i], 1, MPI_INT, z_recv_node[i], COUNT_TAG, cudaMPI.get_comm(),
			   &request[nrequest++]));
      }
      MPICheck(MPI_Waitall(nrequest, request.data(), MPI_STATUSES_IGNORE));
      for (int i=0;i <= nz_comm;i++) z_precv[i] = (i == 0) ? 0 : (z_precv[i-1] + z_nrecv[i-1]);
      // Re-allocate receive buffers
      reallocate<char>(&sendbuf, &sendbuf_len, z_precv[nz_comm]*buf_elem_size, 1.2f);
      if (!cudaMPI.isCudaAware()) {
	reallocate_host<char>(&h_recvbuf, &h_recvbuf_len, z_precv[nz_comm]*buf_elem_size, 1.2f);
      }
    }

    int nrequest = 0;
    // Receive data
    for (int i=0;i < nz_comm;i++) {
      if (z_nrecv[i] > 0) {
	MPICheck(cudaMPI.Irecv(&recvbuf[z_precv[i]], z_nrecv[i]*buf_elem_size, z_recv_node[i], DATA_TAG,
			       &request[nrequest++], &h_recvbuf[z_precv[i]]));
      }
    }

    // Send data
    for (int i=0;i < nz_comm;i++) {
      if (z_nsend[i] > 0) {
	MPICheck(cudaMPI.Isend(&sendbuf[z_psend[i]], z_nsend[i]*buf_elem_size, z_send_node[i], DATA_TAG,
			       &request[nrequest++], &h_sendbuf[z_psend[i]]));
      }
    }

    // For for send and receive to finish
    MPICheck(MPI_Waitall(nrequest, request.data(), MPI_STATUSES_IGNORE));

    // Unpack data from +z-direction into correct arrays
    for (int i=0;i < nz_comm;i++) {
      if (update) {
      }
    }

  } // if (nz_comm > 0)
  
  if (ny_comm > 0) {
    std::cout << "CudaDomdecD2DComm::comm_coord, y-communication not yet implemented" << std::endl;
    exit(1);
  }

  if (nx_comm > 0) {
    std::cout << "CudaDomdecD2DComm::comm_coord, x-communication not yet implemented" << std::endl;
    exit(1);
  }

}
