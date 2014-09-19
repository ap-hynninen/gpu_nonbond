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

  sendbuf_len = 0;
  sendbuf = NULL;

  h_sendbuf_len = 0;
  h_sendbuf = NULL;
  
  z_nsend.resize(get_nz_comm());
  z_send_loc.resize(get_nz_comm());
}

CudaDomdecD2DComm::~CudaDomdecD2DComm() {
  if (sendbuf != NULL) deallocate<char>(&sendbuf);
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
void CudaDomdecD2DComm::comm_coord(cudaXYZ<double> *coord, int *loc2glo, const bool update) {

  double rnl = domdec.get_rnl();
  double inv_boxx = domdec.get_inv_boxx();
  double inv_boxy = domdec.get_inv_boxy();
  double inv_boxz = domdec.get_inv_boxz();
  int homeix = domdec.get_homeix();
  int homeiy = domdec.get_homeiy();
  int homeiz = domdec.get_homeiz();

  const int TAG = 1;

  // Get pointer to local -> global mapping
  thrust::device_ptr<int> loc2glo_ptr(loc2glo);

  int nrequest = 0;

  if (get_nz_comm() > 0) {

    // Start receiving
    if (!update) {
      for (int i=1;i <= get_nz_comm();i++) {
	nrequest++;
      }
    }

    double rnl_grouped = rnl;
    int pos = 0;
    for (int i=1;i <= get_nz_comm();i++) {
      
      int pos_start = pos;

      if (update) {
	// Neighborlist has been updated => update communicated atoms
	double zf;
	get_fz_boundary(homeix, homeiy, homeiz-i, rnl, rnl_grouped, zf);
	if (homeiz-i < 0) zf -= 1.0;

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
	
	// atom_pos[] now contains position to store each atom
	// Scatter to produce packed atom index table
	thrust::scatter_if(thrust::make_counting_iterator(0),
			   thrust::make_counting_iterator(coord->n),
			   atom_pos.begin(), atom_pick.begin(),
			   z_send_loc[i].begin());

	// z_send_loc[i][] now contains the local indices of atoms

	// Re-allocate sendbuf if needed
	int req_sendbuf_len = pos + z_nsend[i]*(sizeof(int) + 3*sizeof(double));
	reallocate<char>(&sendbuf, &sendbuf_len, req_sendbuf_len, 1.5f);
	if (!cudaMPI.isCudaAware()) {
	  reallocate_host<char>(&h_sendbuf, &h_sendbuf_len, req_sendbuf_len, 1.5f);
	}

	// Get int pointer to sendbuf
	thrust::device_ptr<int> sendbuf_ind_ptr((int *)&sendbuf[pos]);
	
	// Pack in atom global indices to sendbuf_ind_ptr
	thrust::gather(z_send_loc[i].begin(), z_send_loc[i].end(), loc2glo_ptr, sendbuf_ind_ptr);

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
      
      int send_len = pos - pos_start;

      MPICheck(cudaMPI.Isend(&sendbuf[pos_start], send_len, z_send_node[i], TAG,
			     &request[nrequest], &h_sendbuf[pos_start]));
      nrequest++;

    }

  } // if (get_nz_comm() > 0)
  
  /*
  // For update=true, special receive that allows us to re-size arrays before receiving the data
  if (update) {
    for (int i=1;i <= get_nz_comm();i++) {
      MPICheck(MPI_Probe(z_send_node[i], TAG, cudaMPI.get_comm(), &status[i]));
      MPICheck(MPI_Get_count(&status[i], MPI_BYTE, &count[i]));
    }
    for (int i=1;i <= get_nz_comm();i++) {
      MPICheck(MPI_);
    }
  }
  */

  // Wait for communication to stop
  if (nrequest > 0) {
    MPICheck(MPI_Waitall(nrequest, request.data(), MPI_STATUSES_IGNORE));
  }

  if (get_ny_comm() > 0) {
    std::cout << "CudaDomdecD2DComm::comm_coord, y-communication not yet implemented" << std::endl;
    exit(1);
  }

  if (get_nx_comm() > 0) {
    std::cout << "CudaDomdecD2DComm::comm_coord, x-communication not yet implemented" << std::endl;
    exit(1);
  }

}
