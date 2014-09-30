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
  //int max_n_comm = std::max(std::max(nx_comm,ny_comm), nz_comm);
  //request.reserve(2*max_n_comm);

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

inline int alignInt(const int pos, const int align) {
  return ((pos-1)/align+1)*align;
}

//
// Communicate coordinates
//
void CudaDomdecD2DComm::comm_coord(cudaXYZ<double>& coord, thrust::device_vector<int>& loc2glo,
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

  // Resize arrays
  if (nx_comm + ny_comm + nz_comm > 0 && update) {
    atom_pick.resize(coord.size()+1);
    atom_pos.resize(coord.size()+1);
  }

  if (nz_comm > 0) {

    double rnl_grouped = rnl;
    int pos = 0;
    z_psend.at(0) = 0;
    for (int i=0;i < nz_comm;i++) {
      
      if (update) {
	// Neighborlist has been updated => update communicated atoms
	double zf;
	get_fz_boundary(homeix, homeiy, homeiz-(i+1), rnl, rnl_grouped, zf);
	//if (homeiz-(i+1) < 0) zf -= 1.0;

	fprintf(stderr,"%d: homeiz=%d zf=%lf\n",domdec.get_mynode(),homeiz,zf);

	// Get pointer to z coordinates
	thrust::device_ptr<double> z_ptr(coord.z());

	// Pick atoms that are in the communication region
	thrust::transform(z_ptr, z_ptr + coord.size(), atom_pick.begin(),
			  z_pick_functor(zf + rnl*inv_boxz, inv_boxz));

	// atom_pick[] now contains atoms that are picked for z-communication
	// Exclusive cumulative sum to find picked atom positions
	thrust::exclusive_scan(atom_pick.begin(), atom_pick.end(), atom_pos.begin());
	
	// Count the number of atoms we are adding to the buffer
	z_nsend.at(i) = atom_pos[coord.size()];
	z_psend.at(i+1) = z_psend.at(i) + z_nsend.at(i);

	z_send_loc.at(i).resize(z_nsend.at(i));

	// atom_pos[] now contains position to store each atom
	// Scatter to produce packed atom index table
	thrust::scatter_if(thrust::make_counting_iterator(0),
			   thrust::make_counting_iterator(coord.size()),
			   atom_pos.begin(), atom_pick.begin(),
			   z_send_loc.at(i).begin());
	
	// z_send_loc[i][] now contains the local indices of atoms

	// Re-allocate sendbuf if needed
	int req_sendbuf_len = pos + alignInt(z_nsend.at(i),2)*sizeof(int) + 
	  z_nsend.at(i)*3*sizeof(double);
	reallocate<char>(&sendbuf, &sendbuf_len, req_sendbuf_len, 1.5f);

	// Get int pointer to sendbuf
	thrust::device_ptr<int> sendbuf_ind_ptr((int *)&sendbuf[pos]);
	
	// Pack in atom global indices to sendbuf[]
	thrust::gather(z_send_loc.at(i).begin(), z_send_loc.at(i).end(),
		       loc2glo.begin(), sendbuf_ind_ptr);

	// Advance sendbuf position
	pos += alignInt(z_nsend.at(i),2)*sizeof(int);
      }

      // Get double pointer to send buffer
      thrust::device_ptr<double> sendbuf_xyz_ptr((double *)&sendbuf[pos]);

      // Pack in coordinates to sendbuf[]
      thrust::device_ptr<double> x_ptr(coord.x());
      thrust::gather(z_send_loc.at(i).begin(), z_send_loc.at(i).end(),
		     x_ptr, sendbuf_xyz_ptr);

      thrust::device_ptr<double> y_ptr(coord.y());
      thrust::gather(z_send_loc.at(i).begin(), z_send_loc.at(i).end(),
		     y_ptr, sendbuf_xyz_ptr + z_nsend.at(i));

      thrust::device_ptr<double> z_ptr(coord.z());
      thrust::gather(z_send_loc.at(i).begin(), z_send_loc.at(i).end(),
		     z_ptr, sendbuf_xyz_ptr + 2*z_nsend.at(i));

      pos += z_nsend[i]*3*sizeof(double);
    } // for (int i=1;i < nz_comm;i++)

    // Compute byte positions
    computeByteNumPos(nz_comm, z_nsend, nsend, psend, update);
    if (pos != psend.at(nz_comm)) {
      std::cout << "CudaDomdecD2DComm::comm_coord, invalid pos (z)" << std::endl;
      exit(1);
    }

    if (update) {
      // Re-allocate h_sendbuf if needed
      if (!cudaMPI.isCudaAware()) {
	reallocate_host<char>(&h_sendbuf, &h_sendbuf_len, psend.at(nz_comm), 1.2f);
      }
      // Send & receive data counts
      for (int i=0;i < nz_comm;i++) {
	MPICheck(MPI_Sendrecv(&z_nsend.at(i), 1, MPI_INT, z_send_node.at(i), COUNT_TAG,
			      &z_nrecv.at(i), 1, MPI_INT, z_recv_node.at(i), COUNT_TAG,
			      cudaMPI.get_comm(), MPI_STATUS_IGNORE));
      }
      // Compute positions
      z_precv.at(0) = 0;
      for (int i=0;i < nz_comm;i++) z_precv.at(i+1) = z_precv.at(i) + z_nrecv.at(i);
    }

    // Compute byte positions
    computeByteNumPos(nz_comm, z_nrecv, nrecv, precv, update);

    if (update) {
      // Re-allocate receive buffers
      reallocate<char>(&recvbuf, &recvbuf_len, precv.at(nz_comm), 1.2f);
      if (!cudaMPI.isCudaAware()) {
	reallocate_host<char>(&h_recvbuf, &h_recvbuf_len, precv.at(nz_comm), 1.2f);
      }
      z_recv_ind.resize(precv.at(nz_comm));
      coord.resize(coord.size()+precv.at(nz_comm));
    }

    // Send & Recv data
    for (int i=0;i < nz_comm;i++) {
      if (nsend.at(i) > 0 && nrecv.at(i) > 0) {
	MPICheck(cudaMPI.Sendrecv(&sendbuf[psend.at(i)], nsend.at(i),
				  z_send_node.at(i), DATA_TAG,
				  &recvbuf[precv.at(i)], nrecv.at(i),
				  z_recv_node.at(i), DATA_TAG, MPI_STATUS_IGNORE,
				  &h_sendbuf[psend.at(i)], &h_recvbuf[precv.at(i)]));

      } else if (nsend.at(i) > 0) {
	MPICheck(cudaMPI.Send(&sendbuf[psend.at(i)], nsend.at(i),
			      z_send_node.at(i), DATA_TAG, &h_sendbuf[psend.at(i)]));
      } else if (nrecv.at(i) > 0) {
	MPICheck(cudaMPI.Recv(&recvbuf[precv.at(i)], nrecv.at(i),
			      z_recv_node.at(i), DATA_TAG, MPI_STATUS_IGNORE,
			      &h_recvbuf[precv.at(i)]));
      }
    }    

    //----------------------------------------------------
    // Unpack data from +z-direction into correct arrays
    //----------------------------------------------------
    // Position where we start adding coordinates
    int cpos = domdec.get_zone_pcoord()[0];
    for (int i=0;i < nz_comm;i++) {
      int pos = 0;
      int src_pos = precv.at(i);
      if (update) {
	// Copy coordinates indices to z_recv_ind
	// format = indices[alignInt(nrecv.at(i),2) x int]
	thrust::device_ptr<double> ind_ptr((double *)&recvbuf[src_pos]);
	thrust::copy(ind_ptr, ind_ptr+z_nrecv.at(i), z_recv_ind.begin()+pos);
	pos += z_nrecv.at(i);
	src_pos += alignInt(z_nrecv.at(i),2)*sizeof(int);
      }
      // Unpack coordinates
      // format = X[nrecv.at(i) x double] | Y[nrecv.at(i) x double] | Z[nrecv.at(i) x double]
      
      thrust::device_ptr<double> x_src_ptr((double *)&recvbuf[src_pos]);
      thrust::device_ptr<double> x_dst_ptr(coord.x() + cpos);
      thrust::copy(x_src_ptr, x_src_ptr+z_nrecv.at(i), x_dst_ptr);
      src_pos += z_nrecv.at(i)*sizeof(double);

      thrust::device_ptr<double> y_src_ptr((double *)&recvbuf[src_pos]);
      thrust::device_ptr<double> y_dst_ptr(coord.y() + cpos);
      thrust::copy(y_src_ptr, y_src_ptr+z_nrecv.at(i), y_dst_ptr);
      src_pos += z_nrecv.at(i)*sizeof(double);

      thrust::device_ptr<double> z_src_ptr((double *)&recvbuf[src_pos]);
      thrust::device_ptr<double> z_dst_ptr(coord.z() + cpos);
      thrust::copy(z_src_ptr, z_src_ptr+z_nrecv.at(i), z_dst_ptr);
      src_pos += z_nrecv.at(i)*sizeof(double);
      cpos += z_nrecv.at(i);
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

void CudaDomdecD2DComm::computeByteNumPos(const int nc_comm, std::vector<int>& c_nsend,
					  std::vector<int>& nsend, std::vector<int>& psend,
					  const bool update) {
  nsend.resize(nc_comm);
  psend.resize(nc_comm+1);
  psend.at(0) = 0;
  if (update) {
    for (int i=0;i < nc_comm;i++) {
      nsend.at(i) = alignInt(c_nsend.at(i),2)*sizeof(int) + c_nsend.at(i)*3*sizeof(double);
      psend.at(i+1) = psend.at(i) + nsend.at(i);
    }
  } else {
    for (int i=0;i < nc_comm;i++) {
      nsend.at(i) = c_nsend.at(i)*3*sizeof(double);
      psend.at(i+1) = psend.at(i) + nsend.at(i);
    }
  }
}
