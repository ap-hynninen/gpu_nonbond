#include <iostream>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/gather.h>
#include "CudaDomdecD2DComm.h"
#include "hostXYZ.h"
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

  z_send_loc0.resize(nz_comm);
  
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
// Pack n coordinates in (x,y,z) with indices in ind to outbuf
// outbuf[i] = x[ind[i]], i=0...n-1
//
void CudaDomdecD2DComm::packXYZ(double* x, double* y, double* z, int* ind, int n, double *outbuf) {

  thrust::device_ptr<int> ind_ptr(ind);
  thrust::device_ptr<double> x_ptr(x);
  thrust::device_ptr<double> y_ptr(y);
  thrust::device_ptr<double> z_ptr(z);
  thrust::device_ptr<double> outbuf_ptr(outbuf);

  thrust::gather(ind_ptr, ind_ptr+n, x_ptr, outbuf_ptr);
  thrust::gather(ind_ptr, ind_ptr+n, y_ptr, outbuf_ptr+n);
  thrust::gather(ind_ptr, ind_ptr+n, z_ptr, outbuf_ptr+2*n);

}

//
// Unpack n coordinates from inbuf to (x, y, z) using ind for locations
// x[ind[i]] = inbuf[i], i=0...n-1
//
void CudaDomdecD2DComm::unpackXYZ(double* inbuf, int n, double* x, double* y, double* z, int* ind) {

  thrust::device_ptr<double> x_ptr(x);
  thrust::device_ptr<double> y_ptr(y);
  thrust::device_ptr<double> z_ptr(z);
  thrust::device_ptr<double> inbuf_ptr(inbuf);

  if (ind == NULL) {
    thrust::copy(inbuf_ptr,     inbuf_ptr+n,   x_ptr);
    thrust::copy(inbuf_ptr+n,   inbuf_ptr+2*n, y_ptr);
    thrust::copy(inbuf_ptr+2*n, inbuf_ptr+3*n, z_ptr);
  } else {
    thrust::device_ptr<int> ind_ptr(ind);
    thrust::scatter(inbuf_ptr,     inbuf_ptr+n,   ind_ptr, x_ptr);
    thrust::scatter(inbuf_ptr+n,   inbuf_ptr+2*n, ind_ptr, y_ptr);
    thrust::scatter(inbuf_ptr+2*n, inbuf_ptr+3*n, ind_ptr, z_ptr);
  }
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

  // Resize arrays
  if (nx_comm + ny_comm + nz_comm > 0 && update) {
    // NOTE: Atoms are only picked from the homezone = first ncoord atoms in coord
    atom_pick.resize(domdec.get_ncoord()+1);
    atom_pos.resize(domdec.get_ncoord()+1);
  }

  if (nz_comm > 0) {

    double rnl_grouped = rnl;
    int pos = 0;
    if (update) z_psend.at(0) = 0;
    for (int i=0;i < nz_comm;i++) {
      
      if (update) {
	// Neighborlist has been updated => update communicated atoms
	double zf;
	get_fz_boundary(homeix, homeiy, homeiz-(i+1), rnl, rnl_grouped, zf);

	// Get pointer to z coordinates
	thrust::device_ptr<double> z_ptr(coord.z());

	// Pick atoms that are in the communication region
	thrust::transform(z_ptr, z_ptr + domdec.get_ncoord(), atom_pick.begin(),
			  z_pick_functor(zf + rnl*inv_boxz, inv_boxz));

	// atom_pick[] now contains atoms that are picked for z-communication
	// Exclusive cumulative sum to find picked atom positions
	thrust::exclusive_scan(atom_pick.begin(), atom_pick.end(), atom_pos.begin());
	
	// Count the number of atoms we are adding to the buffer
	z_nsend.at(i) = atom_pos[domdec.get_ncoord()];
	z_psend.at(i+1) = z_psend.at(i) + z_nsend.at(i);

	z_send_glo.resize(z_psend.at(i+1));

	z_send_loc0.at(i).resize(z_nsend.at(i));

	// atom_pos[] now contains position to store each atom
	// Scatter to produce packed atom index table
	thrust::scatter_if(thrust::make_counting_iterator(0),
			   thrust::make_counting_iterator(domdec.get_ncoord()),
			   atom_pos.begin(), atom_pick.begin(),
			   z_send_loc0.at(i).begin());
	
	// z_send_loc0[i][] now contains the local indices of atoms

	// Re-allocate sendbuf if needed
	int req_sendbuf_len = pos + alignInt(z_nsend.at(i),2)*sizeof(int) + 
	  z_nsend.at(i)*3*sizeof(double);
	resize<char>(&sendbuf, &sendbuf_len, sendbuf_len, req_sendbuf_len, 1.5f);

	// Get int pointer to sendbuf
	thrust::device_ptr<int> sendbuf_ind_ptr((int *)&sendbuf[pos]);
	
	// Pack in atom global indices to sendbuf[]
	thrust::gather(z_send_loc0.at(i).begin(), z_send_loc0.at(i).end(),
		       loc2glo.begin(), sendbuf_ind_ptr);

	// Make a copy of global indices to z_send_glo
	thrust::copy(sendbuf_ind_ptr, sendbuf_ind_ptr+z_nsend.at(i),
		     z_send_glo.begin()+z_psend.at(i));

	// Advance sendbuf position
	pos += alignInt(z_nsend.at(i),2)*sizeof(int);

	// Pack in coordinates to sendbuf[]
	packXYZ(coord.x(), coord.y(), coord.z(),
		thrust::raw_pointer_cast(z_send_loc0.at(i).data()), z_nsend.at(i),
		(double *)&sendbuf[pos]);
      } else {
	// Pack in coordinates to sendbuf[]
	packXYZ(coord.x(), coord.y(), coord.z(),
		thrust::raw_pointer_cast(z_send_loc.data())+z_psend.at(i), z_nsend.at(i),
		(double *)&sendbuf[z_psend.at(i)*3*sizeof(double)]);
      }

      pos += z_nsend.at(i)*3*sizeof(double);
    } // for (int i=1;i < nz_comm;i++)

    // Compute byte positions
    computeByteNumPos(nz_comm, z_nsend, nsendByte, psendByte, update);
    if (pos != psendByte.at(nz_comm)) {
      std::cout << "CudaDomdecD2DComm::comm_coord, invalid pos (z)" << std::endl;
      exit(1);
    }

    if (update) {
      // Re-allocate h_sendbuf if needed
      if (!cudaMPI.isCudaAware()) {
	reallocate_host<char>(&h_sendbuf, &h_sendbuf_len, psendByte.at(nz_comm), 1.2f);
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
    computeByteNumPos(nz_comm, z_nrecv, nrecvByte, precvByte, update);

    if (update) {
      // Re-allocate receive buffers
      reallocate<char>(&recvbuf, &recvbuf_len, precvByte.at(nz_comm), 1.2f);
      if (!cudaMPI.isCudaAware()) {
	reallocate_host<char>(&h_recvbuf, &h_recvbuf_len, precvByte.at(nz_comm), 1.2f);
      }
      z_recv_glo.resize(z_precv.at(nz_comm));
      coord.resize(domdec.get_ncoord()+z_precv.at(nz_comm));
      // Re-allocate loc2glo
      loc2glo.resize(domdec.get_ncoord() + z_precv.at(nz_comm));
    }

    // Wait until packing is done
    cudaCheck(cudaDeviceSynchronize());

    // Send & Recv data
    for (int i=0;i < nz_comm;i++) {
      if (nsendByte.at(i) > 0 && nrecvByte.at(i) > 0) {
	MPICheck(cudaMPI.Sendrecv(&sendbuf[psendByte.at(i)], nsendByte.at(i),
				  z_send_node.at(i), DATA_TAG,
				  &recvbuf[precvByte.at(i)], nrecvByte.at(i),
				  z_recv_node.at(i), DATA_TAG, MPI_STATUS_IGNORE,
				  &h_sendbuf[psendByte.at(i)], &h_recvbuf[precvByte.at(i)]));
      } else if (nsendByte.at(i) > 0) {
	MPICheck(cudaMPI.Send(&sendbuf[psendByte.at(i)], nsendByte.at(i),
			      z_send_node.at(i), DATA_TAG, &h_sendbuf[psendByte.at(i)]));
      } else if (nrecvByte.at(i) > 0) {
	MPICheck(cudaMPI.Recv(&recvbuf[precvByte.at(i)], nrecvByte.at(i),
			      z_recv_node.at(i), DATA_TAG, MPI_STATUS_IGNORE,
			      &h_recvbuf[precvByte.at(i)]));
      }
    }    

    //----------------------------------------------------
    // Unpack data from +z-direction into correct arrays
    //----------------------------------------------------
    int src_pos = precvByte.at(0);
    for (int i=0;i < nz_comm;i++) {
      if (update) {
	// Copy global indices to z_recv_glo
	// format = indices[alignInt(z_nrecv.at(i),2) x int]
	thrust::device_ptr<int> ind_ptr((int *)&recvbuf[src_pos]);
	thrust::copy(ind_ptr, ind_ptr+z_nrecv.at(i), z_recv_glo.begin()+z_precv.at(i));
	// Copy global indices to loc2glo
	thrust::copy(ind_ptr, ind_ptr+z_nrecv.at(i), 
		     loc2glo.begin()+domdec.get_zone_pcoord(Domdec::FZ)+z_precv.at(i));
	src_pos += alignInt(z_nrecv.at(i),2)*sizeof(int);
	// Unpack coordinates

	unpackXYZ((double *)&recvbuf[src_pos], z_nrecv.at(i),
		  coord.x()+domdec.get_zone_pcoord(Domdec::FZ)+z_precv.at(i),
		  coord.y()+domdec.get_zone_pcoord(Domdec::FZ)+z_precv.at(i),
		  coord.z()+domdec.get_zone_pcoord(Domdec::FZ)+z_precv.at(i));

      } else {
	// Unpack coordinates
	// format = X[nrecv.at(i) x double] | Y[nrecv.at(i) x double] | Z[nrecv.at(i) x double]
	unpackXYZ((double *)&recvbuf[src_pos], z_nrecv.at(i),
		  coord.x(), coord.y(), coord.z(),
		  thrust::raw_pointer_cast(z_recv_loc.data())+z_precv.at(i));
      }
      src_pos += z_nrecv.at(i)*3*sizeof(double);
    }
    if (src_pos != precvByte.at(nz_comm)) {
      std::cout << "CudaDomdecD2DComm::comm_coord, invalid src_pos (z)" << std::endl;
      exit(1);
    }

    if (update) {
      domdec.set_zone_ncoord(Domdec::FZ, z_precv.at(nz_comm));
    }

    // Wait until unpacking is done
    cudaCheck(cudaDeviceSynchronize());

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

//
// Test comm_coord() method
//
void CudaDomdecD2DComm::test_comm_coord(cudaXYZ<double>& coord) {
  // Allocate temporary arrays
  int* loc2glo = new int[domdec.get_ncoord()];
  int* loc2glo_glo = new int[domdec.get_ncoord_glo()];
  int* nrecv = new int[domdec.get_numnode()];
  int* precv = new int[domdec.get_numnode()];
  double* recvbuf = new double[domdec.get_ncoord_glo()];
  double *x = new double[domdec.get_ncoord_glo()];
  double *y = new double[domdec.get_ncoord_glo()];
  double *z = new double[domdec.get_ncoord_glo()];

  hostXYZ<double> h_coord(domdec.get_ncoord(), NON_PINNED);

  copy_DtoH_sync<int>(domdec.get_loc2glo_ptr(), loc2glo, domdec.get_ncoord());
  domdec.buildGlobal_loc2glo(loc2glo, loc2glo_glo, nrecv, precv);

  // Set global coordinates to (x, y, z)
  h_coord.set_data_sync(domdec.get_ncoord(), coord);
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, h_coord.x(), x);
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, h_coord.y(), y);
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, h_coord.z(), z);
  // Set local coordinates to h_coord
  h_coord.set_data_sync(domdec.get_ncoord_tot(), coord);

  // Test
  test_comm_coord2(loc2glo, h_coord.x(), h_coord.y(), h_coord.z(), x, y, z);
  // Deallocate temporary arrays
  delete [] loc2glo;
  delete [] loc2glo_glo;
  delete [] nrecv;
  delete [] precv;
  delete [] recvbuf;
  delete [] x;
  delete [] y;
  delete [] z;
}

//
// Updates (z_recv_loc)
// Called after neighborlist update has re-sorted coordinates
// glo2loc = global -> local mapping
//
void CudaDomdecD2DComm::comm_update(int* glo2loc, cudaXYZ<double>& coord) {
  thrust::device_ptr<int> glo2loc_ptr(glo2loc);

  if (nz_comm > 0) {
    // Compute byte positions
    computeByteNumPos(nz_comm, z_nsend, nsendByte, psendByte, true);
    // Re-allocate
    z_recv_loc.resize(z_precv.at(nz_comm));
    z_send_loc.resize(z_psend.at(nz_comm));
    // Map: z_recv_glo => z_recv_loc
    // z_recv_loc[i] = glo2loc[z_recv_glo[i]], for i=0,...,z_precv.at(nzcomm)-1
    thrust::copy(thrust::make_permutation_iterator(glo2loc_ptr, z_recv_glo.begin()),
		 thrust::make_permutation_iterator(glo2loc_ptr, z_recv_glo.end()),
		 z_recv_loc.begin());
    for (int i=0;i < nz_comm;i++) {
      // Map:
      // z_send_loc = glo2loc[z_send_glo]
      thrust::copy(thrust::make_permutation_iterator(glo2loc_ptr, z_send_glo.begin()+z_psend.at(i)),
		   thrust::make_permutation_iterator(glo2loc_ptr, z_send_glo.begin()+z_psend.at(i+1)),
		   z_send_loc.begin()+z_psend.at(i));
    }

  }
}

//
// Communicate forces
//
void CudaDomdecD2DComm::comm_force(Force<long long int>& force) {

  const int DATA_TAG = 2;

  // Double pointers to (recvbuf, sendbuf) to make address math easier
  double* precvbuf = (double *)recvbuf;
  double* psendbuf = (double *)sendbuf;
  double* h_precvbuf = (double *)h_recvbuf;
  double* h_psendbuf = (double *)h_sendbuf;

  // Pack forces in z-direction: force => recvbuf
  for (int i=0;i < nz_comm;i++) {
    // Pack forces to recvbuf[]
    packXYZ((double *)force.x(), (double *)force.y(), (double *)force.z(),
	    thrust::raw_pointer_cast(z_recv_loc.data())+z_precv.at(i), z_nrecv.at(i),
	    &precvbuf[3*z_precv.at(i)]);
  }
  // Wait until packing is done
  cudaCheck(cudaDeviceSynchronize());

  // Send & Recv data in z-direction
  for (int i=0;i < nz_comm;i++) {
    if (z_nsend.at(i) > 0 && z_nrecv.at(i) > 0) {
      MPICheck(cudaMPI.Sendrecv(&precvbuf[3*z_precv.at(i)], 3*z_nrecv.at(i)*sizeof(double),
				z_recv_node.at(i), DATA_TAG,
				&psendbuf[3*z_psend.at(i)], 3*z_nsend.at(i)*sizeof(double),
				z_send_node.at(i), DATA_TAG, MPI_STATUS_IGNORE,
				&h_precvbuf[3*z_precv.at(i)], &h_psendbuf[3*z_psend.at(i)]));      
    } else if (z_nrecv.at(i) > 0) {
      MPICheck(cudaMPI.Send(&precvbuf[3*z_precv.at(i)], 3*z_nrecv.at(i)*sizeof(double),
			    z_recv_node.at(i), DATA_TAG, &h_precvbuf[3*z_precv.at(i)]));
    } else if (z_nsend.at(i) > 0) {
      MPICheck(cudaMPI.Recv(&psendbuf[3*z_psend.at(i)], 3*z_nsend.at(i)*sizeof(double),
			    z_send_node.at(i), DATA_TAG, MPI_STATUS_IGNORE,
			    &h_psendbuf[3*z_psend.at(i)]));
    }
  }

  // Unpack force in z-direction: sendbuf => force
  for (int i=0;i < nz_comm;i++) {
    // format = X[z_nsend.at(i) x double] | Y[z_nsend.at(i) x double] | Z[z_nsend.at(i) x double]
    thrust::device_ptr<double> x_src_ptr(&psendbuf[3*z_psend.at(i)]);
    thrust::device_ptr<double> x_dst_ptr((double *)force.x());

    thrust::device_ptr<double> y_src_ptr(&psendbuf[3*z_psend.at(i)+z_nsend.at(i)]);
    thrust::device_ptr<double> y_dst_ptr((double *)force.y());

    thrust::device_ptr<double> z_src_ptr(&psendbuf[3*z_psend.at(i)+2*z_nsend.at(i)]);
    thrust::device_ptr<double> z_dst_ptr((double *)force.z());

    // x_dst_ptr[z_send_loc[i]] += x_src_ptr[i], i=0,...,z_nsend[i]-1
    thrust::transform(x_src_ptr, x_src_ptr+z_nsend.at(i),
		      thrust::make_permutation_iterator(x_dst_ptr, z_send_loc.begin()+z_psend.at(i)),
		      thrust::make_permutation_iterator(x_dst_ptr, z_send_loc.begin()+z_psend.at(i)),
		      thrust::plus<double>());

    // y_dst_ptr[z_send_loc[i]] += y_src_ptr[i], i=0,...,z_nsend[i]-1
    thrust::transform(y_src_ptr, y_src_ptr+z_nsend.at(i),
		      thrust::make_permutation_iterator(y_dst_ptr, z_send_loc.begin()+z_psend.at(i)),
		      thrust::make_permutation_iterator(y_dst_ptr, z_send_loc.begin()+z_psend.at(i)),
		      thrust::plus<double>());

    // x_dst_ptr[z_send_loc[i]] += x_src_ptr[i], i=0,...,z_nsend[i]-1
    thrust::transform(z_src_ptr, z_src_ptr+z_nsend.at(i),
		      thrust::make_permutation_iterator(z_dst_ptr, z_send_loc.begin()+z_psend.at(i)),
		      thrust::make_permutation_iterator(z_dst_ptr, z_send_loc.begin()+z_psend.at(i)),
		      thrust::plus<double>());

  }
  // Wait until unpacking is done
  cudaCheck(cudaDeviceSynchronize());

}

void CudaDomdecD2DComm::computeByteNumPos(const int num_comm, std::vector<int>& ncomm,
					  std::vector<int>& ncommByte, std::vector<int>& pcommByte,
					  const bool update) {
  ncommByte.resize(num_comm);
  pcommByte.resize(num_comm+1);
  pcommByte.at(0) = 0;
  if (update) {
    for (int i=0;i < num_comm;i++) {
      ncommByte.at(i) = alignInt(ncomm.at(i),2)*sizeof(int) + ncomm.at(i)*3*sizeof(double);
      pcommByte.at(i+1) = pcommByte.at(i) + ncommByte.at(i);
    }
  } else {
    for (int i=0;i < num_comm;i++) {
      ncommByte.at(i) = ncomm.at(i)*3*sizeof(double);
      pcommByte.at(i+1) = pcommByte.at(i) + ncommByte.at(i);
    }
  }
}
