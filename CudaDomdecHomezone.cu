#undef SEEK_SET
#undef SEEK_CUR
#undef SEEK_END
#include <mpi.h>
#include <cassert>
#include "CudaDomdecHomezone.h"
#include "CudaMPI.h"
#include "mpi_utils.h"

//
// Update homezone atomlist. Simple version using atomicAdd (does this have to be faster?)
//
__global__ void fill_neighsend_kernel(const int ncoord, const int stride,
				      const double* __restrict__ coord,
				      const double inv_boxx, const double inv_boxy, const double inv_boxz,
				      const int nx, const int ny, const int nz, const int nneigh,
				      const int homeix, const int homeiy, const int homeiz,
				      int* __restrict__ num_neighsend,
				      int* __restrict__ destind) {
  // Shared memory
  // Requires: 27*sizeof(int)
  __shared__ int sh_num_neighsend[27];

  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  int nxt = min(3, nx);
  int nxyt = nxt*min(3, ny);
  if (i < nneigh) {
    sh_num_neighsend[i] = 0;
  }
  __syncthreads();
  if (i < ncoord) {
    double x = coord[i]*inv_boxx           + 0.5;
    double y = coord[i+stride]*inv_boxy    + 0.5;
    double z = coord[i+stride*2]*inv_boxz  + 0.5;
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    int ix = (int)(x*(double)nx);
    int iy = (int)(y*(double)ny);
    int iz = (int)(z*(double)nz);
    int dix = ix - homeix;
    int diy = iy - homeiy;
    int diz = iz - homeiz;
    // (1 < dix) - (dix < -1) returns: 1 if dix > 1
    //                                -1 if dix < -1
    //                                 0 otherwise
    dix -= nx*((1 < dix) - (dix < -1));
    diy -= ny*((1 < diy) - (diy < -1));
    diz -= nz*((1 < diz) - (diz < -1));
    // After these, dix = {0, .., 2} or {0, .., nx-1}
    dix = (dix+1) % nx;
    diy = (diy+1) % ny;
    diz = (diz+1) % nz;
    // Get neighboring node index, ind = 0...nneigh-1
    int ind = dix + diy*nxt + diz*nxyt;
    atomicAdd(&sh_num_neighsend[ind], 1);
    destind[i] = ind;
  }

  __syncthreads();
  if (i < nneigh) {
    atomicAdd(&num_neighsend[i], sh_num_neighsend[i]);
  }
}

//
// Calculates pos_neighsend[0...nneigh] using exclusive cumulative sum
// Launched with blockDim.x = 32 and nblock = 1
//
__global__ void calc_pos_neighsend_kernel(const int nneigh,
					  const int* __restrict__ num_neighsend,
					  int* __restrict__ pos_neighsend) {
  // Shared memory
  // Requires: 27*sizeof(int)
  __shared__ int sh_pos_neighsend[27];
  // Calculate inclusive scan and then shift to make it exclusive scan
  // Calculate positions into sh_pos_neighsend
  if (threadIdx.x < nneigh) sh_pos_neighsend[threadIdx.x] = num_neighsend[threadIdx.x];
  for (int d=1;d < nneigh;d *= 2) {
    int t = threadIdx.x + d;
    int val = (t < nneigh) ? sh_pos_neighsend[t] : 0;
    __syncthreads();
    sh_pos_neighsend[threadIdx.x] += val;
    __syncthreads();
  }
  if (threadIdx.x < nneigh) {
    // Shift & store result to get exclusive cumulative sum in global memory
    if (threadIdx.x == 0) pos_neighsend[0] = 0;
    pos_neighsend[threadIdx.x+1] = sh_pos_neighsend[threadIdx.x];
  }
}

//
// Packs neighsend -buffer
//
__global__ void pack_neighsend_kernel(const int ncoord, const int stride,
				      const double* __restrict__ coord,
				      const double* __restrict__ coord2,
				      const int* __restrict__ destind,
				      const int* __restrict__ loc2glo,
				      int* __restrict__ pos_neighsend,
				      CudaDomdecHomezone::neighcomm_t* __restrict__ neighsend) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < ncoord) {
    int ind = destind[i];
    int pos = atomicAdd(&pos_neighsend[ind], 1);
    neighsend[pos].gloind = loc2glo[i];
    neighsend[pos].x1 = coord[i];
    neighsend[pos].y1 = coord[i+stride];
    neighsend[pos].z1 = coord[i+stride*2];
    neighsend[pos].x2 = coord2[i];
    neighsend[pos].y2 = coord2[i+stride];
    neighsend[pos].z2 = coord2[i+stride*2];
  }
}

//
// Unpacks received data
//
__global__ void unpack_neighrecv_kernel(const int stride, const int nneigh,
					const int* __restrict__ pos_neighrecv,
					const CudaDomdecHomezone::neighcomm_t* __restrict__ neighrecv,
					double* __restrict__ coord,
					double* __restrict__ coord2,
					int* __restrict__ loc2glo) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  const int tot_neighrecv = pos_neighrecv[nneigh];
  if (i < tot_neighrecv) {
    loc2glo[i]         = neighrecv[i].gloind;
    coord[i]           = neighrecv[i].x1;
    coord[i+stride]    = neighrecv[i].y1;
    coord[i+stride*2]  = neighrecv[i].z1;
    coord2[i]          = neighrecv[i].x2;
    coord2[i+stride]   = neighrecv[i].y2;
    coord2[i+stride*2] = neighrecv[i].z2;
  }
}

//################################################################################
//################################################################################
//################################################################################

//
// Class creator
//
CudaDomdecHomezone::CudaDomdecHomezone(Domdec& domdec, CudaMPI& cudaMPI) : 
  domdec(domdec), cudaMPI(cudaMPI) {

  int nxt = min(3, domdec.get_nx());
  int nyt = min(3, domdec.get_ny());
  int nzt = min(3, domdec.get_nz());
  nneigh = nxt*nyt*nzt;

  allocate<int>(&num_neighind, nneigh);
  allocate<int>(&pos_neighind, nneigh+1);
  allocate_host<int>(&h_pos_neighind, nneigh+1);

  destind_len = 0;
  destind = NULL;

  neighsend_len = 0;
  neighsend = NULL;
  h_neighsend_len = 0;
  h_neighsend = NULL;

  neighrecv_len = 0;
  neighrecv = NULL;

  h_neighrecv_len = 0;
  h_neighrecv = NULL;

  neighnode = new int[nneigh];
  send_request = new MPI_Request[nneigh];

  int ixl = -(nxt-1)/2;
  int ixh = ixl + nxt-1;

  int iyl = -(nyt-1)/2;
  int iyh = iyl + nyt-1;

  int izl = -(nzt-1)/2;
  int izh = izl + nzt-1;

  int k = 0;
  for (int ix=ixl;ix <= ixh;ix++) {
    for (int iy=iyl;iy <= iyh;iy++) {
      for (int iz=izl;iz <= izh;iz++) {
	neighnode[k++] = domdec.get_nodeind_pbc(ix + domdec.get_homeix(),
						iy + domdec.get_homeiy(),
						iz + domdec.get_homeiz());
      }
    }
  }

  if (nneigh != k) {
    std::cout << "CudaDomdecHomezone::CudaDomdecHomezone, error in nneigh" << std::endl;
    exit(1);
  }

}

//
// Class destructor
//
CudaDomdecHomezone::~CudaDomdecHomezone() {
  deallocate<int>(&num_neighind);
  deallocate<int>(&pos_neighind);
  deallocate_host<int>(&h_pos_neighind);
  delete [] neighnode;
  delete [] send_request;
  if (destind != NULL) deallocate<int>(&destind);
  if (neighsend != NULL) deallocate<neighcomm_t>(&neighsend);
  if (neighrecv != NULL) deallocate<neighcomm_t>(&neighrecv);
  if (h_neighsend != NULL) deallocate_host<neighcomm_t>(&h_neighsend);
  if (h_neighrecv != NULL) deallocate_host<neighcomm_t>(&h_neighrecv);
}

//
// Build Homezone, assigns coordinates into sub-boxes.
// Returns the number of coordinates in the homezone
// NOTE: Done on the CPU since this is only done once at the beginning of the simulation
//
int CudaDomdecHomezone::build(hostXYZ<double>& coord) {

  int stride = coord.stride;
  double inv_boxx = domdec.get_inv_boxx();
  double inv_boxy = domdec.get_inv_boxy();
  double inv_boxz = domdec.get_inv_boxz();
  int nx = domdec.get_nx();
  int ny = domdec.get_ny();
  int nz = domdec.get_nz();
  int homeix = domdec.get_homeix();
  int homeiy = domdec.get_homeiy();
  int homeiz = domdec.get_homeiz();

  // Approximate how many coordinates we might end up with
  int h_loc2glo_len = 2*coord.n/(nx*ny*nz);
  h_loc2glo_len = (h_loc2glo_len < coord.n) ? h_loc2glo_len : coord.n;
  int *h_loc2glo = new int[h_loc2glo_len];

  // Find coordinates that are in this sub-box
  int nloc = 0;
  for (int i=0;i < coord.n;i++) {
    double x = coord.data[i]*inv_boxx          + 0.5;
    double y = coord.data[i+stride]*inv_boxy   + 0.5;
    double z = coord.data[i+stride*2]*inv_boxz + 0.5;    
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);
    int ix = (int)(x*(double)nx);
    int iy = (int)(y*(double)ny);
    int iz = (int)(z*(double)nz);
    if (ix == homeix && iy == homeiy && iz == homeiz) {
      h_loc2glo[nloc++] = i;
    }
  }

  loc2glo.reserve(nloc);

  copy_HtoD<int>(h_loc2glo, get_loc2glo_ptr(), nloc);
  delete [] h_loc2glo;

  return nloc;
}

//
// Update Homezone
// Returns the number of coordinates in the homezone
//
int CudaDomdecHomezone::update(cudaXYZ<double> *coord, cudaXYZ<double> *coord2, cudaStream_t stream) {

  assert(coord->n == coord2->n);
  assert(coord->stride == coord2->stride);

  reallocate<int>(&destind, &destind_len, coord->n, 1.2f);
  reallocate<neighcomm_t>(&neighsend, &neighsend_len, coord->n, 1.2f);
  if (!cudaMPI.isCudaAware()) {
    reallocate_host<neighcomm_t>(&h_neighsend, &h_neighsend_len, coord->n, 1.2f);
  }

  int nthread = 1024;
  int nblock = (coord->n - 1)/nthread + 1;

  // Assign coordinates into neighboring, or home, sub-boxes
  fill_neighsend_kernel<<< nblock, nthread, 0, stream >>>
    (coord->n, coord->stride, coord->data,
     domdec.get_inv_boxx(), domdec.get_inv_boxy(), domdec.get_inv_boxz(),
     domdec.get_nx(), domdec.get_ny(), domdec.get_nz(), nneigh,
     domdec.get_homeix(), domdec.get_homeiy(), domdec.get_homeiz(),
     num_neighind, destind);
  cudaCheck(cudaGetLastError());

  // Calculate positions for send buffer
  calc_pos_neighsend_kernel<<< 1, 32, 0, stream >>>(nneigh, num_neighind, pos_neighind);
  cudaCheck(cudaGetLastError());

  // Copy pos_neighind[nneigh+1] to host
  copy_DtoH<int>(pos_neighind, h_pos_neighind, nneigh+1, stream);

  // Pack coordinate data into send buffer
  pack_neighsend_kernel<<< nblock, nthread, 0, stream >>>
    (coord->n, coord->stride, coord->data, coord2->data, destind, get_loc2glo_ptr(),
     pos_neighind, neighsend);
  cudaCheck(cudaGetLastError());

  // Send coordinate data
  const int TAG = 1;
  for (int i=0;i < nneigh;i++) {
    if (neighnode[i] != domdec.get_mynode()) {
      int start = (i > 0) ? h_pos_neighind[i-1] : 0;
      int num_send = h_pos_neighind[i] - start;
      MPICheck(cudaMPI.Isend(&neighsend[start], num_send*sizeof(neighcomm_t), 
			     neighnode[i], TAG, &send_request[i],
			     &h_neighsend[start]));
    }
  }

  // Count how much data we are receiving
  int count_tot = 0;
  MPI_Status status[27];
  int count[27];
  for (int i=0;i < nneigh;i++) {
    if (neighnode[i] != domdec.get_mynode()) {
      // Neighboring sub-boxes
      MPICheck(MPI_Probe(neighnode[i], TAG, cudaMPI.get_comm(), &status[i]));
      MPICheck(MPI_Get_count(&status[i], MPI_BYTE, &count[i]));
      count[i] /= sizeof(neighcomm_t);
    } else {
      // Local (home) sub-box
      int start = (i > 0) ? h_pos_neighind[i-1] : 0;
      count[i] = h_pos_neighind[i] - start;
    }
    count_tot += count[i];
  }

  // Re-allocate memory as needed
  reallocate<neighcomm_t>(&neighrecv, &neighrecv_len, count_tot, 1.2f);
  loc2glo.reserve(count_tot);
  if (!cudaMPI.isCudaAware()) {
    reallocate_host<neighcomm_t>(&h_neighrecv, &h_neighrecv_len, count_tot, 1.2f);
  }

  // Receive data
  int pos = 0;
  for (int i=0;i < nneigh;i++) {
    if (neighnode[i] != domdec.get_mynode()) {
      // Receive data from neighboring sub-box
      MPICheck(cudaMPI.Recv(&neighrecv[pos], count[i]*sizeof(neighcomm_t), neighnode[i],
			    TAG, &status[i], h_neighrecv));
    } else {
      // Copy data from local (home) sub-box
      int lpos = (i > 0) ? h_pos_neighind[i-1] : 0;
      copy_DtoD<neighcomm_t>(&neighrecv[pos], &neighsend[lpos], count[i], stream);
    }
    pos += count[i];
  }

  // Unpack data on GPU
  unpack_neighrecv_kernel<<< nblock, nthread, 0, stream >>>
    (coord->stride, nneigh, pos_neighind, neighrecv, coord->data, coord2->data, get_loc2glo_ptr());
  cudaCheck(cudaGetLastError());

  return count_tot;
}
