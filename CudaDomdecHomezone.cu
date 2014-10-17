#include <cassert>
#include <algorithm>
#include "CudaDomdecHomezone.h"
#include "CudaMPI.h"
#include "mpi_utils.h"
#include "gpu_utils.h"

//
// Returns index from (dix, diy, diz)
//
//  int nxt = min(3, nx);
//  int nxyt = nxt*min(3, ny);
__host__ __device__ inline int dix2ind(int dix, int diy, int diz,
				       const int nx, const int ny, const int nz,
				       const int nxt, const int nxyt) {
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
  return dix + diy*nxt + diz*nxyt;
}


//
// Update homezone atomlist. Simple version using atomicAdd (does this have to be faster?)
//
__global__ void fill_send_kernel(const int ncoord,
				 const double* __restrict__ xin,
				 const double* __restrict__ yin,
				 const double* __restrict__ zin,
				 const double inv_boxx, const double inv_boxy, const double inv_boxz,
				 const double* __restrict__ lohi_buf,
				 const int nx, const int ny, const int nz, const int nneigh,
				 int* __restrict__ num_send,
				 int* __restrict__ destind) {
  // Shared memory
  // Requires: 2*3*sizeof(double) + 2*9*sizeof(double) + 2*27*sizeof(double) + nneigh*sizeof(int)
  extern __shared__ double sh_buf[];
  double *sh_lox = &sh_buf[0];      // In total this is 2*3+2*9+2*27=78 doubles
  double *sh_hix = &sh_buf[3];
  double *sh_loy = &sh_buf[6];
  double *sh_hiy = &sh_buf[15];
  double *sh_loz = &sh_buf[24];
  double *sh_hiz = &sh_buf[51];
  int* sh_num_send = (int *)&sh_buf[78];

  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  int nxt = min(3, nx);
  int nxyt = nxt*min(3, ny);
  if (threadIdx.x < 78) {
    sh_buf[threadIdx.x] = lohi_buf[threadIdx.x];
  }
  const int warpstart = ((78-1)/warpsize+1)*warpsize;
  if (threadIdx.x >= warpstart && threadIdx.x-warpstart < nneigh) {
    sh_num_send[threadIdx.x-warpstart] = 0;
  }
  __syncthreads();
  bool error = false;
  if (i < ncoord) {
    double x = xin[i]*inv_boxx + 0.5;
    double y = yin[i]*inv_boxy + 0.5;
    double z = zin[i]*inv_boxz + 0.5;
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    int dix=3;
    if (x >= sh_lox[0] && x < sh_hix[0]) dix = 0;
    if (x >= sh_lox[1] && x < sh_hix[1]) dix = 1;
    if (x >= sh_lox[2] && x < sh_hix[2]) dix = 2;
    if (dix == 3) error = true;
    dix = (dix % 3);   // simple error recovery

    double *sh_loyp = &sh_loy[dix*3];
    double *sh_hiyp = &sh_hiy[dix*3];
    int diy=3;
    if (y >= sh_loyp[0] && y < sh_hiyp[0]) diy = 0;
    if (y >= sh_loyp[1] && y < sh_hiyp[1]) diy = 1;
    if (y >= sh_loyp[2] && y < sh_hiyp[2]) diy = 2;
    if (diy == 3) error = true;
    diy = (diy % 3);

    double *sh_lozp = &sh_loz[dix*9 + diy*3];
    double *sh_hizp = &sh_hiz[dix*9 + diy*3];
    int diz=3;
    if (z >= sh_lozp[0] && z < sh_hizp[0]) diz = 0;
    if (z >= sh_lozp[1] && z < sh_hizp[1]) diz = 1;
    if (z >= sh_lozp[2] && z < sh_hizp[2]) diz = 2;
    if (diz == 3) error = true;
    diz = (diz % 3);

    // Transform into (-1,0,1)
    dix--;
    diy--;
    diz--;

    int ind = dix2ind(dix, diy, diz, nx, ny, nz, nxt, nxyt);

    atomicAdd(&sh_num_send[ind], 1);
    destind[i] = ind;
  }

  __syncthreads();
  if (threadIdx.x < nneigh) {
    atomicAdd(&num_send[threadIdx.x], sh_num_send[threadIdx.x]);
  }

  // Set error flag into num_send[nneigh]
  if (error) {
    num_send[nneigh] = 1;
  }
}

//
// Calculates pos_send[0...nneigh] using exclusive cumulative sum
// Launched with blockDim.x = 32 and nblock = 1
//
__global__ void calc_pos_send_kernel(const int nneigh,
				     const int* __restrict__ num_send,
				     int* __restrict__ pos_send) {

  // This kernel is so simple, we'll just loop. No fancy stuff here.
  if (threadIdx.x == 0) {
    pos_send[0] = 0;
    for (int i=0;i < nneigh;i++) pos_send[i+1] = pos_send[i] + num_send[i];
  }

  /*
  // Shared memory
  // Requires: 27*sizeof(int)
  __shared__ int sh_pos_send[27];
  // Calculate inclusive scan and then shift to make it exclusive scan
  // Calculate positions into sh_pos_send
  if (threadIdx.x < nneigh) sh_pos_send[threadIdx.x] = num_send[threadIdx.x];
  if (threadIdx.x == 0) printf("num_send = %d %d\n",num_send[0],num_send[1]);
  __syncthreads();
  for (int d=1;d < nneigh;d *= 2) {
    int t = threadIdx.x + d;
    int val = (t < nneigh) ? sh_pos_send[t] : 0;
    __syncthreads();
    if (threadIdx.x < nneigh) sh_pos_send[threadIdx.x] += val;
    __syncthreads();
  }
  if (threadIdx.x < nneigh) {
    // Shift & store result to get exclusive cumulative sum in global memory
    if (threadIdx.x == 0) pos_send[0] = 0;
    pos_send[threadIdx.x+1] = sh_pos_send[threadIdx.x];
  }
  */
}

//
// Packs send -buffer
//
__global__ void pack_send_kernel(const int ncoord,
				 const double* __restrict__ x1,
				 const double* __restrict__ y1,
				 const double* __restrict__ z1,
				 const double* __restrict__ x2,
				 const double* __restrict__ y2,
				 const double* __restrict__ z2,
				 const int* __restrict__ destind,
				 const int* __restrict__ loc2glo,
				 int* __restrict__ pos_send,
				 CudaDomdecHomezone::neighcomm_t* __restrict__ send) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < ncoord) {
    int ind = destind[i];
    int pos = atomicAdd(&pos_send[ind], 1);
    send[pos].gloind = loc2glo[i];
    send[pos].x1 = x1[i];
    send[pos].y1 = y1[i];
    send[pos].z1 = z1[i];
    send[pos].x2 = x2[i];
    send[pos].y2 = y2[i];
    send[pos].z2 = z2[i];
  }
}

//
// Unpacks received data
//
__global__ void unpack_recv_kernel(const int num_recv_tot,
				   const CudaDomdecHomezone::neighcomm_t* __restrict__ recv,
				   double* __restrict__ x1,
				   double* __restrict__ y1,
				   double* __restrict__ z1,
				   double* __restrict__ x2,
				   double* __restrict__ y2,
				   double* __restrict__ z2,
				   int* __restrict__ loc2glo) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < num_recv_tot) {
    loc2glo[i]= recv[i].gloind;
    x1[i]     = recv[i].x1;
    y1[i]     = recv[i].y1;
    z1[i]     = recv[i].z1;
    x2[i]     = recv[i].x2;
    y2[i]     = recv[i].y2;
    z2[i]     = recv[i].z2;
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

  allocate<int>(&num_send, nneigh+1);
  allocate<int>(&pos_send, nneigh+1);
  allocate_host<int>(&h_num_send, nneigh+1);

  allocate<double>(&lohi_buf, 78);
  allocate_host<double>(&h_lohi_buf, 78);

  destind_len = 0;
  destind = NULL;

  send_len = 0;
  send = NULL;
  h_send_len = 0;
  h_send = NULL;

  recv_len = 0;
  recv = NULL;

  h_recv_len = 0;
  h_recv = NULL;

  neighnode.resize(nneigh);
  request.resize((nneigh-1)*2);

  h_pos_send = new int[nneigh+1];
  num_recv.resize(nneigh);
  pos_recv.resize(nneigh+1);

  /*
  int ixl = -(nxt-1)/2;
  int ixh = ixl + nxt-1;

  int iyl = -(nyt-1)/2;
  int iyh = iyl + nyt-1;

  int izl = -(nzt-1)/2;
  int izh = izl + nzt-1;
  */

  std::fill(neighnode.begin(), neighnode.end(), -1);
  for (int diz=-1;diz <= 1;diz++) {
    for (int diy=-1;diy <= 1;diy++) {
      for (int dix=-1;dix <= 1;dix++) {
	int k = dix2ind(dix, diy, diz,
			domdec.get_nx(), domdec.get_ny(), domdec.get_nz(),
			nxt, nxt*nyt);
	neighnode.at(k) = domdec.get_nodeind_pbc(dix + domdec.get_homeix(),
						 diy + domdec.get_homeiy(),
						 diz + domdec.get_homeiz());
      }
    }
  }

  imynode = dix2ind(0, 0, 0, domdec.get_nx(), domdec.get_ny(), domdec.get_nz(),
		    nxt, nxt*nyt);

  if (neighnode.at(imynode) != domdec.get_mynode()) {
    std::cout << "CudaDomdecHomezone::CudaDomdecHomezone, error in setting neighnode(1)" << std::endl;
    exit(1);
  }

  for (int i=0;i < nneigh;i++) {
    if (neighnode.at(i) == -1) {
      std::cout << "CudaDomdecHomezone::CudaDomdecHomezone, error in setting neighnode(2)"
		<< std::endl;
      exit(1);
    }
  }

  //
  // Neighbor node communication order. This is setup such that the self node comes first and the
  // rest of the nodes are according to their MPI rank
  //
  // Get an ordered array of neighbor nodes without the self node
  std::vector<int> neighnode_sort(neighnode);
  neighnode_sort.erase(neighnode_sort.begin() + imynode);
  std::sort(neighnode_sort.begin(), neighnode_sort.end());
  // Place the self node at front, and the rest of the neighbors in sorted order after it
  comm_order.resize(nneigh);
  comm_order.at(0) = imynode;
  for (int i=1;i < nneigh;i++) {
    // MPI rank we're looking for
    int rank = neighnode_sort.at(i-1);
    comm_order.at(i) = (int)(std::find(neighnode.begin(), neighnode.end(), rank) - neighnode.begin());
  }

}

//
// Class destructor
//
CudaDomdecHomezone::~CudaDomdecHomezone() {
  deallocate<int>(&num_send);
  deallocate<int>(&pos_send);
  deallocate_host<int>(&h_num_send);
  deallocate<double>(&lohi_buf);
  deallocate_host<double>(&h_lohi_buf);
  delete [] h_pos_send;
  if (destind != NULL) deallocate<int>(&destind);
  if (send != NULL) deallocate<neighcomm_t>(&send);
  if (recv != NULL) deallocate<neighcomm_t>(&recv);
  if (h_send != NULL) deallocate_host<neighcomm_t>(&h_send);
  if (h_recv != NULL) deallocate_host<neighcomm_t>(&h_recv);
}

//
// Build Homezone, assigns coordinates into sub-boxes. Done on the CPU
// Creates new loc2glo, DOES NOT re-create h_coord according to the new loc2glo
// Returns the number of coordinates in the homezone
// NOTE: h_coord is the global array
//
int CudaDomdecHomezone::build(hostXYZ<double>& h_coord) {

  int nx = domdec.get_nx();
  int ny = domdec.get_ny();
  int nz = domdec.get_nz();
  int homeix = domdec.get_homeix();
  int homeiy = domdec.get_homeiy();
  int homeiz = domdec.get_homeiz();
  double inv_boxx = domdec.get_inv_boxx();
  double inv_boxy = domdec.get_inv_boxy();
  double inv_boxz = domdec.get_inv_boxz();

  int *h_loc2glo = new int[h_coord.size()];

  double lox = domdec.get_lo_bx();
  double hix = domdec.get_hi_bx();
  double loy = domdec.get_lo_by();
  double hiy = domdec.get_hi_by();
  double loz = domdec.get_lo_bz();
  double hiz = domdec.get_hi_bz();

  // Find coordinates that are in this sub-box
  int nloc = 0;
  for (int i=0;i < h_coord.size();i++) {
    double x = h_coord.x()[i]*inv_boxx + 0.5;
    double y = h_coord.y()[i]*inv_boxy + 0.5;
    double z = h_coord.z()[i]*inv_boxz + 0.5;    
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);
    if (x >= lox && x < hix && y >= loy && y < hiy && z >= loz && z < hiz) {
      h_loc2glo[nloc++] = i;
    }
  }

  loc2glo.resize(nloc);

  copy_HtoD_sync<int>(h_loc2glo, get_loc2glo_ptr(), nloc);
  delete [] h_loc2glo;

  return nloc;
}

//
// Update Homezone
// creates new loc2glo, re-creates coord and coord2 accoring to the new loc2glo
// Returns: the number of coordinates in the homezone
//
int CudaDomdecHomezone::update(const int ncoord, cudaXYZ<double>& coord, cudaXYZ<double>& coord2,
			       cudaStream_t stream) {
  assert(ncoord <= coord.size());
  assert(ncoord <= coord2.size());

  // Allocate to #coordinates to avoid busting the buffer limits
  reallocate<int>(&destind, &destind_len, ncoord, 1.2f);
  reallocate<neighcomm_t>(&send, &send_len, ncoord, 1.2f);

  clear_gpu_array<int>(num_send, nneigh+1, stream);

  // Get a copy of lohi_buf and copy it to GPU
  domdec.copy_lohi_buf(h_lohi_buf);
  copy_HtoD<double>(h_lohi_buf, lohi_buf, 78, stream);

  int nthread = 1024;
  int nblock = (ncoord - 1)/nthread + 1;
  int shmem = 2*3*sizeof(double) + 2*9*sizeof(double) + 2*27*sizeof(double) + nneigh*sizeof(int);

  // Assign coordinates into neighboring, or home, sub-boxes
  fill_send_kernel<<< nblock, nthread, shmem, stream >>>
    (ncoord, coord.x(), coord.y(), coord.z(),     
     domdec.get_inv_boxx(), domdec.get_inv_boxy(), domdec.get_inv_boxz(),
     lohi_buf,
     domdec.get_nx(), domdec.get_ny(), domdec.get_nz(), nneigh,
     num_send, destind);
  cudaCheck(cudaGetLastError());

  // Copy num_send => h_num_send
  copy_DtoH<int>(num_send, h_num_send, nneigh+1, stream);

  // Calculate positions for send buffer
  calc_pos_send_kernel<<< 1, 32, 0, stream >>>(nneigh, num_send, pos_send);
  cudaCheck(cudaGetLastError());

  // Pack coordinate data into send buffer
  pack_send_kernel<<< nblock, nthread, 0, stream >>>
    (ncoord, coord.x(), coord.y(), coord.z(), coord2.x(), coord2.y(), coord2.z(),
     destind, get_loc2glo_ptr(), pos_send, send);
  cudaCheck(cudaGetLastError());

  // Wait here for the stream to finish
  cudaCheck(cudaStreamSynchronize(stream));

  // Check for error flag
  if (h_num_send[nneigh] != 0) {
    std::cerr << "CudaDomdecHomezone::update, atom(s) moved more than a single box length"
	      << std::endl;
    exit(1);
  }

  // Compute positions h_pos_send from h_num_send
  // NOTE: h_pos_send[0] = 0 and h_pos_send[nneigh] = total number to send
  h_pos_send[0] = 0;
  for (int i=0;i < nneigh;i++) h_pos_send[i+1] = h_pos_send[i] + h_num_send[i];

  // Total number of coordinates to send is h_pos_send[nneigh]
  if (!cudaMPI.isCudaAware()) {
    reallocate_host<neighcomm_t>(&h_send, &h_send_len, h_pos_send[nneigh], 1.4f);
  }

  const int COUNT_TAG = 1;
  int nrequest = 0;
  // Send number of coordinates
  for (int i=0;i < nneigh;i++) {
    if (neighnode.at(i) != domdec.get_mynode()) {
      MPICheck(MPI_Isend(&h_num_send[i], 1, MPI_INT, neighnode.at(i), COUNT_TAG,
			 cudaMPI.get_comm(), &request.at(nrequest)));
      nrequest++;
    }
  }  

  // Receive number of coordinates
  for (int i=0;i < nneigh;i++) {
    if (neighnode.at(i) != domdec.get_mynode()) {
      MPICheck(MPI_Irecv(&num_recv.at(i), 1, MPI_INT, neighnode.at(i), COUNT_TAG,
			 cudaMPI.get_comm(), &request.at(nrequest)));
      nrequest++;
    } else {
      num_recv.at(i) = h_num_send[i];
    }
  }

  // Wait for communication to finish
  MPICheck(MPI_Waitall(nrequest, request.data(), MPI_STATUSES_IGNORE));

  pos_recv.at(0) = 0;
  for (int i=0;i < nneigh;i++) pos_recv.at(i+1) = pos_recv.at(i) + num_recv.at(i);
  int num_recv_tot = pos_recv.at(nneigh);

  // Re-allocate memory as needed
  reallocate<neighcomm_t>(&recv, &recv_len, num_recv_tot, 1.2f);
  loc2glo.resize(num_recv_tot);
  if (!cudaMPI.isCudaAware()) {
    reallocate_host<neighcomm_t>(&h_recv, &h_recv_len, num_recv_tot, 1.2f);
  }

  const int COORD_TAG = 1;

  // Send & Recv coordinate data
  for (int j=0;j < nneigh;j++) {
    int i = comm_order.at(j);
    if (neighnode.at(i) != domdec.get_mynode()) {
      if (h_num_send[i] > 0 && num_recv.at(i) > 0) {
	MPICheck(cudaMPI.Sendrecv(&send[h_pos_send[i]], h_num_send[i]*sizeof(neighcomm_t), 
				  neighnode.at(i), COORD_TAG,
				  &recv[pos_recv.at(i)], num_recv.at(i)*sizeof(neighcomm_t),
				  neighnode.at(i), COORD_TAG, MPI_STATUS_IGNORE,
				  &h_send[h_pos_send[i]], &h_recv[pos_recv.at(i)]));
      } else if (h_num_send[i] > 0) {
	MPICheck(cudaMPI.Send(&send[h_pos_send[i]], h_num_send[i]*sizeof(neighcomm_t), 
			      neighnode.at(i), COORD_TAG,
			      &h_send[h_pos_send[i]]));
      } else if (num_recv.at(i) > 0) {
	MPICheck(cudaMPI.Recv(&recv[pos_recv.at(i)], num_recv.at(i)*sizeof(neighcomm_t),
			      neighnode.at(i), COORD_TAG, MPI_STATUS_IGNORE,
			      &h_recv[pos_recv.at(i)]));
      }
    } else if (num_recv.at(i) > 0) {
      // Copy data from local (home) sub-box
      copy_DtoD<neighcomm_t>(&send[h_pos_send[i]], &recv[pos_recv.at(i)], num_recv.at(i), stream);
    }
  }
  // Wait until the copy_DtoD above finishes
  cudaCheck(cudaStreamSynchronize(stream));

  // Re-allocate coord and coord2 if needed
  coord.realloc(num_recv_tot);
  coord2.realloc(num_recv_tot);

  // Unpack data on GPU
  unpack_recv_kernel<<< nblock, nthread, 0, stream >>>
    (num_recv_tot, recv, coord.x(), coord.y(), coord.z(), coord2.x(), coord2.y(), coord2.z(),
     get_loc2glo_ptr());
  cudaCheck(cudaGetLastError());

  // Wait here for the stream to finish
  cudaCheck(cudaStreamSynchronize(stream));

  return num_recv_tot;
}
