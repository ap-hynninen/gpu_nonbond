#include <iostream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "Matrix3d.h"
#include "Grid.h"

//
// Grid class
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//
// In real space:
// Each instance of grid is responsible for grid region (x0..x1) x (y0..y1) x (z0..z1)
// Note that usually x0=0, x1=nfftx-1
//

template <typename T>
__forceinline__ __device__ void write_grid(const float val, const int ind,
					   T* data) {
  // The generic version can not be used
}

// Template specialization for 64bit integer = "long long int"
template <>
__forceinline__ __device__ void write_grid <long long int> (const float val,
							    const int ind,
							    long long int* data) {
  unsigned long long int qintp = llitoulli(lliroundf(FORCE_SCALE*val));
  atomicAdd((unsigned long long int *)&data[ind], qintp);
}

template <typename AT, typename CT>
__global__ void reduce_charge_data(const int nfft_tot,
				   const AT *data_in,
				   CT *data_out) {
  // The generic version can not be used
}

// Convert "long long int" -> "float"
template <>
__global__ void reduce_charge_data<long long int, float>(const int nfft_tot,
							 const long long int *data_in,
							 float *data_out) {
  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (pos < nfft_tot) {
    long long int val = data_in[pos];
    data_out[pos] = ((float)val)*INV_FORCE_SCALE;
    pos += blockDim.x*gridDim.x;
  }

}

//
// Spreads the charge on the grid
//
template <typename T>
__global__ void
__launch_bounds__(64, 1)
  spread_charge_4(const int ncoord,
		  const gridp_t *gridp,
		  const float3 *theta,
		  const int nfftx, const int nffty, const int nfftz,
		  T* data) {

  // Number of coordinates to load at once
  const int NLOADCOORD = 64;

  // Process atoms pos to pos_end-1
  unsigned int pos = blockIdx.x*NLOADCOORD + threadIdx.x;
  const unsigned int pos_end = min((blockIdx.x+1)*NLOADCOORD, ncoord);

  __shared__ int x_sh[NLOADCOORD];
  __shared__ int y_sh[NLOADCOORD];
  __shared__ int z_sh[NLOADCOORD];
  __shared__ float3 theta_sh[NLOADCOORD*4];

  // Grid point location, values of (ix0, iy0, iz0) are in range 0..3
  const int x0 = threadIdx.x & 3;
  const int y0 = (threadIdx.x >> 2) & 3;
  const int z0 = threadIdx.x >> 4;

  // Load atom data into shared memory
  if (pos < ncoord) {
    gridp_t gridpval = gridp[pos];
    int x = gridpval.x;
    int y = gridpval.y;
    int z = gridpval.z;
    float q = gridpval.q;
    x_sh[threadIdx.x] = x;
    y_sh[threadIdx.x] = y;
    z_sh[threadIdx.x] = z;
    // For each atom we write 4x4x4=64 grid points
    // 3*4 = 12 values per atom stored:
    // theta_sh[i].x = theta_x for grid point i=0...3
    // theta_sh[i].y = theta_y for grid point i=0...3
    // theta_sh[i].z = theta_z for grid point i=0...3
    int pos4 = pos*4;
    float3 theta_tmp;

    theta_tmp = theta[pos4];
    theta_tmp.x *= q;
    theta_sh[threadIdx.x*4]   = theta_tmp;

    theta_tmp = theta[pos4+1];
    theta_tmp.x *= q;
    theta_sh[threadIdx.x*4+1] = theta_tmp;

    theta_tmp = theta[pos4+2];
    theta_tmp.x *= q;
    theta_sh[threadIdx.x*4+2] = theta_tmp;

    theta_tmp = theta[pos4+3];
    theta_tmp.x *= q;
    theta_sh[threadIdx.x*4+3] = theta_tmp;
  }
  __syncthreads();

  // Loop over atoms pos..pos_end-1
  int i = 0;
  pos = blockIdx.x*NLOADCOORD;
  for (;pos < pos_end;pos++,i++) {
    int x = x_sh[i] + x0;
    int y = y_sh[i] + y0;
    int z = z_sh[i] + z0;
      
    if (x >= nfftx) x -= nfftx;
    if (y >= nffty) y -= nffty;
    if (z >= nfftz) z -= nfftz;
      
    // Get position on the grid
    int ind = x + nfftx*(y + nffty*z);
      
    // Here we unroll the 4x4x4 loop with 64 threads
    // NOTE: many thread access the same elements in the shared memory
      
    // Calculate interpolated charge value and store it to global memory
    int jx = i*4 + x0;
    int jy = i*4 + y0;
    int jz = i*4 + z0;

    //if (ind < 0 || ind >= 64*64*64) printf("ind = %d\n",ind);

    write_grid<T>(theta_sh[jx].x*theta_sh[jy].y*theta_sh[jz].z, ind, data);

  }

}

template <typename AT, typename CT>
void Grid<AT, CT>::init(int x0, int x1, int y0, int y1, int z0, int z1, int order, 
			bool y_land_locked, bool z_land_locked) {
  
  this->x0 = x0;
  this->x1 = x1;
  
  this->y0 = y0;
  this->y1 = y1;
  
  this->z0 = z0;
  this->z1 = z1;
  
  this->order = order;
  
  xlo = x0;
  xhi = x1;

  ylo = y0;
  yhi = y1;

  zlo = z0;
  zhi = z1;

  /*
  xhi += (order-1);

  if (y_land_locked) ylo -= (order-1);
  yhi += (order-1);
  
  if (z_land_locked) zlo -= (order-1);
  zhi += (order-1);
  */

  xsize = xhi - xlo + 1;
  ysize = yhi - ylo + 1;
  zsize = zhi - zlo + 1;
  
  data_size = ((xsize/2+1)*2)*ysize*zsize;

  // mat1 is used for accumulation, make sure it has enough space
  mat1.init(data_size*sizeof(AT)/sizeof(CT));
  mat2.init(data_size);
}

template <typename AT, typename CT>
Grid<AT, CT>::Grid(int nfftx, int nffty, int nfftz, int order,
		   int nnode=1,
		   int mynode=0) : nfftx(nfftx), nffty(nffty), nfftz(nfftz) {

    assert(nnode >= 1);
    assert(mynode >= 0 && mynode < nnode);
    assert(sizeof(AT) >= sizeof(CT));

    int nnode_y = max(1,(int)ceil( sqrt( (double)(nnode*nffty) / (double)(nfftz) )));
    int nnode_z = nnode/nnode_y;

    while (nnode_y*nnode_z != nnode) {
      nnode_y = nnode_y - 1;
      nnode_z = nnode/nnode_y;
    }

    assert(nnode_y != 0);
    assert(nnode_z != 0);

    // We have nodes nnode_y * nnode_z. Get y and z index of this node:
    int inode_y = mynode % nnode_y;
    int inode_z = mynode/nnode_y;

    int x0 = 0;
    int x1 = nfftx-1;

    int y0 = inode_y*nffty/nnode_y;
    int y1 = (inode_y+1)*nffty/nnode_y - 1;

    int z0 = inode_z*nfftz/nnode_z;
    int z1 = (inode_z+1)*nfftz/nnode_z - 1;

    bool y_land_locked = (inode_y-1 >= 0) && (inode_y+1 < nnode_y);
    bool z_land_locked = (inode_z-1 >= 0) && (inode_z+1 < nnode_z);

    init(x0, x1, y0, y1, z0, z1, order, y_land_locked, z_land_locked);
  }

template <typename AT, typename CT>
Grid<AT, CT>::~Grid() {}

template <typename AT, typename CT>
void Grid<AT, CT>::print_info() {
  std::cout << "order = " << order << std::endl;
  std::cout << "x0...x1   = " << x0 << " ... " << x1 << std::endl;
  std::cout << "y0...y1   = " << y0 << " ... " << y1 << std::endl;
  std::cout << "z0...z1   = " << z0 << " ... " << z1 << std::endl;
  std::cout << "xlo...xhi = " << xlo << " ... " << xhi << std::endl;
  std::cout << "ylo...yhi = " << ylo << " ... " << yhi << std::endl;
  std::cout << "zlo...zhi = " << zlo << " ... " << zhi << std::endl;
  std::cout << "xsize, ysize, zsize = " << xsize << " " << ysize << " " << zsize << std::endl;
  std::cout << "data_size = " << data_size << std::endl;
}

template <typename AT, typename CT>
void Grid<AT, CT>::spread_charge(const int ncoord, const Bspline<CT> &bspline) {

  clear_gpu_array<AT>((AT *)mat1.data, nfftx*nffty*nfftz);

  int nthread=64;
  int nblock=(ncoord-1)/nthread+1;

  std::cout<<"spread_charge, nblock = "<< nblock << std::endl;

  switch(order) {
  case 4:
    spread_charge_4<AT> <<< nblock, nthread >>>(ncoord, bspline.gridp,
						(float3 *) bspline.theta,
						nfftx, nffty, nfftz,
						(AT *)mat1.data);
    break;

  default:
    std::cerr<<"order "<<order<<" not implemented"<<std::endl;
    exit(1);
  }
  cudaCheck(cudaGetLastError());

  // Reduce charge data back to a float/double value
  nthread = 512;
  nblock = (nfftx*nffty*nfftz-1)/nthread + 1;
  reduce_charge_data<AT, CT> <<< nblock, nthread >>>(nfftx*nffty*nfftz,
						     (AT *)mat1.data,
						     mat2.data);
  cudaCheck(cudaGetLastError());

  mat1.set_nx_ny_nz(nfftx, nffty, nfftz);
  mat2.set_nx_ny_nz(nfftx, nffty, nfftz);
}

template <typename AT, typename CT>
void Grid<AT, CT>::make_fft_plans() {
  cufftResult error;

  int n = nfftx;
  int inembed = xsize;
  int istride = 1;
  int idist = xsize;
  int onembed = xsize/2;
  int ostride = 1;
  int odist = xsize/2;
  int batch = (y1-y0+1)*(z1-z0+1);

  error = cufftPlanMany(&xf_plan, 1, &n, &inembed, istride, idist,
			&onembed, ostride, odist, 
			CUFFT_R2C, batch);
  if (error != CUFFT_SUCCESS) {
    std::cerr<< "xf_plan failed" <<std::endl;
    exit(1);
  }

}

template <typename AT, typename CT>
void Grid<AT, CT>::test_copy() {

  // Copy mat2 -> mat1
  mat2.copy(mat1);

  // Compare mat2 vs. mat1
  if (!mat2.compare(mat1, 0.0f)) {
    std::cout << "test_copy FAILED" << std::endl;
    exit(1);
  }

  std::cout << "test_copy OK" << std::endl;
}

template <typename AT, typename CT>
void Grid<AT, CT>::test_transpose() {

  // Transpose using GPU: mat2(x, y, z) -> mat1(y, z, x)
  mat2.transpose_xyz_yzx(mat1);

  // Transpose using CPU: mat2(x, y, z) -> mat3(y, z, x)
  Matrix3d<CT> mat3(xsize, ysize, zsize);
  mat2.transpose_xyz_yzx_host(mat3);

  // Compare mat1 vs. mat3
  if (!mat1.compare(mat3, 0.0f)) {
    std::cout << "test_transpose FAILED" << std::endl;
    exit(1);
  }

  std::cout << "test_transpose OK" << std::endl;
}

//
// FFT x coordinate Real -> Complex
//
template <typename AT, typename CT>
void Grid<AT, CT>::x_fft_r2c() {

  cufftResult error = cufftExecR2C(xf_plan,
				   (cufftReal *)mat2.data,
				   (cufftComplex *)mat2.data);
  if (error != CUFFT_SUCCESS) {
    std::cerr<< "x_fft R2C failed" <<std::endl;
    exit(1);
  }

  mat2.set_nx_ny_nz((nfftx/2+1)*2, nffty, nfftz);

}

template <typename AT, typename CT>
void Grid<AT, CT>::real2complex_fft() {

  std::cout<<"-----------------"<<std::endl;
  print_gpu_float(mat2.data, 10);

  x_fft_r2c();

  std::cout<<"-----------------"<<std::endl;

  print_gpu_float(mat2.data, 10);

}

//
// Explicit instances of Grid
//
template class Grid<long long int, float>;
//template void Grid<long long int>::spread_charge<float>(const int,
//							const Bspline<float> &);
