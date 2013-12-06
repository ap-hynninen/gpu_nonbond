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

template <typename AT, typename CT, typename CT2>
__global__ void reduce_charge_data(const int nfft_tot,
				   const AT *data_in,
				   CT *data_out) {
  // The generic version can not be used
}

// Convert "long long int" -> "float"
template <>
__global__ void reduce_charge_data<long long int, float, float2>(const int nfft_tot,
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

//
// Performs scalar sum on data(nfft1, nfft2, nfft3)
// T = {float, double}
// T2 = {float2, double2}
//
template <typename T, typename T2>
__global__ void scalar_sum_ortho_kernel(const int nfft1, const int nfft2, const int nfft3,
					const int nf1, const int nf2, const int nf3,
					const T recip11, const T recip22, const T recip33,
					const T* prefac1, const T* prefac2, const T* prefac3,
					const bool global_base, T2* data) {
  extern __shared__ T sh_buf[];

  // Create pointers to shared memory
  T* sh_prefac1 = (T *)&sh_buf[0];
  T* sh_prefac2 = (T *)&sh_buf[nfft1];
  T* sh_prefac3 = (T *)&sh_buf[nfft1+nfft2];

  // Calculate start position (k1, k2, k3) for each thread
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int k3 = tid/(nfft1*nfft2);
  tid -= k3*nfft1*nfft2;
  int k2 = tid/nfft1;
  int k1 = tid - k2*nfft1;

  // Calculate increments (k1_inc, k2_inc, k3_inc)
  int tot_inc = blockDim.x*gridDim.x;
  int k3_inc = tot_inc/(nfft1*nfft2);
  tot_inc -= k3_inc*nfft1*nfft2;
  int k2_inc = tot_inc/nfft1;
  int k1_inc = tot_inc - k2_inc*nfft1;

  // Set data[0] = 0 for the global (0,0,0)
  if (global_base && (blockIdx.x + threadIdx.x == 0)) {
    data[0] = T2(0.0, 0.0);
  }

  // Load prefac data into shared memory

  while (k3 < nfft3) {

    int pos = k1 + (k2 + k3*nfft2)*nfft1;
    T2 q = data[pos];

    int m1 = k1;
    int m2 = k2;
    int m3 = k3;
    if (k1 >= nf1) m1 -= nfft1;
    if (k2 >= nf2) m2 -= nfft2;
    if (k3 >= nf3) m3 -= nfft3;

    T mhat1 = recip11*m1;
    T mhat2 = recip22*m2;
    T mhat3 = recip33*m3;

    T msq = mhat1*mhat1 + mhat2*mhat2 + mhat3*mhat3;
    T msq_inv = (T)1.0/msq;

    // NOTE: check if it's faster to pre-calculate exp()
    T eterm = exp(-fac*msq)*piv_inv*sh_prefac1[k1]*sh_prefac2[k2]*sh_prefac3[k3]*msq_inv;

    q.x *= eterm;
    q.y *= eterm;
    data[pos] = q;
    
    // Increment position
    k1 += k1_inc;
    if (k1 >= nfft1) {
      k1 -= nfft1;
      k2++;
    }
    k2 += k2_inc;
    if (k2 >= nfft2) {
      k2 -= nfft2;
      k3++;
    }
    k3 += k3_inc;
  }

}

template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::init(int x0, int x1, int y0, int y1, int z0, int z1, int order, 
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

  data_size = (2*(xsize/2+1))*ysize*zsize;

  // data1 is used for accumulation, make sure it has enough space
  allocate<CT>(&data1, data_size*sizeof(AT)/sizeof(CT));
  allocate<CT>(&data2, data_size);

  data1_len = data_size*sizeof(AT)/sizeof(CT);
  data2_len = data_size;

  accum_grid  = new Matrix3d<AT>(xsize,         ysize, zsize, xsize,         ysize, zsize,
				 (AT *)data1);

  charge_grid = new Matrix3d<CT>(xsize,         ysize, zsize, 2*(xsize/2+1), ysize, zsize,
				 (CT *)data2);

  xfft_grid   = new Matrix3d<CT2>(xsize/2+1, ysize, zsize, xsize/2+1, ysize, zsize,
				  (CT2 *)data2);

  yfft_grid   = new Matrix3d<CT2>(ysize, zsize, xsize/2+1, ysize, zsize, xsize/2+1,
				  (CT2 *)data1);

  zfft_grid   = new Matrix3d<CT2>(zsize, xsize/2+1, ysize, zsize, xsize/2+1, ysize,
				  (CT2 *)data2);

}

template <typename AT, typename CT, typename CT2>
Grid<AT, CT, CT2>::Grid(int nfftx, int nffty, int nfftz, int order,
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

template <typename AT, typename CT, typename CT2>
Grid<AT, CT, CT2>::~Grid() {
  delete accum_grid;
  delete charge_grid;
  delete xfft_grid;
  deallocate<CT>(&data1);
  deallocate<CT>(&data2);
}

template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::print_info() {
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

//
// Spread the charge on grid
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::spread_charge(const int ncoord, const Bspline<CT> &bspline) {

  clear_gpu_array<AT>((AT *)accum_grid->data, xsize*ysize*zsize);

  int nthread=64;
  int nblock=(ncoord-1)/nthread+1;

  std::cout<<"spread_charge, nblock = "<< nblock << std::endl;

  switch(order) {
  case 4:
    spread_charge_4<AT> <<< nblock, nthread >>>(ncoord, bspline.gridp,
						(float3 *) bspline.theta,
						nfftx, nffty, nfftz,
						(AT *)accum_grid->data);
    break;

  default:
    std::cerr<<"order "<<order<<" not implemented"<<std::endl;
    exit(1);
  }
  cudaCheck(cudaGetLastError());

  // Reduce charge data back to a float/double value
  nthread = 512;
  nblock = (nfftx*nffty*nfftz-1)/nthread + 1;
  reduce_charge_data<AT, CT, CT2> <<< nblock, nthread >>>(xsize*ysize*zsize,
							  (AT *)accum_grid->data,
							  charge_grid->data);
  cudaCheck(cudaGetLastError());

}

//
// Perform scalar sum without calculating virial or energy (faster)
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::scalar_sum(const CT* recip,
				   const CT* prefac_x, const CT* prefac_y, const CT* prefac_z) {

  int nthread = 512;
  int nblock = 10;
  int shmem_size = sizeof(CT)*(nfftx + nffty + nfftz);

  int nfx = nfftx/2 + (nfftx % 2);
  int nfy = nffty/2 + (nffty % 2);
  int nfz = nfftz/2 + (nfftz % 2);

  scalar_sum_ortho_kernel<CT, CT2>
    <<< nblock, nthread, shmem_size >>> (nfftz, nfftx/2+1, nffty,
					 nfz, nfx, nfy,
					 recip[2], recip[0], recip[1],
					 prefac_z, prefac_x, prefac_y,
					 true, zfft_grid->data);
  
  cudaCheck(cudaGetLastError());

}

#define cufftCheck(stmt) do {						\
    cufftResult err = stmt;						\
    if (err != CUFFT_SUCCESS) {						\
      printf("Error running %s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
      exit(1);								\
    }									\
  } while(0)

template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::make_fft_plans() {
  int batch;

  batch = (y1-y0+1)*(z1-z0+1);
  cufftCheck(cufftPlanMany(&x_r2c_plan, 1, &nfftx,
			   NULL, 0, 0,
			   NULL, 0, 0, 
			   CUFFT_R2C, batch));
  cufftCheck(cufftSetCompatibilityMode(x_r2c_plan, CUFFT_COMPATIBILITY_NATIVE));

  batch = (z1-z0+1)*(x1-x0+1);
  cufftCheck(cufftPlanMany(&y_c2c_plan, 1, &nffty,
			   NULL, 0, 0,
			   NULL, 0, 0, 
			   CUFFT_C2C, batch));
  cufftCheck(cufftSetCompatibilityMode(y_c2c_plan, CUFFT_COMPATIBILITY_NATIVE));

  batch = (x1-x0+1)*(y1-y0+1);
  cufftCheck(cufftPlanMany(&z_c2c_plan, 1, &nfftz,
			   NULL, 0, 0,
			   NULL, 0, 0, 
			   CUFFT_C2C, batch));
  cufftCheck(cufftSetCompatibilityMode(z_c2c_plan, CUFFT_COMPATIBILITY_NATIVE));
}

/*
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
*/

//
// FFT x coordinate Real -> Complex
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::x_fft_r2c() {

  cufftCheck(cufftExecR2C(x_r2c_plan,
			  (cufftReal *)xfft_grid->data,
			  (cufftComplex *)xfft_grid->data));

}

//
// FFT y coordinate Complex -> Complex
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::y_fft_c2c() {

  cufftCheck(cufftExecC2C(y_c2c_plan,
			  (cufftComplex *)yfft_grid->data,
			  (cufftComplex *)yfft_grid->data,
			  CUFFT_FORWARD));

}

//
// FFT z coordinate Complex -> Complex
//
template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::z_fft_c2c() {

  cufftCheck(cufftExecC2C(z_c2c_plan,
			  (cufftComplex *)zfft_grid->data,
			  (cufftComplex *)zfft_grid->data,
			  CUFFT_FORWARD));

}

template <typename AT, typename CT, typename CT2>
void Grid<AT, CT, CT2>::r2c_fft() {

  // data2(x, y, z)
  x_fft_r2c();
  xfft_grid->transpose_xyz_yzx(yfft_grid);

  // data1(y, z, x)
  y_fft_c2c();
  yfft_grid->transpose_xyz_yzx(zfft_grid);

  // data2(z, x, y)
  z_fft_c2c();
}

//
// Explicit instances of Grid
//
template class Grid<long long int, float, float2>;
//template void Grid<long long int>::spread_charge<float>(const int,
//							const Bspline<float> &);
