#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "Matrix3d.h"

const int TILEDIM = 32;
const int TILEROWS = 8;

template <typename T>
__global__ void transpose_xyz_yzx_kernel() {
}

//
// Copies a 3d matrixL data_in(x, y, z) -> data_out(x, y, z)
//
template <typename T>
__global__ void copy_kernel(const int nx, const int ny, const int nz,
			    const int xsize, const int ysize, const int zsize,
			    const T* data_in, T* data_out) {

  const int x = blockIdx.x * TILEDIM + threadIdx.x;
  const int y = blockIdx.y * TILEDIM + threadIdx.y;
  const int z = blockIdx.z + threadIdx.z;

  for (int j=0;j < TILEDIM;j += TILEROWS)
    if ((x < nx) && (y < ny) && (z < nz))
      data_out[x + (y + j + z*ysize)*xsize] = data_in[x + (y + j + z*ysize)*xsize];

}

//
// Transposes a 3d matrix out-of-place: data_in(x, y, z) -> data_out(y, z, x)
//
template <typename T>
__global__ void transpose_xyz_yzx_kernel(const int nx, const int ny, const int nz,
					  const int xsize, const int ysize, const int zsize,
					  const T* data_in, T* data_out) {

  // Shared memory
  __shared__ T tile[TILEDIM][TILEDIM+1];

  int x = blockIdx.x * TILEDIM + threadIdx.x;
  int y = blockIdx.y * TILEDIM + threadIdx.y;
  int z = blockIdx.z           + threadIdx.z;

  // Read (x,y) data_in into tile (shared memory)
  for (int j=0;j < TILEDIM;j += TILEROWS)
    if ((x < nx) && (y + j < ny) && (z < nz))
      tile[threadIdx.y + j][threadIdx.x] = data_in[x + (y + j + z*ysize)*xsize];

  __syncthreads();

  // Write (y,x) tile into data_out
  x = blockIdx.x * TILEDIM + threadIdx.y;
  y = blockIdx.y * TILEDIM + threadIdx.x;
  for (int j=0;j < TILEDIM;j += TILEROWS)
    if ((x + j < nx) && (y < ny) && (z < nz))
      data_out[y + (z + (x+j)*zsize)*ysize] = tile[threadIdx.x][threadIdx.y + j];

}

//
// Transposes a 3d matrix out-of-place: data_in(x, y, z) -> data_out(z, x, y)
//
template <typename T>
__global__ void transpose_xyz_zxy_kernel(const int nx, const int ny, const int nz,
					 const int xsize, const int ysize, const int zsize,
					 const T* data_in, T* data_out) {

  // Shared memory
  __shared__ T tile[TILEDIM][TILEDIM+1];

  int x = blockIdx.x * TILEDIM + threadIdx.x;
  int y = blockIdx.z           + threadIdx.z;
  int z = blockIdx.y * TILEDIM + threadIdx.y;

  // Read (x,z) data_in into tile (shared memory)
  for (int k=0;k < TILEDIM;k += TILEROWS)
    if ((x < nx) && (y < ny) && (z + k < nz))
      tile[threadIdx.y + k][threadIdx.x] = data_in[x + (y + (z + k)*ysize)*xsize];

  __syncthreads();

  // Write (z,x) tile into data_out
  x = blockIdx.x * TILEDIM + threadIdx.y;
  z = blockIdx.y * TILEDIM + threadIdx.x;
  for (int k=0;k < TILEDIM;k += TILEROWS)
    if ((x + k < nx) && (y < ny) && (z < nz))
      data_out[z + (x + k + y*xsize)*zsize] = tile[threadIdx.x][threadIdx.y + k];

}

__device__ inline float2 operator*(float2 lhs, const float2& rhs) {
  lhs.x *= rhs.x;
  lhs.y *= rhs.y;
  return lhs;
}

//
// Scales matrix
//
template <typename T>
__global__ void scale_kernel(const int nx, const int ny, const int nz,
			     const int xsize, const int ysize, const int zsize,
			     const T fac, T* data) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int z = tid/(nx*ny);
  tid -= z*nx*ny;
  int y = tid/nx;
  tid -= y*nx;
  int x = tid;

  if ((x < nx) && (y < ny) && (z < nz))
    data[x + (y + z*ysize)*xsize] = data[x + (y + z*ysize)*xsize]*fac;

}

//template <typename T>
//Matrix3d<T>::Matrix3d() : nx(0), ny(0), nz(0), xsize(0), ysize(0), zsize(0) {
//  data = NULL;
//  external_storage = false;
//}

template <typename T>
Matrix3d<T>::Matrix3d(const int nx, const int ny, const int nz, T* ext_data) : 
  nx(nx), ny(ny), nz(nz), xsize(nx), ysize(ny), zsize(nz) {
  init(xsize*ysize*zsize, ext_data);
}

template <typename T>
Matrix3d<T>::Matrix3d(const int nx, const int ny, const int nz,
		      const int xsize, const int ysize, const int zsize, T* ext_data) : 
  nx(nx), ny(ny), nz(nz), xsize(xsize), ysize(ysize), zsize(zsize) {
  init(xsize*ysize*zsize, ext_data);
}

template <typename T>
Matrix3d<T>::Matrix3d(const int nx, const int ny, const int nz,
		      const char *filename, T* ext_data) : 
  nx(nx), ny(ny), nz(nz), xsize(nx), ysize(ny), zsize(nz) {
  init(xsize*ysize*zsize, ext_data);
  load(nx, ny, nz, filename);
}

template <typename T>
Matrix3d<T>::~Matrix3d() {
  if (!external_storage)
    deallocate<T>(&data);
}

template <typename T>
void Matrix3d<T>::init(const int size, T* ext_data) {
  if (ext_data == NULL) {
    allocate<T>(&data, size);
    external_storage = false;
  } else {
    data = ext_data;
    external_storage = true;
  }
}

//
// Prints matrix size on screen
//
template <typename T>
void Matrix3d<T>::print_info() {
  std::cout << "nx ny nz          = " << nx << " "<< ny << " "<< nz << std::endl;
  std::cout << "xsize ysize zsize = " << xsize << " "<< ysize << " "<< zsize << std::endl;
}

template <>
inline double Matrix3d<long long int>::norm(long long int a, long long int b) {
  return (double)llabs(a-b);
}

template <>
inline double Matrix3d<float>::norm(float a, float b) {
  return (double)fabsf(a-b);
}

template <>
inline double Matrix3d<float2>::norm(float2 a, float2 b) {
  return (double)max(fabsf(a.x-b.x), fabsf(a.y-b.y) );
}

template<>
inline bool Matrix3d<long long int>::is_nan(long long int a) {return false;};

template<>
inline bool Matrix3d<float>::is_nan(float a) {
  return isnan(a);
}

template<>
inline bool Matrix3d<float2>::is_nan(float2 a) {
  return (isnan(a.x) || isnan(a.y));
}

std::ostream& operator<<(std::ostream& os, float2& a) {
  os << a.x << " " << a.y;
  return os;
}

std::istream& operator>>(std::istream& is, float2& a) {
  is >> a.x >> a.y;
  return is;
}

//
// Compares two matrices, returns true if the difference is within tolerance
// NOTE: Comparison is done in double precision
//
template <typename T>
bool Matrix3d<T>::compare(Matrix3d<T>* mat, const double tol, double& max_diff) {

  assert(mat->nx == nx);
  assert(mat->ny == ny);
  assert(mat->nz == nz);

  T *h_data1 = new T[xsize*ysize*zsize];
  T *h_data2 = new T[mat->xsize*mat->ysize*mat->zsize];

  copy_DtoH<T>(data,      h_data1, xsize*ysize*zsize);
  copy_DtoH<T>(mat->data, h_data2, mat->xsize*mat->ysize*mat->zsize);

  bool ok = true;

  max_diff = 0.0;

  int x, y, z;
  double diff;
  try {
    for (z=0;z < nz;z++)
      for (y=0;y < ny;y++)
	for (x=0;x < nx;x++) {
	  if (is_nan(h_data1[x + (y + z*ysize)*xsize]) || 
	      is_nan(h_data2[x + (y + z*mat->ysize)*mat->xsize])) throw 1;
	  diff = norm(h_data1[x + (y + z*ysize)*xsize], h_data2[x + (y + z*mat->ysize)*mat->xsize]);
	  max_diff = (diff > max_diff) ? diff : max_diff;
	  if (diff > tol) throw 2;
	}
  }
  catch (int a) {
    std::cout << "x y z = " << x << " "<< y << " "<< z << std::endl;
    std::cout << "this: " << h_data1[x + (y + z*ysize)*xsize] << std::endl;
    std::cout << "mat:  " << h_data2[x + (y + z*mat->ysize)*mat->xsize] << std::endl;
    if (a == 2) std::cout << "difference: " << diff << std::endl;
    ok = false;
  }

  delete [] h_data1;
  delete [] h_data2;
  
  return ok;
}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
// NOTE: this is a slow reference calculation performed on the host
//
template <typename T>
void Matrix3d<T>::transpose_xyz_yzx_host(Matrix3d<T>* mat) {

  assert(mat->nx == ny);
  assert(mat->ny == nz);
  assert(mat->nz == nx);
  assert(mat->xsize == ysize);
  assert(mat->ysize == zsize);
  assert(mat->zsize == xsize);

  T *h_data1 = new T[xsize*ysize*zsize];
  T *h_data2 = new T[xsize*ysize*zsize];

  copy_DtoH<T>(data,      h_data1, xsize*ysize*zsize);
  copy_DtoH<T>(mat->data, h_data2, xsize*ysize*zsize);

  for (int z=0;z < nz;z++)
    for (int y=0;y < ny;y++)
      for (int x=0;x < nx;x++)
	h_data2[y + (z + x*zsize)*ysize] = h_data1[x + (y + z*ysize)*xsize];

  copy_HtoD<T>(h_data1, data,      xsize*ysize*zsize);
  copy_HtoD<T>(h_data2, mat->data, xsize*ysize*zsize);

  delete [] h_data1;
  delete [] h_data2;

}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
// NOTE: this is a slow reference calculation performed on the host
//
template <typename T>
void Matrix3d<T>::transpose_xyz_zxy_host(Matrix3d<T>* mat) {

  assert(mat->nx == nz);
  assert(mat->ny == nx);
  assert(mat->nz == ny);
  assert(mat->xsize == zsize);
  assert(mat->ysize == xsize);
  assert(mat->zsize == ysize);

  T *h_data1 = new T[xsize*ysize*zsize];
  T *h_data2 = new T[xsize*ysize*zsize];

  copy_DtoH<T>(data,      h_data1, xsize*ysize*zsize);
  copy_DtoH<T>(mat->data, h_data2, xsize*ysize*zsize);

  for (int z=0;z < nz;z++)
    for (int y=0;y < ny;y++)
      for (int x=0;x < nx;x++)
	h_data2[z + (x + y*xsize)*zsize] = h_data1[x + (y + z*ysize)*xsize];

  copy_HtoD<T>(h_data1, data,      xsize*ysize*zsize);
  copy_HtoD<T>(h_data2, mat->data, xsize*ysize*zsize);

  delete [] h_data1;
  delete [] h_data2;

}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
//
template <typename T>
void Matrix3d<T>::transpose_xyz_yzx(Matrix3d<T>* mat) {

  assert(mat->nx == ny);
  assert(mat->ny == nz);
  assert(mat->nz == nx);
  assert(mat->xsize == ysize);
  assert(mat->ysize == zsize);
  assert(mat->zsize == xsize);

  dim3 nthread(TILEDIM, TILEROWS, 1);
  dim3 nblock((nx-1)/TILEDIM+1, (ny-1)/TILEDIM+1, nz);

  transpose_xyz_yzx_kernel<<< nblock, nthread >>>(nx, ny, nz, xsize, ysize, zsize,
						  data, mat->data);

  cudaCheck(cudaGetLastError());

}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(z, x, y)
//
template <typename T>
void Matrix3d<T>::transpose_xyz_zxy(Matrix3d<T>* mat) {

  assert(mat->nx == nz);
  assert(mat->ny == nx);
  assert(mat->nz == ny);
  assert(mat->xsize == zsize);
  assert(mat->ysize == xsize);
  assert(mat->zsize == ysize);

  dim3 nthread(TILEDIM, TILEROWS, 1);
  dim3 nblock((nx-1)/TILEDIM+1, (nz-1)/TILEDIM+1, ny);

  transpose_xyz_zxy_kernel<<< nblock, nthread >>>(nx, ny, nz, xsize, ysize, zsize,
						  data, mat->data);

  cudaCheck(cudaGetLastError());

}

//
// Copies a 3d matrix data(x, y, z) -> data(x, y, z)
//
template <typename T>
void Matrix3d<T>::copy(Matrix3d<T>* mat) {

  assert(mat->nx == nx);
  assert(mat->ny == ny);
  assert(mat->nz == nz);
  assert(mat->xsize == xsize);
  assert(mat->ysize == ysize);
  assert(mat->zsize == zsize);

  dim3 nthread(TILEDIM, TILEROWS, 1);
  dim3 nblock((nx-1)/TILEDIM+1, (ny-1)/TILEDIM+1, nz);

  copy_kernel<<< nblock, nthread >>>(nx, ny, nz, xsize, ysize, zsize,
				     data, mat->data);

  cudaCheck(cudaGetLastError());

}

//
// Prints part of matrix (x0:x1, y0:y1, z0:z1) on screen
//
template <typename T>
void Matrix3d<T>::print(const int x0, const int x1, 
			const int y0, const int y1,
			const int z0, const int z1) {
  T *h_data = new T[xsize*ysize*zsize];

  copy_DtoH<T>(data, h_data, xsize*ysize*zsize);

  for (int z=z0;z <= z1;z++)
    for (int y=y0;y <= y1;y++)
      for (int x=x0;x <= x1;x++)
	std::cout << h_data[x + (y + z*ysize)*xsize] << std::endl;

  delete [] h_data;
}

//
// Loads Matrix of size nx,ny,nz from file "filename"
//
template <typename T>
void Matrix3d<T>::load(const int nx, const int ny, const int nz,
		       const char *filename) {

  assert(this->nx == nx);
  assert(this->ny == ny);
  assert(this->nz == nz);

  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(filename);

    // Allocate CPU memory
    T *h_data = new T[xsize*ysize*zsize];
    
    // Read data
    for (int z=0;z < nz;z++)
      for (int y=0;y < ny;y++)
	for (int x=0;x < nx;x++)
	  file >> h_data[x + (y + z*ysize)*xsize];

    // Copy data from CPU to GPU
    copy_HtoD<T>(h_data, data, xsize*ysize*zsize);

    // Deallocate CPU memory
    delete [] h_data;

    // Close file
    file.close();
  }
  catch(std::ifstream::failure e) {
    std::cerr << "Error opening/reading/closing file " << filename << std::endl;
    exit(1);
  }

}

//
// Scales the matrix by a factor "fac"
//
template <typename T>
void Matrix3d<T>::scale(const T fac) {

  int nthread = 512;
  int nblock = (nx*ny*nz-1)/512 + 1;

  scale_kernel<<< nblock, nthread >>>(nx, ny, nz, xsize, ysize, zsize, fac, data);

  cudaCheck(cudaGetLastError());

}

//
// Explicit instances of Matrix3d
//
template class Matrix3d<float>;
template class Matrix3d<float2>;
template class Matrix3d<long long int>;
