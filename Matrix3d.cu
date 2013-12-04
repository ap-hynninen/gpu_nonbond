#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include "gpu_utils.h"
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
			    const T* data_in, T* data_out) {

  const int x = blockIdx.x * TILEDIM + threadIdx.x;
  const int y = blockIdx.y * TILEDIM + threadIdx.y;
  const int z = blockIdx.z + threadIdx.z;

  for (int j=0;j < TILEDIM;j += TILEROWS)
    data_out[x + (y + j + z*ny)*nx] = data_in[x + (y + j + z*ny)*nx];

}


//
// Transposes a 3d matrix out-of-place: data_in(x, y, z) -> data_out(y, z, x)
//
template <typename T>
__global__ void transpose_xyz_yzx_kernel(const int nx, const int ny, const int nz,
					 const T* data_in, T* data_out) {

  // Shared memory
  __shared__ T tile[TILEDIM][TILEDIM+1];

  int x = blockIdx.x * TILEDIM + threadIdx.x;
  int y = blockIdx.y * TILEDIM + threadIdx.y;
  int z = blockIdx.z           + threadIdx.z;

  // Read (x,y) data_in into tile (shared memory)
  for (int j=0;j < TILEDIM;j += TILEROWS)
    tile[threadIdx.y + j][threadIdx.x] = data_in[x + (y + j + z*ny)*nx];

  __syncthreads();

  // Write (y,x) tile into data_out
  y = blockIdx.y * TILEDIM + threadIdx.x;
  x = blockIdx.x * TILEDIM + threadIdx.y;
  for (int j=0;j < TILEDIM;j += TILEROWS)
    data_out[y + (z + (x+j)*nz)*ny] = tile[threadIdx.x][threadIdx.y + j];

}

template <typename T>
Matrix3d<T>::Matrix3d() {
  nx = 0;
  ny = 0;
  nz = 0;
  data = NULL;
  data_len = 0;
}

template <typename T>
Matrix3d<T>::Matrix3d(const int nx, const int ny, const int nz) {
  set_nx_ny_nz(nx, ny, nz);
  data = NULL;
  init(nx*ny*nz);
}

template <typename T>
Matrix3d<T>::Matrix3d(const int nx, const int ny, const int nz,
		      const char *filename) {
  set_nx_ny_nz(nx, ny, nz);
  data = NULL;
  init(nx*ny*nz);
  load(nx, ny, nz, filename);
}

template <typename T>
Matrix3d<T>::~Matrix3d() {
  deallocate<T>(&data);
}

template <typename T>
void Matrix3d<T>::init(const int size) {
  reallocate<T>(&data, &data_len, size);
}

template <typename T>
void Matrix3d<T>::set_nx_ny_nz(const int nx, const int ny, const int nz) {
  this->nx = nx;
  this->ny = ny;
  this->nz = nz;
}

//
// Prints matrix size on screen
//
template <typename T>
void Matrix3d<T>::print_info() {
  std::cout << "nx ny nz = " << nx << " "<< ny << " "<< nz << std::endl;
  std::cout << "data_len = " << data_len << std::endl;
}

//
// Makes sure Matrix "mat" is has at least enough storage as this matrix
//
template <typename T>
void Matrix3d<T>::assert_size(Matrix3d<T>& mat) {
  assert(mat.data_len >= data_len);
}

//
// Compares two matrices, returns true if the difference is within tolerance
// NOTE: Comparison is done in double precision
//
template <typename T>
bool Matrix3d<T>::compare(Matrix3d<T>& mat, const T tol) {

  assert(mat.data_len >= nx*ny*nz);
  assert(mat.nx == nx);
  assert(mat.ny == ny);
  assert(mat.nz == nz);

  T *h_data1 = new T[nx*ny*nz];
  T *h_data2 = new T[nx*ny*nz];

  copy_DtoH<T>(data,     h_data1, nx*ny*nz);
  copy_DtoH<T>(mat.data, h_data2, nx*ny*nz);

  double toldbl = (double)tol;

  bool ok = true;

  for (int z=0;z < nz;z++)
    for (int y=0;y < ny;y++)
      for (int x=0;x < nx;x++) {
	double diff = fabs(h_data1[x + (y + z*ny)*nx] - h_data2[x + (y + z*ny)*nx]);
	if (ok && diff > toldbl) {
	  std::cout << "x y z = " << x << " "<< y << " "<< z << std::endl;
	  std::cout << "this: " << h_data1[x + (y + z*ny)*nx] << std::endl;
	  std::cout << "mat:  " << h_data2[x + (y + z*ny)*nx] << std::endl;
	  std::cout << "difference: " << diff << std::endl;
	  ok = false;
	}
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
void Matrix3d<T>::transpose_xyz_yzx_host(Matrix3d<T>& mat_out) {

  assert_size(mat_out);

  T *h_data1 = new T[nx*ny*nz];
  T *h_data2 = new T[nx*ny*nz];

  copy_DtoH<T>(data,         h_data1, nx*ny*nz);
  copy_DtoH<T>(mat_out.data, h_data2, nx*ny*nz);

  for (int z=0;z < nz;z++)
    for (int y=0;y < ny;y++)
      for (int x=0;x < nx;x++)
	h_data2[y + (z + x*nz)*ny] = h_data1[x + (y + z*ny)*nx];

  copy_HtoD<T>(h_data1, data,         nx*ny*nz);
  copy_HtoD<T>(h_data2, mat_out.data, nx*ny*nz);

  delete [] h_data1;
  delete [] h_data2;

  mat_out.set_nx_ny_nz(ny, nz, nx);

}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
//
template <typename T>
void Matrix3d<T>::transpose_xyz_yzx(Matrix3d<T>& mat_out) {

  assert_size(mat_out);

  dim3 nthread(TILEDIM, TILEROWS, 1);
  dim3 nblock((nx-1)/TILEDIM+1, (ny-1)/TILEDIM+1, nz);

  transpose_xyz_yzx_kernel<<< nblock, nthread >>>(nx, ny, nz, data, mat_out.data);

  cudaCheck(cudaGetLastError());

  mat_out.set_nx_ny_nz(ny, nz, nx);
}

//
// Copies a 3d matrix data(x, y, z) -> data(x, y, z)
//
template <typename T>
void Matrix3d<T>::copy(Matrix3d<T>& mat_out) {

  assert_size(mat_out);

  dim3 nthread(TILEDIM, TILEROWS, 1);
  dim3 nblock((nx-1)/TILEDIM+1, (ny-1)/TILEDIM+1, nz);

  copy_kernel<<< nblock, nthread >>>(nx, ny, nz, data, mat_out.data);

  cudaCheck(cudaGetLastError());

  mat_out.set_nx_ny_nz(nx, ny, nz);
}

//
// Loads Matrix of size nx,ny,nz from file "filename"
//
template <typename T>
void Matrix3d<T>::load(const int nx, const int ny, const int nz,
		       const char *filename) {

  assert(data_len >= nx*ny*nz);

  set_nx_ny_nz(nx, ny, nz);

  std::ifstream file(filename);
  if (file.is_open()) {
    
    // Allocate CPU memory
    T *h_data = new T[nx*ny*nz];
    
    // Read data
    for (int i=0;i < nx*ny*nz;i++)
      file >> h_data[i];

    // Copy data from CPU to GPU
    copy_HtoD<T>(h_data, data, nx*ny*nz);

    // Deallocate CPU memory
    delete [] h_data;

  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }

}

//
// Explicit instances of Matrix3d
//
template class Matrix3d<float>;
//template class Matrix3d<long long int>;
