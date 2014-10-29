#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <math.h>
#include "cpu_utils.h"
#include "CpuMatrix3d.h"
#ifdef _OPENMP
#include <omp.h>
#endif

//
// Class Creators
//
template <typename T>
CpuMatrix3d<T>::CpuMatrix3d(const int nx, const int ny, const int nz,
			    const int tiledim, T* ext_data) : 
  nx(nx), ny(ny), nz(nz), xsize(nx), ysize(ny), zsize(nz), tiledim(tiledim) {
  assert(nx > 0);
  assert(ny > 0);
  assert(nz > 0);
  assert(tiledim > 0);
  init(xsize*ysize*zsize, ext_data);
}

template <typename T>
CpuMatrix3d<T>::CpuMatrix3d(const int nx, const int ny, const int nz,
			    const int xsize, const int ysize, const int zsize,
			    const int tiledim, T* ext_data) : 
  nx(nx), ny(ny), nz(nz), xsize(xsize), ysize(ysize), zsize(zsize), tiledim(tiledim) {
  assert(nx > 0);
  assert(ny > 0);
  assert(nz > 0);
  assert(tiledim > 0);
  assert(xsize >= nx);
  assert(ysize >= ny);
  assert(zsize >= nz);
  init(xsize*ysize*zsize, ext_data);
}

template <typename T>
CpuMatrix3d<T>::CpuMatrix3d(const int nx, const int ny, const int nz,
			    const char *filename, const int tiledim, T* ext_data) : 
  nx(nx), ny(ny), nz(nz), xsize(nx), ysize(ny), zsize(nz), tiledim(tiledim) {
  assert(nx > 0);
  assert(ny > 0);
  assert(nz > 0);
  assert(tiledim > 0);
  init(xsize*ysize*zsize, ext_data);
  load(nx, ny, nz, filename);
}

//
// Class destructor
//
template <typename T>
CpuMatrix3d<T>::~CpuMatrix3d() {
  if (external_storage == false)
    delete [] data;
  dealloc_tile();
}

//
// Allocate tilebuf_th
//
template <typename T>
void CpuMatrix3d<T>::alloc_tile() {
  if (tilebuf_th == NULL) {
    num_tilebuf_th = 1;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
    {
      num_tilebuf_th = omp_get_num_threads();
    }
#endif
    tilebuf_th = new T*[num_tilebuf_th];
    for (int i=0;i < num_tilebuf_th;i++) {
      tilebuf_th[i] = new T[tiledim*tiledim];
    }
  }
}

//
// Deallocate tilebuf_th
//
template <typename T>
void CpuMatrix3d<T>::dealloc_tile() {
  if (tilebuf_th != NULL) {
    for (int i=0;i < num_tilebuf_th;i++) {
      if (tilebuf_th[i] != NULL) delete [] tilebuf_th[i];
    }
    delete [] tilebuf_th;
  }
}

template <typename T>
void CpuMatrix3d<T>::init(const int size, T* ext_data) {
  assert(size > 0);
  if (ext_data == NULL) {
    data = new T[size];
    external_storage = false;
  } else {
    data = ext_data;
    external_storage = true;
  }
  tilebuf_th = NULL;
}

//
// Prints matrix size on screen
//
template <typename T>
void CpuMatrix3d<T>::print_info() {
  std::cout << "nx ny nz          = " << nx << " "<< ny << " "<< nz << std::endl;
  std::cout << "xsize ysize zsize = " << xsize << " "<< ysize << " "<< zsize << std::endl;
}

template <>
inline double CpuMatrix3d<long long int>::norm(long long int a, long long int b) {
  return (double)llabs(a-b);
}

template <>
inline double CpuMatrix3d<int>::norm(int a, int b) {
  return (double)abs(a-b);
}

template <>
inline double CpuMatrix3d<float>::norm(float a, float b) {
  return (double)fabsf(a-b);
}

template <>
inline double CpuMatrix3d<double>::norm(double a, double b) {
  return fabs(a-b);
}

template <>
inline double CpuMatrix3d<float2>::norm(float2 a, float2 b) {
  float dx = a.x-b.x;
  float dy = a.y-b.y;
  return (double)sqrtf(dx*dx + dy*dy);
}

template <>
inline double CpuMatrix3d<double2>::norm(double2 a, double2 b) {
  double dx = a.x-b.x;
  double dy = a.y-b.y;
  return sqrt(dx*dx + dy*dy);
}

template<>
inline bool CpuMatrix3d<long long int>::is_nan(long long int a) {return false;};

template<>
inline bool CpuMatrix3d<int>::is_nan(int a) {return false;};

template<>
inline bool CpuMatrix3d<float>::is_nan(float a) {
  return isnan(a);
}

template<>
inline bool CpuMatrix3d<double>::is_nan(double a) {
  return isnan(a);
}

template<>
inline bool CpuMatrix3d<float2>::is_nan(float2 a) {
  return isnan(a.x) || isnan(a.y);
}

template<>
inline bool CpuMatrix3d<double2>::is_nan(double2 a) {
  return isnan(a.x) || isnan(a.y);
}

std::ostream& operator<<(std::ostream& os, float2& a) {
  os << a.x << " " << a.y;
  return os;
}

std::istream& operator>>(std::istream& is, float2& a) {
  is >> a.x >> a.y;
  return is;
}

std::ostream& operator<<(std::ostream& os, double2& a) {
  os << a.x << " " << a.y;
  return os;
}

std::istream& operator>>(std::istream& is, double2& a) {
  is >> a.x >> a.y;
  return is;
}

//
// Compares two matrices, returns true if the difference is within tolerance
// NOTE: Comparison is done in double precision
//
template <typename T>
bool CpuMatrix3d<T>::compare(CpuMatrix3d<T>& mat, const double tol, double& max_diff) {

  assert(mat.nx == nx);
  assert(mat.ny == ny);
  assert(mat.nz == nz);

  bool ok = true;

  max_diff = 0.0;

  int x, y, z;
  double diff;
  try {
    for (z=0;z < nz;z++)
      for (y=0;y < ny;y++)
	for (x=0;x < nx;x++) {
	  if (is_nan(data[x + (y + z*ysize)*xsize]) || 
	      is_nan(mat.data[x + (y + z*mat.ysize)*mat.xsize])) throw 1;
	  diff = norm(data[x + (y + z*ysize)*xsize], mat.data[x + (y + z*mat.ysize)*mat.xsize]);
	  max_diff = (diff > max_diff) ? diff : max_diff;
	  if (diff > tol) throw 2;
	}
  }
  catch (int a) {
    std::cout << "x y z = " << x << " "<< y << " "<< z << std::endl;
    std::cout << "this: " << data[x + (y + z*ysize)*xsize] << std::endl;
    std::cout << "mat:  " << mat.data[x + (y + z*mat.ysize)*mat.xsize] << std::endl;
    if (a == 2) std::cout << "difference: " << diff << std::endl;
    ok = false;
  }

  return ok;
}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
// NOTE: this is a slow reference calculation
//
template <typename T>
void CpuMatrix3d<T>::transpose_yzx_ref(const int src_x0, const int src_y0, const int src_z0,
				       const int dst_x0, const int dst_y0, const int dst_z0,
				       const int xlen, const int ylen, const int zlen,
				       CpuMatrix3d<T>& mat) {

  assert(xlen > 0);
  assert(ylen > 0);
  assert(zlen > 0);

  assert(src_x0 >= 0 && src_x0 + xlen <= nx);
  assert(src_y0 >= 0 && src_y0 + ylen <= ny);
  assert(src_z0 >= 0 && src_z0 + zlen <= nz);

  assert(dst_x0 >= 0 && dst_x0 + ylen <= mat.nx);
  assert(dst_y0 >= 0 && dst_y0 + zlen <= mat.ny);
  assert(dst_z0 >= 0 && dst_z0 + xlen <= mat.nz);

  for (int z=0;z < zlen;z++)
    for (int y=0;y < ylen;y++)
      for (int x=0;x < xlen;x++) {
	mat.data[y+dst_x0 + (z+dst_y0 + (x+dst_z0)*mat.ysize)*mat.xsize] = 
	  data[x+src_x0 + (y+src_y0 + (z+src_z0)*ysize)*xsize];
      }
  
}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
// NOTE: this is a slow reference calculation performed on the host
//
template <typename T>
void CpuMatrix3d<T>::transpose_yzx_ref(CpuMatrix3d<T>& mat) {

  assert(mat.nx == ny);
  assert(mat.ny == nz);
  assert(mat.nz == nx);

  transpose_yzx_ref(0,0,0, 0,0,0, nx,ny,nz, mat);
}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
// NOTE: this is a slow reference calculation
//
template <typename T>
void CpuMatrix3d<T>::transpose_zxy_ref(const int src_x0, const int src_y0, const int src_z0,
				       const int dst_x0, const int dst_y0, const int dst_z0,
				       const int xlen, const int ylen, const int zlen,
				       CpuMatrix3d<T>& mat) {

  assert(xlen > 0);
  assert(ylen > 0);
  assert(zlen > 0);

  assert(src_x0 >= 0 && src_x0 + xlen <= nx);
  assert(src_y0 >= 0 && src_y0 + ylen <= ny);
  assert(src_z0 >= 0 && src_z0 + zlen <= nz);

  assert(dst_x0 >= 0 && dst_x0 + zlen <= mat.nx);
  assert(dst_y0 >= 0 && dst_y0 + xlen <= mat.ny);
  assert(dst_z0 >= 0 && dst_z0 + ylen <= mat.nz);

  for (int z=0;z < nz;z++)
    for (int y=0;y < ny;y++)
      for (int x=0;x < nx;x++)
	mat.data[z+dst_x0 + (x+dst_y0 + (y+dst_z0)*mat.ysize)*mat.xsize] = 
	  data[x+src_x0 + (y+src_y0 + (z+src_z0)*ysize)*xsize];

}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
// NOTE: this is a slow reference calculation performed on the host
//
template <typename T>
void CpuMatrix3d<T>::transpose_zxy_ref(CpuMatrix3d<T>& mat) {

  assert(mat.nx == nz);
  assert(mat.ny == nx);
  assert(mat.nz == ny);

  transpose_zxy_ref(0,0,0, 0,0,0, nx,ny,nz, mat);
}

//
// Transpose with order
//
template <typename T>
void CpuMatrix3d<T>::transpose(const int src_x0, const int src_y0, const int src_z0,
			       const int dst_x0, const int dst_y0, const int dst_z0,
			       const int xlen, const int ylen, const int zlen,
			       CpuMatrix3d<T>& mat, const int order) {
  assert(order == YZX || order == ZXY);
  if (order == YZX) {
    transpose_yzx(src_x0, src_y0, src_z0,
		  dst_x0, dst_y0, dst_z0,
		  xlen, ylen, zlen, mat);
  } else {
    transpose_zxy(src_x0, src_y0, src_z0,
		  dst_x0, dst_y0, dst_z0,
		  xlen, ylen, zlen, mat);
  }
}

//
// Transposes a sub block of a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
// Sub block is: (x0...x1) x (y0...y1) x (z0...z1)
//
template <typename T>
void CpuMatrix3d<T>::transpose_yzx(const int src_x0, const int src_y0, const int src_z0,
				   const int dst_x0, const int dst_y0, const int dst_z0,
				   const int xlen, const int ylen, const int zlen,
				   CpuMatrix3d<T>& mat) {
  alloc_tile();

  assert(xlen > 0);
  assert(ylen > 0);
  assert(zlen > 0);

  assert(src_x0 >= 0 && src_x0 + xlen <= nx);
  assert(src_y0 >= 0 && src_y0 + ylen <= ny);
  assert(src_z0 >= 0 && src_z0 + zlen <= nz);

  assert(dst_x0 >= 0 && dst_x0 + ylen <= mat.nx);
  assert(dst_y0 >= 0 && dst_y0 + zlen <= mat.ny);
  assert(dst_z0 >= 0 && dst_z0 + xlen <= mat.nz);

  int tid = 0;
  int ntilex = (xlen-1)/tiledim+1;
  int ntiley = (ylen-1)/tiledim+1;
  int ntile = ntilex*ntiley*zlen;
  T* tilebuf;
#ifdef _OPENMP
#pragma omp parallel private(tid, tilebuf)
#endif
  {
    tid = omp_get_thread_num();
    tilebuf = tilebuf_th[tid];
    int tile;
#ifdef _OPENMP
#pragma omp for schedule(static) private(tile)
#endif
    for (tile=0;tile < ntile;tile++) {
      // Calculate position (tilex, tiley, z)
      int tmp = tile;
      int z = tmp/(ntilex*ntiley);
      tmp -= z*(ntilex*ntiley);
      int tiley = tmp/ntilex;
      int tilex = tmp - tiley*ntilex;
      //
      int xstart = tilex*tiledim;
      int xend   = (tilex+1)*tiledim;
      xend = (xend > xlen) ? xlen : xend;
      int ystart = tiley*tiledim;
      int yend   = (tiley+1)*tiledim;
      yend = (yend > ylen) ? ylen : yend;
      // Read in data
      for (int y=ystart;y < yend;y++) {
	int src_pos = src_x0 + (y+src_y0 + (z+src_z0)*ysize)*xsize;
	int dst_pos = (y-ystart)*tiledim + (0-xstart);
	for (int x=xstart;x < xend;x++) {
	  tilebuf[dst_pos + x] = data[src_pos + x];
	}
      }
      // Write out data
      for (int x=xstart;x < xend;x++) {
	int src_pos = (x-xstart) + (0-ystart)*tiledim;
	int dst_pos = dst_x0 + (z+dst_y0 + (dst_z0 + x)*mat.ysize)*mat.xsize;
	for (int y=ystart;y < yend;y++) {
	  mat.data[dst_pos + y] = tilebuf[src_pos + y*tiledim];
	}
      }
    }
  }

}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(z, x, y)
//
template <typename T>
void CpuMatrix3d<T>::transpose_zxy(CpuMatrix3d<T>& mat) {
  assert(mat.nx == nz);
  assert(mat.ny == nx);
  assert(mat.nz == ny);
  transpose_zxy(0,0,0, 0,0,0, nx,ny,nz, mat);
}

//
// Transposes a sub block of a 3d matrix out-of-place: data(x, y, z) -> data(z, x, y)
// Sub block is: (x0...x1) x (y0...y1) x (z0...z1)
//
template <typename T>
void CpuMatrix3d<T>::transpose_zxy(const int src_x0, const int src_y0, const int src_z0,
				   const int dst_x0, const int dst_y0, const int dst_z0,
				   const int xlen, const int ylen, const int zlen,
				   CpuMatrix3d<T>& mat) {
  alloc_tile();

  assert(xlen > 0);
  assert(ylen > 0);
  assert(zlen > 0);

  assert(src_x0 >= 0 && src_x0 + xlen <= nx);
  assert(src_y0 >= 0 && src_y0 + ylen <= ny);
  assert(src_z0 >= 0 && src_z0 + zlen <= nz);

  assert(dst_x0 >= 0 && dst_x0 + zlen <= mat.nx);
  assert(dst_y0 >= 0 && dst_y0 + xlen <= mat.ny);
  assert(dst_z0 >= 0 && dst_z0 + ylen <= mat.nz);

  int tid = 0;
  int ntilex = (xlen-1)/tiledim+1;
  int ntilez = (zlen-1)/tiledim+1;
  int ntile = ntilex*ntilez*ylen;
  T* tilebuf;
#ifdef _OPENMP
#pragma omp parallel private(tid, tilebuf)
#endif
  {
    tid = omp_get_thread_num();
    tilebuf = tilebuf_th[tid];
    int tile;
#ifdef _OPENMP
#pragma omp for schedule(static) private(tile)
#endif
    for (tile=0;tile < ntile;tile++) {
      // Calculate position (tilex, tilez, y)
      int tmp = tile;
      int y = tmp/(ntilex*ntilez);
      tmp -= y*(ntilex*ntilez);
      int tilez = tmp/ntilex;
      int tilex = tmp - tilez*ntilex;
      //
      int xstart = tilex*tiledim;
      int xend   = (tilex+1)*tiledim;
      xend = (xend > xlen) ? xlen : xend;
      int zstart = tilez*tiledim;
      int zend   = (tilez+1)*tiledim;
      zend = (zend > zlen) ? zlen : zend;
      // Read in data
      for (int z=zstart;z < zend;z++) {
	int src_pos = src_x0 + (y+src_y0 + (z+src_z0)*ysize)*xsize;
	int dst_pos = (z-zstart)*tiledim + (0-xstart);
	for (int x=xstart;x < xend;x++) {
	  tilebuf[dst_pos + x] = data[src_pos + x];
	}
      }
      // Write out data
      for (int x=xstart;x < xend;x++) {
	int src_pos = (x-xstart) + (0-zstart)*tiledim;
	int dst_pos = dst_x0 + (x+dst_y0 + (y+dst_z0)*mat.ysize)*mat.xsize;
	for (int z=zstart;z < zend;z++) {
	  mat.data[dst_pos + z] = tilebuf[src_pos + z*tiledim];
	}
      }
    }
  }
}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
//
template <typename T>
void CpuMatrix3d<T>::transpose_yzx(CpuMatrix3d<T>& mat) {
  assert(mat.nx == ny);
  assert(mat.ny == nz);
  assert(mat.nz == nx);
  transpose_yzx(0,0,0, 0,0,0, nx,ny,nz, mat);
}

//
// Copies a 3d matrix this->data(x, y, z) -> mat.data(x, y, z)
//
template <typename T>
void CpuMatrix3d<T>::copy(int src_x0, int src_y0, int src_z0,
			  int dst_x0, int dst_y0, int dst_z0,
			  int xlen, int ylen, int zlen,
			  CpuMatrix3d<T>& mat) {

  assert(xlen > 0);
  assert(ylen > 0);
  assert(zlen > 0);

  assert(src_x0 >= 0 && src_x0 + xlen <= nx);
  assert(src_y0 >= 0 && src_y0 + ylen <= ny);
  assert(src_z0 >= 0 && src_z0 + zlen <= nz);

  assert(dst_x0 >= 0 && dst_x0 + xlen <= mat.nx);
  assert(dst_y0 >= 0 && dst_y0 + ylen <= mat.ny);
  assert(dst_z0 >= 0 && dst_z0 + zlen <= mat.nz);

  copy3D_HtoH<T>(this->data, mat.data,
		 src_x0, src_y0, src_z0,
		 (size_t)this->xsize, (size_t)this->ysize,
		 dst_x0, dst_y0, dst_z0,
		 (size_t)xlen, (size_t)ylen, (size_t)zlen,
		 (size_t)mat.xsize, (size_t)mat.ysize);

}

//
// Transposes a sub block of a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
// Sub block is: (x0...x1) x (y0...y1) x (z0...z1)
//
template <typename T>
void CpuMatrix3d<T>::transpose_yzx_legacy(const int src_x0, const int src_y0, const int src_z0,
					  const int dst_x0, const int dst_y0, const int dst_z0,
					  const int xlen, const int ylen, const int zlen,
					  CpuMatrix3d<T>& mat) {
  assert(xlen > 0);
  assert(ylen > 0);
  assert(zlen > 0);

  assert(src_x0 >= 0 && src_x0 + xlen <= nx);
  assert(src_y0 >= 0 && src_y0 + ylen <= ny);
  assert(src_z0 >= 0 && src_z0 + zlen <= nz);

  assert(dst_x0 >= 0 && dst_x0 + ylen <= mat.nx);
  assert(dst_y0 >= 0 && dst_y0 + zlen <= mat.ny);
  assert(dst_z0 >= 0 && dst_z0 + xlen <= mat.nz);

  int xysize = mat.xsize*mat.ysize;

  int z;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(z)
#endif
  for (z=0;z < zlen;z++) {
    for (int y=0;y < ylen;y++) {
      int dst_pos = y+dst_x0 + (z+dst_y0 + (dst_z0)*mat.ysize)*mat.xsize;
      int src_pos = src_x0 + (y+src_y0 + (z+src_z0)*ysize)*xsize;
      for (int x=0;x < xlen;x++) {
	mat.data[dst_pos + x*xysize] = data[src_pos + x];
      }
    }
  }

}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(y, z, x)
//
template <typename T>
void CpuMatrix3d<T>::transpose_yzx_legacy(CpuMatrix3d<T>& mat) {
  assert(mat.nx == ny);
  assert(mat.ny == nz);
  assert(mat.nz == nx);
  transpose_yzx(0,0,0, 0,0,0, nx,ny,nz, mat);
}

//
// Transposes a sub block of a 3d matrix out-of-place: data(x, y, z) -> data(z, x, y)
// Sub block is: (x0...x1) x (y0...y1) x (z0...z1)
//
template <typename T>
void CpuMatrix3d<T>::transpose_zxy_legacy(const int src_x0, const int src_y0, const int src_z0,
					  const int dst_x0, const int dst_y0, const int dst_z0,
					  const int xlen, const int ylen, const int zlen,
					  CpuMatrix3d<T>& mat) {
  assert(xlen > 0);
  assert(ylen > 0);
  assert(zlen > 0);

  assert(src_x0 >= 0 && src_x0 + xlen <= nx);
  assert(src_y0 >= 0 && src_y0 + ylen <= ny);
  assert(src_z0 >= 0 && src_z0 + zlen <= nz);

  assert(dst_x0 >= 0 && dst_x0 + zlen <= mat.nx);
  assert(dst_y0 >= 0 && dst_y0 + xlen <= mat.ny);
  assert(dst_z0 >= 0 && dst_z0 + ylen <= mat.nz);

  int xysize = xsize*ysize;

  int y;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(y)
#endif
  for (y=0;y < ylen;y++) {
    for (int x=0;x < xlen;x++) {
      int dst_pos = dst_x0 + (x+dst_y0 + (y+dst_z0)*mat.ysize)*mat.xsize;
      int src_pos = x+src_x0 + (y+src_y0 + (src_z0)*ysize)*xsize;
      for (int z=0;z < zlen;z++) {
	mat.data[dst_pos + z] = data[src_pos + z*xysize];
      }
    }
  }

}

//
// Transposes a 3d matrix out-of-place: data(x, y, z) -> data(z, x, y)
//
template <typename T>
void CpuMatrix3d<T>::transpose_zxy_legacy(CpuMatrix3d<T>& mat) {
  assert(mat.nx == nz);
  assert(mat.ny == nx);
  assert(mat.nz == ny);
  transpose_zxy_legacy(0,0,0, 0,0,0, nx,ny,nz, mat);
}

//
// Copies a 3d matrix data(x, y, z) -> data(x, y, z)
//
template <typename T>
void CpuMatrix3d<T>::copy(CpuMatrix3d<T>& mat) {
  assert(mat.nx == nx);
  assert(mat.ny == ny);
  assert(mat.nz == nz);
  copy(0,0,0, 0,0,0, nx, ny, nz, mat);
}

//
// Prints part of matrix (x0:x1, y0:y1, z0:z1) on screen
//
template <typename T>
void CpuMatrix3d<T>::print(const int x0, const int x1, 
			   const int y0, const int y1,
			   const int z0, const int z1) {

  for (int z=z0;z <= z1;z++)
    for (int y=y0;y <= y1;y++)
      for (int x=x0;x <= x1;x++)
	std::cout << data[x + (y + z*ysize)*xsize] << std::endl;

}

//
// Loads Matrix block (x0...x1) x (y0...y1) x (z0...z1) from file "filename"
// Matrix in file has size nx x ny x nz
//
template <typename T>
void CpuMatrix3d<T>::load(const int x0, const int x1, const int nx,
		       const int y0, const int y1, const int ny,
		       const int z0, const int z1, const int nz,
		       const char *filename) {

  assert(x0 < x1);
  assert(y0 < y1);
  assert(z0 < z1);
  assert(x0 >= 0 && x1 < nx);
  assert(y0 >= 0 && y1 < ny);
  assert(z0 >= 0 && z1 < nz);

  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(filename);

    // Read data
    for (int z=0;z < nz;z++)
      for (int y=0;y < ny;y++)
	for (int x=0;x < nx;x++)
	  if (x >= x0 && x <= x1 &&
	      y >= y0 && y <= y1 &&
	      z >= z0 && z <= z1) {
	    file >> data[x-x0 + (y-y0 + (z-z0)*ysize)*xsize];
	  } else {
	    T dummy;
	    file >> dummy;
	  }

    // Close file
    file.close();
  }
  catch(std::ifstream::failure e) {
    std::cerr << "Error opening/reading/closing file " << filename << std::endl;
    exit(1);
  }

}

//
// Loads Matrix of size nx x ny x nz from file "filename"
//
template <typename T>
void CpuMatrix3d<T>::load(const int nx, const int ny, const int nz,
		       const char *filename) {


  assert(this->nx == nx);
  assert(this->ny == ny);
  assert(this->nz == nz);

  load(0, nx-1, nx,
       0, ny-1, ny,
       0, nz-1, nz, filename);

}

/*
//
// Scales the matrix by a factor "fac"
//
template <typename T>
void CpuMatrix3d<T>::scale(const T fac) {
  for (int z=0;z < nz;z++)
    for (int y=0;y < ny;y++)
      for (int x=0;x < nx;x++)
	data[x + (y + z*ysize)*xsize] *= fac;
}
*/

//
// Explicit instances of CpuMatrix3d
//
template class CpuMatrix3d<float>;
template class CpuMatrix3d<float2>;
template class CpuMatrix3d<double>;
template class CpuMatrix3d<double2>;
template class CpuMatrix3d<long long int>;
template class CpuMatrix3d<int>;
