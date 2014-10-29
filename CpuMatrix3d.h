#ifndef CPUMATRIX3D_H
#define CPUMATRIX3D_H

enum {YZX=0, ZXY=1};

template <typename T>
class CpuMatrix3d {

private:
 
  // True if we are using an external storage for the data
  bool external_storage;

  // Number of entries in tilebuf_th, must be equal to the number of threads!
  int num_tilebuf_th;

  // Storage for tile: tilebuf_th[0..num_tilebuf_th-1][0..tiledim*tiledim-1]
  T** tilebuf_th;
  T* tilebuf_heap;

  // Initializes (allocates) data
  void init(const int size, T* ext_data = NULL);

  void alloc_tile();
  void dealloc_tile();

  double norm(T a, T b);
  bool is_nan(T a);

protected:
  // Tile dimensions
  const int tiledim;

  // Size of the matrix
  const int nx, ny, nz;

  // Size of the matrix in storage. Allows for padding.
  const int xsize, ysize, zsize;

  // Matrix data
  T* data;

public:

  CpuMatrix3d(const int nx, const int ny, const int nz,
	      const int tiledim=64, T* ext_data = NULL);
  CpuMatrix3d(const int nx, const int ny, const int nz,
	      const int xsize, const int ysize, const int zsize,
	      const int tiledim=64, T* ext_data = NULL);
  CpuMatrix3d(const int nx, const int ny, const int nz, const char *filename,
	      const int tiledim=64, T* ext_data = NULL);
  ~CpuMatrix3d();

  void print_info();

  bool compare(CpuMatrix3d<T>& mat, const double tol, double& max_diff);

  void transpose_yzx_ref(const int src_x0, const int src_y0, const int src_z0,
			 const int dst_x0, const int dst_y0, const int dst_z0,
			 const int xlen, const int ylen, const int zlen,
			 CpuMatrix3d<T>& mat);
  void transpose_yzx_ref(CpuMatrix3d<T>& mat);

  void transpose_zxy_ref(const int src_x0, const int src_y0, const int src_z0,
			 const int dst_x0, const int dst_y0, const int dst_z0,
			 const int xlen, const int ylen, const int zlen,
			 CpuMatrix3d<T>& mat);
  void transpose_zxy_ref(CpuMatrix3d<T>& mat);

  void transpose(const int src_x0, const int src_y0, const int src_z0,
		 const int dst_x0, const int dst_y0, const int dst_z0,
		 const int xlen, const int ylen, const int zlen,
		 CpuMatrix3d<T>& mat, const int order);

  void transpose_yzx(const int src_x0, const int src_y0, const int src_z0,
		     const int dst_x0, const int dst_y0, const int dst_z0,
		     const int xlen, const int ylen, const int zlen,
		     CpuMatrix3d<T>& mat);
  void transpose_yzx(CpuMatrix3d<T>& mat);

  void transpose_zxy(const int src_x0, const int src_y0, const int src_z0,
		     const int dst_x0, const int dst_y0, const int dst_z0,
		     const int xlen, const int ylen, const int zlen,
		     CpuMatrix3d<T>& mat);
  void transpose_zxy(CpuMatrix3d<T>& mat);

  void transpose_yzx_legacy(CpuMatrix3d<T>& mat);
  void transpose_yzx_legacy(const int src_x0, const int src_y0, const int src_z0,
			    const int dst_x0, const int dst_y0, const int dst_z0,
			    const int xlen, const int ylen, const int zlen,
			    CpuMatrix3d<T>& mat);

  void transpose_zxy_legacy(CpuMatrix3d<T>& mat);
  void transpose_zxy_legacy(const int src_x0, const int src_y0, const int src_z0,
			    const int dst_x0, const int dst_y0, const int dst_z0,
			    const int xlen, const int ylen, const int zlen,
			    CpuMatrix3d<T>& mat);

  void copy(int src_x0, int src_y0, int src_z0,
	    int dst_x0, int dst_y0, int dst_z0,
	    int xlen, int ylen, int zlen,
	    CpuMatrix3d<T>& mat);
  void copy(CpuMatrix3d<T>& mat);

  void print(const int x0, const int x1, 
	     const int y0, const int y1,
	     const int z0, const int z1);
  
  void load(const int x0, const int x1, const int nx,
	    const int y0, const int y1, const int ny,
	    const int z0, const int z1, const int nz,
	    const char *filename);

  void load(const int nx, const int ny, const int nz,
	    const char *filename);

  void scale(const T fac);

  int get_nx() {return nx;}
  int get_ny() {return ny;}
  int get_nz() {return nz;}

  int get_xsize() {return xsize;}
  int get_ysize() {return ysize;}
  int get_zsize() {return zsize;}

  T* get_data() {return data;}

};

#endif // CPUMATRIX3D_H
