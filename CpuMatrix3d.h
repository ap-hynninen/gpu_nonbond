#ifndef CPUMATRIX3D_H
#define CPUMATRIX3D_H

template <typename T>
class CpuMatrix3d {

private:
 
  // True if we are using an external storage for the data
  bool external_storage;

  // Initializes (allocates) data
  void init(const int size, T* ext_data = NULL);

  double norm(T a, T b);
  bool is_nan(T a);

protected:
  // Size of the matrix
  int nx, ny, nz;

  // Size of the matrix in storage. Allows for padding.
  int xsize, ysize, zsize;

public:

  // Matrix data
  T *data;

  CpuMatrix3d(const int nx, const int ny, const int nz, T* ext_data = NULL);
  CpuMatrix3d(const int nx, const int ny, const int nz,
	      const int xsize, const int ysize, const int zsize, T* ext_data = NULL);
  CpuMatrix3d(const int nx, const int ny, const int nz, const char *filename, T* ext_data = NULL);
  ~CpuMatrix3d();

  void print_info();

  bool compare(CpuMatrix3d<T>* mat, const double tol, double& max_diff);

  void transpose_xyz_yzx_ref(int src_x0, int src_y0, int src_z0,
			     int dst_x0, int dst_y0, int dst_z0,
			     int xlen, int ylen, int zlen,
			     CpuMatrix3d<T>* mat);
  void transpose_xyz_yzx_ref(CpuMatrix3d<T>* mat);

  void transpose_xyz_zxy_ref(CpuMatrix3d<T>* mat);

  void transpose_xyz_yzx(CpuMatrix3d<T>* mat);
  void transpose_xyz_yzx(const int src_x0, const int src_y0, const int src_z0,
			 const int dst_x0, const int dst_y0, const int dst_z0,
			 const int xlen, const int ylen, const int zlen,
			 CpuMatrix3d<T>* mat);

  void transpose_xyz_zxy(CpuMatrix3d<T>* mat);

  void copy(int src_x0, int src_y0, int src_z0,
	    int dst_x0, int dst_y0, int dst_z0,
	    int xlen, int ylen, int zlen,
	    CpuMatrix3d<T>* mat);
  void copy(CpuMatrix3d<T>* mat);

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

  int get_nx();
  int get_ny();
  int get_nz();

  int get_xsize();
  int get_ysize();
  int get_zsize();

};

#endif // CPUMATRIX3D_H
