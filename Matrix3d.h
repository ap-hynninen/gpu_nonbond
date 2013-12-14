#ifndef MATRIX3D_H
#define MATRIX3D_H

template <typename T>
class Matrix3d {

private:
  
  // Size of the matrix
  int nx, ny, nz;

  // True if we are using an external storage for the data
  bool external_storage;

  // Initializes (allocates) data
  void init(const int size, T* ext_data = NULL);

  double norm(T a, T b);
  bool is_nan(T a);

protected:
  // Size of the matrix in storage. Allows for padding.
  int xsize, ysize, zsize;

public:

  // Matrix data
  T *data;

  //  Matrix3d();
  Matrix3d(const int nx, const int ny, const int nz, T* ext_data = NULL);
  Matrix3d(const int nx, const int ny, const int nz,
	   const int xsize, const int ysize, const int zsize, T* ext_data = NULL);
  Matrix3d(const int nx, const int ny, const int nz, const char *filename, T* ext_data = NULL);
  ~Matrix3d();

  void print_info();

  bool compare(Matrix3d<T>* mat, const double tol, double& max_diff);

  void transpose_xyz_yzx_host(Matrix3d<T>* mat);
  void transpose_xyz_zxy_host(Matrix3d<T>* mat);

  void transpose_xyz_yzx(Matrix3d<T>* mat);
  void transpose_xyz_zxy(Matrix3d<T>* mat);

  void copy(Matrix3d<T>* mat);

  void print(const int x0, const int x1, 
	     const int y0, const int y1,
	     const int z0, const int z1);
  
  void load(const int nx, const int ny, const int nz,
	    const char *filename);

  void scale(const T fac);
};

#endif // MATRIX3D_H
