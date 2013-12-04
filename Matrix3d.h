#ifndef MATRIX3D_H
#define MATRIX3D_H

template <typename T>
class Matrix3d {

private:
  
  // Size of the matrix
  int nx, ny, nz;

  // Size of the matrix data buffer
  int data_len;

  void assert_size(Matrix3d<T>& mat);

public:

  // Matrix data
  T *data;

  Matrix3d();
  Matrix3d(const int nx, const int ny, const int nz);
  Matrix3d(const int nx, const int ny, const int nz, const char *filename);
  ~Matrix3d();

  void print_info();

  void init(const int size);

  void set_nx_ny_nz(const int nx, const int ny, const int nz);

  bool compare(Matrix3d<T>& mat, const T tol);

  void transpose_xyz_yzx_host(Matrix3d<T>& mat_out);
  void transpose_xyz_yzx(Matrix3d<T>& mat_out);

  void copy(Matrix3d<T>& mat_out);

  void load(const int nx, const int ny, const int nz,
	    const char *filename);
};

#endif // MATRIX3D_H
