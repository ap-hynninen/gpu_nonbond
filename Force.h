#ifndef FORCE_H
#define FORCE_H

//
// Simple storage class for forces
//
template <typename T>
class Force {

private:
  int calc_stride();

public:

  // Number of coordinates in the force array
  int ncoord;

  // Stride of the force data:
  // x data is in data[0...ncoord-1];
  // y data is in data[stride...stride+ncoord-1];
  // z data is in data[stride*2...stride*2+ncoord-1];
  int stride;

  // Force data
  int data_len;
  T *data;

  Force();
  Force(const int ncoord);
  Force(const char *filename);
  ~Force();

  void clear();
  bool compare(Force<T>* force, const double tol, double& max_diff);

  void set_ncoord(int ncoord, float fac=1.0f);
  int get_stride();
  void get_force(T *h_data);

  template <typename T2> void convert(Force<T2>* force);
  template <typename T2> void convert();
};


#endif // FORCE_H
