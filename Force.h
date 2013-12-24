#ifndef FORCE_H
#define FORCE_H

//
// Simple storage class for forces
//
template <typename T>
class Force {

public:

  // Number of coordinates in the force array
  int ncoord;

  // Stride of the force data:
  // x data is in data[0...ncoord-1];
  // y data is in data[stride...stride+ncoord-1];
  // z data is in data[stride*2...stride*2+ncoord-1];
  int stride;

  // Force data
  T *data;

  Force(const int ncoord);
  Force(const char *filename);
  ~Force();

  void setzero();
  bool compare(Force<T>* force, const double tol, double& max_diff);

  template <typename T2> void convert(Force<T2>* force);
};


#endif // FORCE_H
