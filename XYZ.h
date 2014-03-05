#ifndef XYZ_H
#define XYZ_H

//
// XYZ strided array base class
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//

template <typename T>
class XYZ {

public:
  int n;        // Number of entries
  int stride;   // Stride
  int size;     // Size of the array xyz
  T* data;      // Data array

  XYZ() {
    n = 0;
    stride = 0;
    size = 0;
    data = NULL;
  }

  // Returns true if the XYZ strided arrays match in data content sizes
  template <typename P>
  bool match(XYZ<P> &xyz) {
    return ((sizeof(T) == sizeof(P)) && (this->n == xyz.n) && (this->stride == xyz.stride));
  }

  // Resizes array to contain n entries with reallocation factor "fac"
  virtual void resize(int n, float fac=1.0f) = 0;

};

#endif // XYZ_H
