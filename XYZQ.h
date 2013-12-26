//
// XYZQ class
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//
//
class XYZQ {
 public:
  int ncoord;
  float4 *xyzq;

  XYZQ(int ncoord, int align=1);
  XYZQ(const char *filename, int align=1);
  ~XYZQ();
};
