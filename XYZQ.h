//
// XYZQ class
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//
//
class XYZQ {

private:
  int get_xyzq_len();

public:
  int align;
  int ncoord;
  int xyzq_len;
  float4 *xyzq;

  XYZQ();
  XYZQ(int ncoord, int align=1);
  XYZQ(const char *filename, int align=1);
  ~XYZQ();

  void set_ncoord(int ncoord, float fac=1.0f);
  void set_xyzq(int ncoord, float4 *h_xyzq);
};
