//
// (c) Antti-Pekka Hynninen, MC barostat
//

class CudaMonteCarloBarostat {

 private:

  // Reference pressure and temperature
  double Pref;
  double Tref;

  // Number of particles
  int N;

 public:
  CudaMonteCarloBarostat(const double Pref, const double Tref, const int N);
  ~CudaMonteCarloBarostat();

  void scaleCoord();
};
