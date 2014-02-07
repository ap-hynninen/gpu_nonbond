#ifndef HOLOCONST_H
#define HOLOCONST_H

class HoloConst {

private:

  // Constants for solvent SETTLE algorithm:
  double mO_div_mH2O;
  double mH_div_mH2O;
  double ra, rc, rb, ra_inv, rc2;

  // Solvent indices
  int nsolvent;
  int solvent_ind_len;
  int3 *solvent_ind;

public:

  HoloConst();
  ~HoloConst();

  void setup(double mO, double mH, double rOHsq, double rHHsq);
  void set_solvent_ind(int nsolvent, int3 *h_solvent_ind);
  void apply(double *xyz0, double *xyz1, int stride);

};

#endif // HOLOCONST_H
