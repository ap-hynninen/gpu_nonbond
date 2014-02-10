#ifndef HOLOCONST_H
#define HOLOCONST_H

class HoloConst {

private:

  bool use_textures;
  void setup_textures(double *xyz0, double *xyz1, int stride);

  // Maximum number of iterations for triplet and quad shake
  int max_niter;

  // Shake tolerance
  double shake_tol;

  //----------------------------------------------------------
  // Solvents
  //----------------------------------------------------------
  // Constants for solvent SETTLE algorithm:
  double mO_div_mH2O;
  double mH_div_mH2O;
  double ra, rc, rb, ra_inv, rc2;

  // Solvent indices
  int nsolvent;
  int solvent_ind_len;
  int3 *solvent_ind;

  //----------------------------------------------------------
  // Solute pairs
  //----------------------------------------------------------
  int npair;
  int pair_ind_len;
  int2 *pair_ind;
  int pair_constr_len;
  double *pair_constr;
  int pair_mass_len;
  double *pair_mass;

  //----------------------------------------------------------
  // Solute trips
  //----------------------------------------------------------
  int ntrip;
  int trip_ind_len;
  int3 *trip_ind;
  int trip_constr_len;
  double *trip_constr;
  int trip_mass_len;
  double *trip_mass;

  //----------------------------------------------------------
  // Solute quads
  //----------------------------------------------------------
  int nquad;
  int quad_ind_len;
  int4 *quad_ind;
  int quad_constr_len;
  double *quad_constr;
  int quad_mass_len;
  double *quad_mass;

public:

  HoloConst();
  ~HoloConst();

  void setup(double mO, double mH, double rOHsq, double rHHsq);
  void set_solvent_ind(int nsolvent, int3 *h_solvent_ind);

  void set_pair_ind(int npair, int2 *h_pair_ind,
		    double *h_pair_constr, double *h_pair_mass);

  void set_trip_ind(int ntrip, int3 *h_trip_ind,
		    double *h_trip_constr, double *h_trip_mass);

  void set_quad_ind(int nquad, int4 *h_quad_ind,
		    double *h_quad_constr, double *h_quad_mass);

  void apply(double *xyz0, double *xyz1, int stride);

};

#endif // HOLOCONST_H
