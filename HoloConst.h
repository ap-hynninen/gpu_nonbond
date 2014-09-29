#ifndef HOLOCONST_H
#define HOLOCONST_H
#include "cudaXYZ.h"

class HoloConst {

private:

  bool use_textures;
  void setup_textures(cudaXYZ<double>& xyz, int i);

  void update_setup(cudaXYZ<double>& xyz0, cudaXYZ<double>& xyz1, cudaXYZ<double>& xyz2,
		    cudaStream_t stream=0);

  void set_solvent(const int nsolvent, const int3 *h_solvent_ind);
  void set_solvent(const int nsolvent, const int3 *global_solvent_ind, const int *loc2glo);

  void set_pair(const int npair, const int2 *h_pair_ind,
		const double *h_pair_constr, const double *h_pair_mass);
  void set_pair(const int npair, const int2 *global_pair_ind,
		const double *global_pair_constr, const double *global_pair_mass,
		const int *loc2glo);

  void set_trip(const int ntrip, const int3 *h_trip_ind,
		const double *h_trip_constr, const double *h_trip_mass);
  void set_trip(const int ntrip, const int3 *global_trip_ind,
		const double *global_trip_constr, const double *global_trip_mass,
		const int *loc2glo);

  void set_quad(const int nquad, const int4 *h_quad_ind,
		const double *h_quad_constr, const double *h_quad_mass);
  void set_quad(const int nquad, const int4 *global_quad_ind,
		const double *global_quad_constr, const double *global_quad_mass,
		const int *loc2glo);

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

  void setup_solvent_parameters(double mO, double mH, double rOHsq, double rHHsq);

  void setup_ind_mass_constr(const int npair, const int2 *h_pair_ind,
			     const double *h_pair_constr, const double *h_pair_mass,
			     const int ntrip, const int3 *h_trip_ind,
			     const double *h_trip_constr, const double *h_trip_mass,
			     const int nquad, const int4 *h_quad_ind,
			     const double *h_quad_constr, const double *h_quad_mass,
			     const int nsolvent, const int3 *h_solvent_ind);

  void setup_ind_mass_constr(const int npair, const int2 *global_pair_ind,
			     const double *global_pair_constr, const double *global_pair_mass,
			     const int ntrip, const int3 *global_trip_ind,
			     const double *global_trip_constr, const double *global_trip_mass,
			     const int nquad, const int4 *global_quad_ind,
			     const double *global_quad_constr, const double *global_quad_mass,
			     const int nsolvent, const int3 *global_solvent_ind,
			     const int* loc2glo);

  void apply(cudaXYZ<double>& xyz0, cudaXYZ<double>& xyz1, cudaStream_t stream=0);

};

#endif // HOLOCONST_H
