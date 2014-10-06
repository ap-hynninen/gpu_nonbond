#ifndef HOLOCONST_H
#define HOLOCONST_H
#include "cudaXYZ.h"
#include "Bonded_struct.h"

class HoloConst {

private:

  // If true, we use indexed constraint and mass data
  bool use_indexed;
  // If true, we use SETTLE for all triplets
  bool use_settle;

  bool use_textures;
  void setup_textures(cudaXYZ<double>& xyz, int i);

  void update_setup(cudaXYZ<double>& xyz0, cudaXYZ<double>& xyz1, cudaXYZ<double>& xyz2,
		    cudaStream_t stream=0);

  void set_solvent(const int nsolvent, const solvent_t *h_solvent_ind);
  void set_solvent(const int nsolvent, const solvent_t *global_solvent_ind, const int *loc2glo);

  void set_pair(const int npair, const int2 *h_pair_ind);
  void set_pair(const int npair, const bond_t* h_pair_indtype);
  void set_pair_type(const int npair_type, const double *h_pair_constr, const double *h_pair_mass);
  void set_trip(const int ntrip, const int3 *h_trip_ind);
  void set_trip(const int ntrip, const angle_t* h_trip_indtype);
  void set_trip_type(const int ntrip_type, const double *h_trip_constr, const double *h_trip_mass);
  void set_quad(const int nquad, const int4 *h_quad_ind);
  void set_quad(const int nquad, const dihe_t* h_quad_indtype);
  void set_quad_type(const int nquad_type, const double *h_quad_constr, const double *h_quad_mass);

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

  //----------------------------------------------------------
  // Following arrays change at every neighborlist update
  //----------------------------------------------------------
  // Solvent indices
  //----------------------------------------------------------
  int nsolvent;
  int solvent_ind_len;
  solvent_t *solvent_ind;

  //----------------------------------------------------------
  // Solute pairs
  //----------------------------------------------------------
  int npair;
  int pair_ind_len;
  int2 *pair_ind;
  int pair_indtype_len;
  bond_t* pair_indtype;

  int npair_type;
  double *pair_constr;
  double *pair_mass;

  //----------------------------------------------------------
  // Solute trips
  //----------------------------------------------------------
  int ntrip;
  int trip_ind_len;
  int3 *trip_ind;
  int trip_indtype_len;
  angle_t* trip_indtype;
  
  int ntrip_type;
  double *trip_constr;
  double *trip_mass;

  //----------------------------------------------------------
  // Solute quads
  //----------------------------------------------------------
  int nquad;
  int quad_ind_len;
  int4 *quad_ind;
  int quad_indtype_len;
  dihe_t* quad_indtype;

  int nquad_type;
  double *quad_constr;
  double *quad_mass;
  
  // Pointer to all constrains and masses
  int ntot_type;
  int constr_mass_len;
  double* constr_mass;

public:

  HoloConst();
  ~HoloConst();

  void setup_solvent_parameters(double mO, double mH, double rOHsq, double rHHsq);

  void setup_settle_parameters(const int nsettle,
			       const double* h_massP, const double* h_massH,
			       const double* h_rPHsq, const double* h_rHHsq);

  void realloc_constr_mass(const int npair_type, const int ntrip_type, const int nquad_type);

  void setup_ind_mass_constr(const int npair, const int2 *h_pair_ind,
			     const double *h_pair_constr, const double *h_pair_mass,
			     const int ntrip, const int3 *h_trip_ind,
			     const double *h_trip_constr, const double *h_trip_mass,
			     const int nquad, const int4 *h_quad_ind,
			     const double *h_quad_constr, const double *h_quad_mass,
			     const int nsolvent, const solvent_t *h_solvent_ind);

  void setup_indexed(const int npair, const bond_t* h_pair_indtype,
		     const int npair_type, const double* h_pair_constr, const double* h_pair_mass,
		     const int ntrip, const angle_t* h_trip_indtype,
		     const int ntrip_type, const double* h_trip_constr, const double* h_trip_mass,
		     const int nquad, const dihe_t* h_quad_indtype,
		     const int nquad_type, const double* h_quad_constr, const double* h_quad_mass,
		     const int nsolvent, const solvent_t* h_solvent_ind);

  void setup_indexed(const int npair, const bond_t* h_pair_indtype,
		     const int npair_type, const double* h_pair_constr, const double* h_pair_mass,
		     const int ntrip, const angle_t* h_trip_indtype,
		     const int ntrip_type, const double* h_trip_constr, const double* h_trip_mass,
		     const int nquad, const dihe_t* h_quad_indtype,
		     const int nquad_type, const double* h_quad_constr, const double* h_quad_mass,
		     const int nsettle, const angle_t* h_settle_indtype);

  void setup_list(const int* glo2loc,
		  const int npair, const int* pair_tbl, const bond_t* global_pair_indtype,
		  const int ntrip, const int* trip_tbl, const angle_t* global_trip_indtype,
		  const int nquad, const int* quad_tbl, const dihe_t* global_quad_indtype,
		  const int nsolvent, const int* solvent_tbl, const solvent_t* global_solvent_ind,
		  cudaStream_t stream=0);

  void setup_types(const int npair_type, const double* h_pair_constr, const double* h_pair_mass,
		   const int ntrip_type, const double* h_trip_constr, const double* h_trip_mass,
		   const int nquad_type, const double* h_quad_constr, const double* h_quad_mass);

  void apply(cudaXYZ<double>& xyz0, cudaXYZ<double>& xyz1, cudaStream_t stream=0);

};

#endif // HOLOCONST_H
