#ifndef BONDEDFORCE_H
#define BONDEDFORCE_H

//
// Calculates bonded interactions on GPU
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//
// AT = accumulation type
// CT = calculation type
//

// Data structures for bonds, angles, dihedrals, and cmap
struct bondlist_t {
  int i, j, itype, ishift;
};

struct anglelist_t {
  int i, j, k, itype, ishift1, ishift2;
};

struct dihelist_t {
  int i, j, k, l, itype, ishift1, ishift2, ishift3;
};

struct cmaplist_t {
  int i1, j1, k1, l1, i2, j2, k2, l2, itype, ishift1, ishift2, ishift3;
};

/*
// Data structure for settings
struct BondedSettings_t {
  // ------
  // Bonds
  // ------
  int nbondlist;
  bondlist_t *bondlist;
  float2 *bondcoef;

  // -------------
  // Urey-Bradley
  // -------------
  int nureyblist;
  bondlist_t *ureyblist;
  float2 *ureybcoef;

  // -------
  // Angles
  // -------
  int nanglelist;
  anglelist_t *anglelist;
  float2 *anglecoef;

  // ----------
  // Dihedrals
  // ----------
  int ndihelist;
  dihelist_t *dihelist;
  float4 *dihecoef;

  // -------------------
  // Improper Dihedrals
  // -------------------
  int nimdihelist;
  dihelist_t *imdihelist;
  float4 *imdihecoef;

  // ------
  // CMAPs
  // ------
  int ncmaplist;
  cmaplist_t *cmaplist;
  float2 *cmapcoef;

  // Other stuff
  const float4 *xyzq;
  int stride;
  float boxx;
  float boxy;
  float boxz;
  void *force;
};
*/

// Data structure for saving energies and virials
struct BondedEnergyVirial_t {
  // Energies
  double energy_bond;
  double energy_ureyb;
  double energy_angle;
  double energy_dihe;
  double energy_imdihe;
  double energy_cmap;

  // Shift forces for calculating virials
  double sforcex[27];
  double sforcey[27];
  double sforcez[27];
};


template <typename AT, typename CT>
class BondedForce {

private:

  //BondedSettings_t *h_setup;
  
  BondedEnergyVirial_t *h_energy_virial;

  // ------
  // Bonds
  // ------
  int nbondlist;
  int nbondcoef;

  int bondlist_len;
  bondlist_t *bondlist;

  int bondcoef_len;
  float2 *bondcoef;

  // -------------
  // Urey-Bradley
  // -------------
  int nureyblist;
  int nureybcoef;

  int ureyblist_len;
  bondlist_t *ureyblist;

  int ureybcoef_len;
  float2 *ureybcoef;


  // -------
  // Angles
  // -------
  int nanglelist;
  int nanglecoef;

  int anglelist_len;
  anglelist_t *anglelist;

  int anglecoef_len;
  float2 *anglecoef;

  // ----------
  // Dihedrals
  // ----------
  int ndihelist;
  int ndihecoef;

  int dihelist_len;
  dihelist_t *dihelist;

  int dihecoef_len;
  float4 *dihecoef;

  // -------------------
  // Improper Dihedrals
  // -------------------
  int nimdihelist;
  int nimdihecoef;

  int imdihelist_len;
  dihelist_t *imdihelist;

  int imdihecoef_len;
  float4 *imdihecoef;

  // ------
  // CMAPs
  // ------
  int ncmaplist;
  int ncmapcoef;

  int cmaplist_len;
  cmaplist_t *cmaplist;

  int cmapcoef_len;
  float2 *cmapcoef;

public:
  BondedForce();
  ~BondedForce();

  void setup_coef(const int nbondcoef, const float2 *h_bondcoef,
		  const int nureybcoef, const float2 *h_ureybcoef,
		  const int nanglecoef, const float2 *h_anglecoef,
		  const int ndihecoef, const float4 *h_dihecoef,
		  const int nimdihecoef, const float4 *h_imdihecoef,
		  const int ncmapcoef, const float2 *h_cmapcoef);

  void setup_list(const int nbondlist, const bondlist_t *h_bondlist, 
		  const int nureyblist, const bondlist_t *h_ureyblist,
		  const int nanglelist, const anglelist_t *h_anglelist,
		  const int ndihelist, const dihelist_t *h_dihelist,
		  const int nimdihelist, const dihelist_t *h_imdihelist,
		  const int ncmaplist, const cmaplist_t *h_cmaplist);

  void calc_force(const float4 *xyzq,
		  const float boxx, const float boxy, const float boxz,
		  const bool calc_energy,
		  const bool calc_virial,
		  const int stride, AT *force,
		  const bool calc_bond=true, const bool calc_ureyb=true,
		  const bool calc_angle=true, const bool calc_dihe=true,
		  const bool calc_imdihe=true,
		  cudaStream_t stream=0);


  void clear_energy_virial();
  void get_energy_virial(bool prev_calc_energy, bool prev_calc_virial,
			 double *energy_bond,  double *energy_ureyb, 
			 double *energy_angle,
			 double *energy_dihe, double *energy_imdihe,
			 double *energy_cmap,
			 double *sforcex, double *sforcey, double *sforcez);
};

#endif // BONDEDFORCE_H
