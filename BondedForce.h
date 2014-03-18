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

struct list14_t {
  int i, j, ishift;
};

struct cmaplist_t {
  int i1, j1, k1, l1, i2, j2, k2, l2, itype, ishift1, ishift2, ishift3;
};

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
  float2 *dihecoef;

  // -------------------
  // Improper Dihedrals
  // -------------------
  int nimdihelist;
  int nimdihecoef;

  int imdihelist_len;
  dihelist_t *imdihelist;

  int imdihecoef_len;
  float2 *imdihecoef;

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

  void setup_coef(int nbondcoef, float2 *h_bondcoef,
		  int nureybcoef, float2 *h_ureybcoef,
		  int nanglecoef, float2 *h_anglecoef,
		  int ndihecoef, float2 *h_dihecoef,
		  int nimdihecoef, float2 *h_imdihecoef,
		  int ncmapcoef, float2 *h_cmapcoef);

  void setup_list(int nbondlist, bondlist_t *h_bondlist, 
		  int nureyblist, bondlist_t *h_ureyblist,
		  int nanglelist, anglelist_t *h_anglelist,
		  int ndihelist, dihelist_t *h_dihelist,
		  int nimdihelist, dihelist_t *h_imdihelist,
		  int ncmaplist, cmaplist_t *h_cmaplist);

  void calc_force(const float4 *xyzq,
		  const float boxx, const float boxy, const float boxz,
		  const bool calc_energy,
		  const bool calc_virial,
		  const int stride, AT *force, cudaStream_t stream=0);

  void clear_energy_virial();
  void get_energy_virial(bool prev_calc_energy, bool prev_calc_virial,
			 double *energy_bond,  double *energy_ureyb, 
			 double *energy_angle,
			 double *energy_dihe, double *energy_imdihe,
			 double *energy_cmap,
			 double *sforcex, double *sforcey, double *sforcez);
};

#endif // BONDEDFORCE_H
