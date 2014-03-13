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
  double energy_angle;
  double energy_dihe;
  double energy_imdihe;

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

  int bondlist_len;
  bondlist_t *bondlist;

  int bondcoef_len;
  float2 *bondcoef;

public:
  BondedForce();
  ~BondedForce();

  void calc_force(const float4 *xyzq,
		  const float boxx, const float boxy, const float boxz,
		  const bool calc_energy,
		  const bool calc_virial,
		  const int stride, AT *force, cudaStream_t stream);

  void clear_energy_virial();
  void get_energy_virial(bool prev_calc_energy, bool prev_calc_virial,
			 double *energy_bond, double *energy_angle,
			 double *energy_dihe, double *energy_imdihe,
			 double *sforcex, double *sforcey, double *sforcez);
};

#endif // BONDEDFORCE_H
