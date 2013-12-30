#ifndef DIRECTFORCE_H
#define DIRECTFORCE_H
#include <cuda.h>

//
// AT = accumulation type
// CT = calculation type
//

#ifndef NEIGHBORLIST_H
template <int tilesize> class NeighborList;
#endif

enum {NONE, EWALD, EWALD_LOOKUP, VDW_CUT, VDW_VSH, VDW_VSW, VDW_VFSW};

template <typename AT, typename CT>
class DirectForce {

private:

  // VdW parameters
  int nvdwparam;
  int vdwparam_len;
  CT *vdwparam;

  bool use_tex_vdwparam;

  // VdW types
  int vdwtype_len;
  int *vdwtype;
  
  int vdw_model;
  int elec_model;

  // These flags are true if the vdw/elec terms are calculated
  // true by default
  bool calc_vdw;
  bool calc_elec;

  // Lookup table for Ewald. Used if elec_model == EWALD_LOOKUP
  CT *ewald_force;
  int n_ewald_force;

  void setup_ewald_force(CT h);
  void set_elec_model(int elec_model, CT h=0.01);

public:

  DirectForce();
  ~DirectForce();

  void setup(CT boxx, CT boxy, CT boxz, 
	     CT kappa,
	     CT roff, CT ron,
	     int vdw_model, int elec_model,
	     bool calc_vdw, bool calc_elec);
  void set_calc_vdw(bool calc_vdw);
  void set_calc_elec(bool calc_elec);
  void set_box_size(CT boxx, CT boxy, CT boxz);

  void set_vdwparam(int nvdwparam, CT *h_vdwparam);
  void set_vdwparam(const char *filename);

  void set_vdwtype(int ncoord, int *h_vdwtype);
  void set_vdwtype(const char *filename);

  void calc_force(const int ncoord, const float4 *xyzq,
		  const NeighborList<32> *nlist,
		  const bool calc_energy,
		  const bool calc_virial,
		  const int stride, AT *force);

};

#endif // DIRECTFORCE_H
