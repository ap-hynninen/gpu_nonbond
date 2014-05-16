#ifndef DIRECTFORCE_H
#define DIRECTFORCE_H
#include <cuda.h>

//
// Calculates direct non-bonded interactions on GPU
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//
// AT = accumulation type
// CT = calculation type
//

struct list14_t {
  int i, j, ishift;
};

struct DirectEnergyVirial_t {
  // Energies
  double energy_vdw;
  double energy_elec;
  double energy_excl;

  // Finished virial
  double vir[9];

  // Shift forces for virial calculation
  double sforcex[27];
  double sforcey[27];
  double sforcez[27];
};

struct DirectSettings_t {
  float kappa;
  float kappa2;

  float boxx;
  float boxy;
  float boxz;

  float roff2;
  float ron2;

  float roffinv6;
  float roffinv12;
  float roffinv18;

  float inv_roff2_ron2;

  float k6, k12, dv6, dv12;
  float roffinv3;

  float e14fac;

  float hinv;
  float *ewald_force;

};

#ifndef NEIGHBORLIST_H
template <int tilesize> class NeighborList;
#endif

// Enum for VdW and electrostatic models
enum {NONE=0, 
      VDW_VSH=1, VDW_VSW=2, VDW_VFSW=3, 
      EWALD=4,
      CSHIFT=5, CFSWIT=6, CSHFT=7, CSWIT=8, RSWIT=9,
      RSHFT=10, RSHIFT=11, RFSWIT=12,
      VDW_CUT=13,
      EWALD_LOOKUP=14};

// Enum for vdwparam
enum {VDW_MAIN, VDW_IN14};

template <typename AT, typename CT>
class DirectForce {

private:

  // VdW parameters
  int nvdwparam;
  int vdwparam_len;
  CT *vdwparam;
  bool use_tex_vdwparam;

  // VdW 1-4 parameters
  int nvdwparam14;
  int vdwparam14_len;
  CT *vdwparam14;
  bool use_tex_vdwparam14;

  // 1-4 interaction and exclusion lists
  int nin14list;
  int in14list_len;
  list14_t* in14list;

  int nex14list;
  int ex14list_len;
  list14_t* ex14list;

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

  DirectSettings_t *h_setup;
  DirectEnergyVirial_t *h_energy_virial;

  void setup_ewald_force(CT h);
  void set_elec_model(int elec_model, CT h=0.01);
  void update_setup();

  void setup_vdwparam(const int type, const int nvdwparam, const CT *h_vdwparam);
  void load_vdwparam(const char *filename, const int nvdwparam, CT **h_vdwparam);

public:

  DirectForce();
  ~DirectForce();

  void setup(CT boxx, CT boxy, CT boxz, 
	     CT kappa,
	     CT roff, CT ron,
	     CT e14fac,
	     int vdw_model, int elec_model);

  void get_box_size(CT &boxx, CT &boxy, CT &boxz);
  void set_box_size(const CT boxx, const CT boxy, const CT boxz);

  void set_calc_vdw(const bool calc_vdw);
  void set_calc_elec(const bool calc_elec);

  void set_vdwparam(const int nvdwparam, const CT *h_vdwparam);
  void set_vdwparam(const int nvdwparam, const char *filename);
  void set_vdwparam14(const int nvdwparam, const CT *h_vdwparam);
  void set_vdwparam14(const int nvdwparam, const char *filename);

  void set_vdwtype(const int ncoord, const int *h_vdwtype);
  void set_vdwtype(const int ncoord, const char *filename);
  void set_vdwtype(const int ncoord, const int *glo_vdwtype,
		   const int *loc2glo, cudaStream_t stream=0);

  void set_14_list(int nin14list, int nex14list,
		   list14_t* h_in14list, list14_t* h_ex14list);

  void calc_14_force(const float4 *xyzq,
		     const bool calc_energy, const bool calc_virial,
		     const int stride, AT *force, cudaStream_t stream=0);

  void calc_force(const float4 *xyzq,
		  const NeighborList<32> *nlist,
		  const bool calc_energy,
		  const bool calc_virial,
		  const int stride, AT *force, cudaStream_t stream=0);

  void calc_virial(const int ncoord, const float4 *xyzq,
		   const int stride, AT *force,
		   cudaStream_t stream=0);

  void clear_energy_virial(cudaStream_t stream=0);
  
  void get_energy_virial(bool prev_calc_energy, bool prev_calc_virial,
			 double *energy_vdw, double *energy_elec,
			 double *energy_excl,
			 double *vir);
};

#endif // DIRECTFORCE_H
