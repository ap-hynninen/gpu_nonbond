#ifndef CUDAPMEFORCEFIELD_H
#define CUDAPMEFORCEFIELD_H
#include "CudaForcefield.h"
#include "cudaXYZ.h"
#include "hostXYZ.h"
#include "XYZQ.h"
#include "NeighborList.h"
#include "CudaPMEDirectForce.h"
#include "BondedForce.h"
#include "Grid.h"
#include "CudaDomdec.h"
#include "CudaDomdecBonded.h"
#include "CudaDomdecRecip.h"
#include "CudaDomdecRecipComm.h"

class CudaPMEForcefield : public CudaForcefield {

private:

  // Reference coordinates for neighborlist building
  cudaXYZ<double> ref_coord;

  // true if neighborlist was updated in this call
  bool neighborlist_updated;

  // flag for checking heuristic neighborlist update
  int *d_heuristic_flag;
  int *h_heuristic_flag;

  // Cut-offs:
  double roff, ron;
  
  // Global charge table
  float *q;

  // Coordinates in XYZQ format
  XYZQ xyzq;

  // Coordinates in XYZQ format
  XYZQ xyzq_copy;

  // --------------
  // Neighbor list
  // --------------
  NeighborList<32> *nlist;

  // ------------------------
  // Direct non-bonded force
  // ------------------------
  CudaPMEDirectForce<long long int, float> dir;

  // Global vdw types
  int *glo_vdwtype;

  // -------------
  // Bonded force
  // -------------
  
  BondedForce<long long int, float> bonded;

  // -----------------
  // Reciprocal force
  // -----------------
  double kappa;
  CudaDomdecRecip* recip;
  CudaDomdecRecipComm& recipComm;
  //Grid<int, float, float2> *grid;
  //Force<float> recip_force;

  // ---------------------
  // Domain decomposition
  // ---------------------
  CudaDomdec& domdec;
  CudaDomdecBonded *domdec_bonded;

  // Host version of loc2glo
  int h_loc2glo_len;
  int *h_loc2glo;

  // ---------------------
  // Energies and virials
  // ---------------------
  double energy_bond;
  double energy_ureyb;
  double energy_angle;
  double energy_dihe;
  double energy_imdihe;
  double energy_cmap;
  double sforcex[27];
  double sforcey[27];
  double sforcez[27];

  double energy_vdw;
  double energy_elec;
  double energy_excl;
  double energy_ewksum;
  double energy_ewself;
  double vir[9];

  // ------------------------------
  // Streams for force calculation
  // ------------------------------
  cudaStream_t direct_stream[2];
  cudaStream_t recip_stream;
  cudaStream_t in14_stream;
  cudaStream_t bonded_stream;

  // -----------------------------
  // Events for force calculation
  // -----------------------------
  cudaEvent_t done_direct_event;
  cudaEvent_t done_recip_event;
  cudaEvent_t done_in14_event;
  cudaEvent_t done_bonded_event;
  cudaEvent_t done_calc_event;

  // ------------------------------------------------------------
  // Flags for energy terms that are included in the calculation
  // NOTE: All true by default
  // ------------------------------------------------------------
  bool calc_bond;
  bool calc_ureyb;
  bool calc_angle;
  bool calc_dihe;
  bool calc_imdihe;
  bool calc_cmap;

  bool heuristic_check(const cudaXYZ<double> *coord, cudaStream_t stream=0);

  void setup_direct_nonbonded(const double roff, const double ron,
			      const double kappa, const double e14fac,
			      const int vdw_model, const int elec_model,
			      const int nvdwparam, const float *h_vdwparam,
			      const float *h_vdwparam14, const int *h_glo_vdwtype);

  void setup_recip_nonbonded(const double kappa,
			     const int nfftx, const int nffty, const int nfftz,
			     const int order);

public:

  CudaPMEForcefield(CudaDomdec& domdec, CudaDomdecBonded *domdec_bonded,
		    NeighborList<32> *nlist,
		    const int nbondcoef, const float2 *h_bondcoef,
		    const int nureybcoef, const float2 *h_ureybcoef,
		    const int nanglecoef, const float2 *h_anglecoef,
		    const int ndihecoef, const float4 *h_dihecoef,
		    const int nimdihecoef, const float4 *h_imdihecoef,
		    const int ncmapcoef, const float2 *h_cmapcoef,
		    const double roff, const double ron,
		    const double kappa, const double e14fac,
		    const int vdw_model, const int elec_model,
		    const int nvdwparam, const float *h_vdwparam,
		    const float *h_vdwparam14,
		    const int* h_glo_vdwtype, const float *h_q,
		    CudaDomdecRecip* recip, CudaDomdecRecipComm& recipComm);
  ~CudaPMEForcefield();


  void pre_calc(cudaXYZ<double> *coord, cudaXYZ<double> *prev_step);
  void calc(const bool calc_energy, const bool calc_virial, Force<long long int>& force);
  void post_calc(const float *global_mass, float *mass);

  void wait_calc(cudaStream_t stream);

  int init_coord(hostXYZ<double>& coord);

  void get_restart_data(hostXYZ<double> *h_coord, hostXYZ<double> *h_step, hostXYZ<double> *h_force,
			double *x, double *y, double *z, double *dx, double *dy, double *dz,
			double *fx, double *fy, double *fz);
  
  void print_energy_virial(int step);
};

#endif // CUDAPMEFORCEFIELD_H
