#ifndef CUDAPMEFORCEFIELD_H
#define CUDAPMEFORCEFIELD_H
#include "CudaForcefield.h"
#include "cudaXYZ.h"
#include "hostXYZ.h"
#include "XYZQ.h"
#include "CudaPMEDirectForce.h"
#include "BondedForce.h"
#include "Grid.h"
#include "CudaDomdec.h"
#include "CudaDomdecGroups.h"
#include "CudaDomdecRecip.h"
#include "CudaDomdecRecipComm.h"
#include "CudaNeighborList.h"

class CudaPMEForcefield : public CudaForcefield {

private:

  // Reference coordinates for neighborlist building (size ncoord)
  cudaXYZ<double> ref_coord;

  // true if neighborlist was updated in this call
  bool neighborlist_updated;

  // flag for checking heuristic neighborlist update
  int *d_heuristic_flag;
  int *h_heuristic_flag;

  // Cut-offs:
  double roff, ron;
  
  // Global charge table
  float *glo_q;

  // Coordinates in XYZQ format (size ncoord_tot)
  XYZQ xyzq;

  // Coordinates in XYZQ format (size ncoord_tot)
  XYZQ xyzq_copy;

  // -----------------
  // Neighbor list(s)
  // -----------------
  CudaNeighborList<32> nlist;

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
  XYZQ recip_xyzq;
  int recip_force_len;
  float3* recip_force;

  // ---------------------
  // Domain decomposition
  // ---------------------
  CudaDomdec& domdec;
  CudaDomdecGroups& domdecGroups;

  // ---------------------
  // Energies and virials
  // ---------------------
  double energy_bond;
  double energy_ureyb;
  double energy_angle;
  double energy_dihe;
  double energy_imdihe;
  double energy_cmap;
  double sforce[27*3];

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
  cudaEvent_t done_direct_event[2];
  cudaEvent_t done_recip_event;
  cudaEvent_t done_in14_event;
  cudaEvent_t done_bonded_event;
  cudaEvent_t done_calc_event;
  cudaEvent_t done_force_clear_event;
  cudaEvent_t xyzq_ready_event[2];
  cudaEvent_t recip_coord_ready_event;

  cudaEvent_t setup_bond_done_event;
  cudaEvent_t setup_nonbond_done_event;
  cudaEvent_t setup_14_done_event;

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

  bool heuristic_check(const cudaXYZ<double>& coord, cudaStream_t stream=0);

  void setup_direct_nonbonded(const double roff, const double ron,
			      const double kappa, const double e14fac,
			      const int vdw_model, const int elec_model,
			      const int nvdwparam, const float *h_vdwparam,
			      const float *h_vdwparam14, const int *h_glo_vdwtype);

  void setup_recip_nonbonded(const double kappa,
			     const int nfftx, const int nffty, const int nfftz,
			     const int order);

public:

  CudaPMEForcefield(CudaDomdec& domdec, CudaDomdecGroups& domdecGroups,
		    const CudaTopExcl& topExcl,
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
		    const int* h_glo_vdwtype, const float *h_glo_q,
		    CudaDomdecRecip* recip, CudaDomdecRecipComm& recipComm);
  ~CudaPMEForcefield();


  void calc(const bool calc_energy, const bool calc_virial,
	    cudaXYZ<double>& coord, cudaXYZ<double>& prev_step, Force<long long int>& force,
	    cudaStream_t stream);
  void post_calc(const float *global_mass, float *mass, HoloConst *holoconst, cudaStream_t stream);
  void stop_calc(cudaStream_t stream) {
    cudaCheck(cudaStreamSynchronize(stream));
    recipComm.send_stop();
  }

  void constComm(const int dir, cudaXYZ<double>& coord, cudaStream_t stream);

  void assignCoordToNodes(hostXYZ<double>& coord, std::vector<int>& h_loc2glo);

  void get_restart_data(cudaXYZ<double>& coord, cudaXYZ<double>& step,
			Force<long long int>& force,
			double *x, double *y, double *z,
			double *dx, double *dy, double *dz,
			double *fx, double *fy, double *fz);

  void print_energy_virial(int step, const double temp);
};

#endif // CUDAPMEFORCEFIELD_H
