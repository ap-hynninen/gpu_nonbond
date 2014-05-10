#ifndef CUDAPMEFORCEFIELD_H
#define CUDAPMEFORCEFIELD_H
#include "cudaXYZ.h"
#include "XYZQ.h"
#include "NeighborList.h"
#include "CudaForcefield.h"
#include "DirectForce.h"
#include "BondedForce.h"
#include "Grid.h"

class CudaPMEForcefield : public CudaForcefield {

private:

  // Reference coordinates for neighborlist building
  cudaXYZ<double> ref_coord;

  // flag for checking heuristic neighborlist update
  int *d_heuristic_flag;
  int *h_heuristic_flag;

  // Cut-offs:
  // Neighborlist
  double rnl;
  // Force
  double roff;
  // Force cut on
  double ron;

  // Coordinates in XYZQ format
  XYZQ xyzq;

  // Coordinates in XYZQ format
  XYZQ xyzq_sorted;

  // Neighbor list
  NeighborList<32> nlist;

  // Direct non-bonded force
  DirectForce<long long int, float> dir;

  // -------------
  // Bonded force
  // -------------
  // Global bonded lists
  int nbondlist;
  bondlist_t* bondlist;

  int nureyblist;
  bondlist_t* ureyblist;

  int nanglelist;
  anglelist_t* anglelist;

  int ndihelist;
  dihelist_t* dihelist;

  int nimdihelist;
  dihelist_t* imdihelist;

  int ncmaplist;
  cmaplist_t* cmaplist;
  
  BondedForce<long long int, float> bonded;

  // Reciprocal force
  Grid<int, float, float2> *grid; //(nfftx, nffty, nfftz, order, fft_type, numnode, mynode);

  bool heuristic_check(const cudaXYZ<double> *coord);

  void setup_bonded(const int nbondlist, const bondlist_t* h_bondlist,
		    const int nureyblist, const bondlist_t* h_ureyblist,
		    const int nanglelist, const anglelist_t* h_anglelist,
		    const int ndihelist, const dihelist_t* h_dihelist,
		    const int nimdihelist, const dihelist_t* imdihelist,
		    const int ncmaplist, const cmaplist_t* cmaplist);
public:

  CudaPMEForcefield(const int nbondlist, const bondlist_t* h_bondlist,
		    const int nureyblist, const bondlist_t* h_ureyblist,
		    const int nanglelist, const anglelist_t* h_anglelist,
		    const int ndihelist, const dihelist_t* h_dihelist,
		    const int nimdihelist, const dihelist_t* imdihelist,
		    const int ncmaplist, const cmaplist_t* cmaplist,
		    const int nbondcoef, const float2 *h_bondcoef,
		    const int nureybcoef, const float2 *h_ureybcoef,
		    const int nanglecoef, const float2 *h_anglecoef,
		    const int ndihecoef, const float4 *h_dihecoef,
		    const int nimdihecoef, const float4 *h_imdihecoef,
		    const int ncmapcoef, const float2 *h_cmapcoef,
		    const double rnl, const double roff, const double ron,
		    const double kappa, const double e14fac,
		    const int vdw_model, const int elec_model,
		    const int nvdwparam, const float *h_vdwparam,
		    const int* h_vdwtype,
		    const int nfftx, const int nffty, const int nttz,
		    const int order);
  ~CudaPMEForcefield();

  void calc(const cudaXYZ<double> *coord, const bool calc_energy, const bool calc_virial,
	    Force<long long int> *force);

};

#endif // CUDAPMEFORCEFIELD_H
