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

  // Neighbor list
  NeighborList<32> nlist;

  // Direct non-bonded force
  DirectForce<long long int, float> dir;

  // Bonded force
  BondedForce<long long int, float> bonded;

  // Reciprocal force
  Grid<int, float, float2> *grid; //(nfftx, nffty, nfftz, order, fft_type, numnode, mynode);

  bool heuristic_check(const cudaXYZ<double> *coord);

public:

  CudaPMEForcefield();
  ~CudaPMEForcefield();

  void calc(const cudaXYZ<double> *coord, const bool calc_energy, const bool calc_virial,
	    Force<long long int> *force);

};

#endif // CUDAPMEFORCEFIELD_H
