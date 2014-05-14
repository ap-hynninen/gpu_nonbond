#ifndef CUDADOMDEC_H
#define CUDADOMDEC_H

#include "Decomp.h"
#include "cudaXYZ.h"
#include "Force.h"

class CudaDomdecBonded;

class CudaDomdec : public Decomp {

  friend class CudaDomdecBonded;

 private:

  // Size of the box
  double boxx, boxy, boxz;

  // Size of the neighborlist cut-off radius
  double rnl;

  // Number of sub-boxes in each coordinate direction
  int nx, ny, nz;

  // Total number of nodes (=nx*ny*nz)
  int numnode;

  // This node index (=0...numnode-1)
  int mynode;

  // Local -> global mapping
  // NOTE: also serves as a list of atom on this node
  int loc2glo_len;
  int *loc2glo;

  // Number of atoms in each node
  int zone_natom[8];
  int zone_patom[8];

  // (x,y,z) shift
  int xyz_shift_len;
  float3 *xyz_shift;

 public:

  CudaDomdec(int ncoord_tot, double boxx, double boxy, double boxz, double rnl,
	     int nx, int ny, int nz, int mynode);
  ~CudaDomdec();

  double get_boxx() {return boxx;}
  double get_boxy() {return boxy;}
  double get_boxz() {return boxz;}
  int* get_zone_patom() {return zone_patom;}
  int* get_loc2glo() {return loc2glo;}
  float3* get_xyz_shift() {return xyz_shift;}

  void get_zone_patom(int *zone_patom_out);

  double get_rnl() {return rnl;}

  void build_homezone(cudaXYZ<double> *coord, cudaStream_t stream=0);
  void update_homezone(cudaXYZ<double> *coord, cudaStream_t stream=0);

  void comm_coord(cudaXYZ<double> *coord, bool update, cudaStream_t stream=0);
  void comm_force(Force<long long int> *force, cudaStream_t stream=0);

};

#endif // CUDADOMDEC_H
