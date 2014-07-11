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

  // Local -> global mapping
  // NOTE: also serves as a list of atom on this node
  int loc2glo_len;
  int *loc2glo;

  // Number of coordinates in each node
  int zone_ncoord[8];
  int zone_pcoord[8];

  // (x,y,z) shift
  // NOTE: we have two copies for mappings
  int xyz_shift0_len;
  float3 *xyz_shift0;

  int xyz_shift1_len;
  float3 *xyz_shift1;

  // Temporary mass array
  int mass_tmp_len;
  float *mass_tmp;

 public:

  CudaDomdec(int ncoord_glo, double boxx, double boxy, double boxz, double rnl,
	     int nx, int ny, int nz, int mynode);
  ~CudaDomdec();

  // Return box size
  double get_boxx() {return boxx;}
  double get_boxy() {return boxy;}
  double get_boxz() {return boxz;}

  // Return the cumulative coordinate number
  int* get_zone_pcoord() {return zone_pcoord;}

  // Return the total number of coordinates in all zones
  int get_ncoord_tot() {return zone_pcoord[7];};

  // Return pointer to local -> global mapping
  int* get_loc2glo() {return loc2glo;}

  // Return pointer to (x, y, z) shift (=-1.0f, 0.0f, 1.0f)
  float3* get_xyz_shift() {return xyz_shift0;}

  // Return neighborlist cut-off
  double get_rnl() {return rnl;}

  void build_homezone(cudaXYZ<double> *coord, cudaStream_t stream=0);
  void update_homezone(cudaXYZ<double> *coord, cudaXYZ<double> *coord2, cudaStream_t stream=0);

  void comm_coord(cudaXYZ<double> *coord, const bool update, cudaStream_t stream=0);
  void comm_force(Force<long long int> *force, cudaStream_t stream=0);

  void reorder_coord(cudaXYZ<double> *coord_src, cudaXYZ<double> *coord_dst,
		     const int* ind_sorted, cudaStream_t stream=0);

  void reorder_xyz_shift(const int* ind_sorted, cudaStream_t stream=0);

  void reorder_mass(float *mass, const int* ind_sorted, cudaStream_t stream=0);

};

#endif // CUDADOMDEC_H
