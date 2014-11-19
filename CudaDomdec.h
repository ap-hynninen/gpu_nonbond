#ifndef CUDADOMDEC_H
#define CUDADOMDEC_H

#include "Domdec.h"
#include "cudaXYZ.h"
#include "Force.h"
#include "CudaDomdecHomezone.h"
#include "CudaMPI.h"
#include "CudaDomdecD2DComm.h"
#include "CudaDomdecConstComm.h"

//class CudaDomdecBonded;

class CudaDomdec : public Domdec {

  //friend class CudaDomdecBonded;

 private:

  // (x,y,z) shift
  // NOTE: we have two copies for mappings
  int xyz_shift0_len;
  float3 *xyz_shift0;

  int xyz_shift1_len;
  float3 *xyz_shift1;
  
  // Coordinate location relative to the home box. Indexed with the global index.
  // NOTE: this array has size ncoord_glo !
  char* coordLoc;

  // Temporary mass array
  int mass_tmp_len;
  float *mass_tmp;

  // Reference to Cuda MPI object
  CudaMPI& cudaMPI;

  // Domain decomposition home zone
  CudaDomdecHomezone homezone;

  // Domain decomposition direct-direct communication
  CudaDomdecD2DComm D2Dcomm;

  // Domain decomposition constraint communicator
  CudaDomdecConstComm *constComm;

  // Error flag for calc_xyz_shift. Used for detecting atoms that are out of bounds
  int* error_flag;
  int* h_error_flag;

 public:

  CudaDomdec(int ncoord_glo, double boxx, double boxy, double boxz, double rnl,
	     int nx, int ny, int nz, int mynode, CudaMPI& cudaMPI);
  ~CudaDomdec();

  // Return box size
  double get_boxx() {return boxx;}
  double get_boxy() {return boxy;}
  double get_boxz() {return boxz;}
  double get_boxx() const {return boxx;}
  double get_boxy() const {return boxy;}
  double get_boxz() const {return boxz;}
  
  char* get_coordLoc() {return coordLoc;}
  const char* get_coordLoc() const {return coordLoc;}

  // Return pointer to local -> global mapping
  int* get_loc2glo_ptr() {return homezone.get_loc2glo_ptr();}
  const int* get_loc2glo_ptr() const {return homezone.get_loc2glo_ptr();}

  // Return pointer to (x, y, z) shift (=-1.0f, 0.0f, 1.0f)
  float3* get_xyz_shift() {return xyz_shift0;}
  const float3* get_xyz_shift() const {return xyz_shift0;}

  void constCommSetup(const int* neighPos, int* coordInd, const int* glo2loc,
		      cudaStream_t stream);
  void constCommDo(const int dir, cudaXYZ<double>& coord, cudaStream_t stream);

  void build_homezone(hostXYZ<double>& coord);
  void update_homezone(cudaXYZ<double>& coord, cudaXYZ<double>& coord2, cudaStream_t stream=0);

  void comm_coord(cudaXYZ<double>& coord, const bool update, cudaStream_t stream=0);
  void comm_update(int* glo2loc, cudaStream_t stream=0);
  void comm_force(Force<long long int>& force, cudaStream_t stream=0);

  void test_comm_coord(const int* glo2loc, cudaXYZ<double>& coord);

  void reorder_coord(const int n, cudaXYZ<double>& coord_src, cudaXYZ<double>& coord_dst,
		     const int* ind_sorted, cudaStream_t stream=0);

  void reorder_xyz_shift(const int* ind_sorted, cudaStream_t stream=0);

  //void reorder_mass(float *mass, const int* ind_sorted, cudaStream_t stream=0);

};

#endif // CUDADOMDEC_H
