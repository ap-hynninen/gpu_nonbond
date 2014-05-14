#ifndef CUDADOMDECBONDED_H
#define CUDADOMDECBONDED_H

#include "Bonded_struct.h"
#include "CudaDomdec.h"

class CudaDomdecBonded {

 private:

  // Global bonded lists, constant
  int nbond;
  bond_t* bond;

  int nureyb;
  bond_t* ureyb;

  int nangle;
  angle_t* angle;

  int ndihe;
  dihe_t* dihe;

  int nimdihe;
  dihe_t* imdihe;

  int ncmap;
  cmap_t* cmap;

  // Bonded tables, potentially change at every neighborlist build
  int nbond_tbl;
  int bond_tbl_len;
  int *bond_tbl;

  int nureyb_tbl;
  int ureyb_tbl_len;
  int *ureyb_tbl;

  int nangle_tbl;
  int angle_tbl_len;
  int *angle_tbl;

  int ndihe_tbl;
  int dihe_tbl_len;
  int *dihe_tbl;

  int nimdihe_tbl;
  int imdihe_tbl_len;
  int *imdihe_tbl;

  int ncmap_tbl;
  int cmap_tbl_len;
  int *cmap_tbl;

 public:

  CudaDomdecBonded(const int nbond, const bond_t* h_bond,
		   const int nureyb, const bond_t* h_ureyb,
		   const int nangle, const angle_t* h_angle,
		   const int ndihe, const dihe_t* h_dihe,
		   const int nimdihe, const dihe_t* h_imdihe,
		   const int ncmap, const cmap_t* h_cmap);
  ~CudaDomdecBonded();

  void build_tbl(const CudaDomdec* domdec, const int *zone_patom,
		 cudaStream_t stream=0);

  bond_t* get_bond() {return bond;}
  bond_t* get_ureyb() {return ureyb;}
  angle_t* get_angle() {return angle;}
  dihe_t* get_dihe() {return dihe;}
  dihe_t* get_imdihe() {return imdihe;}
  cmap_t* get_cmap() {return cmap;}

  int get_nbond_tbl() {return nbond_tbl;}
  int get_nureyb_tbl() {return nureyb_tbl;}
  int get_nangle_tbl() {return nangle_tbl;}
  int get_ndihe_tbl() {return ndihe_tbl;}
  int get_nimdihe_tbl() {return nimdihe_tbl;}
  int get_ncmap_tbl() {return ncmap_tbl;}

  int* get_bond_tbl() {return bond_tbl;}
  int* get_ureyb_tbl() {return ureyb_tbl;}
  int* get_angle_tbl() {return angle_tbl;}
  int* get_dihe_tbl() {return dihe_tbl;}
  int* get_imdihe_tbl() {return imdihe_tbl;}
  int* get_cmap_tbl() {return cmap_tbl;}

};

#endif // CUDADOMDECBONDED_H
