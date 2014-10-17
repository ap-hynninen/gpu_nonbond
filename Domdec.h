#ifndef DOMDEC_H
#define DOMDEC_H

#include <cassert>
#include <vector>
#include <mpi.h>
#include "AtomGroupBase.h"

//
// Base class for atom decompositors
//
class Domdec {

private:
  // MPI communicator
  MPI_Comm comm;

  // Number of coordinates in each zone
  // zone_ncoord[i] = number of coordinates in zone i
  // zone_pcoord[i] = zone i starting position in the coordinate arrays
  int zone_ncoord[8];
  int zone_pcoord[9];

  // Fractional sizes of sub-boxes
  std::vector<double> fx;
  std::vector< std::vector<double> > fy;
  std::vector< std::vector<std::vector<double> > > fz;

  // Fractional borders of sub-boxes
  std::vector<double> bx;
  std::vector< std::vector<double> > by;
  std::vector< std::vector<std::vector<double> > > bz;

  // Used for checking the total number of groups
  std::vector<int> numGroups;
  std::vector<int> numGroupsTot;

  // Calculate zone_pcoord
  void update_zone_pcoord() {
    zone_pcoord[0] = 0;
    for (int i=1;i <= 8;i++) zone_pcoord[i] = zone_pcoord[i-1] + zone_ncoord[i-1];
  }

  // Calculates (bx, by, bz)
  void update_bxyz() {
    bx.at(0) = 0.0;
    for (int ix=1;ix <= nx;ix++) {
      bx.at(ix) = bx.at(ix-1) + fx.at(ix-1);
    }

    for (int ix=0;ix < nx;ix++) {
      by.at(0).at(0) = 0.0;
      for (int iy=1;iy <= ny;iy++) {
	by.at(ix).at(iy) = by.at(ix).at(iy-1) + fy.at(ix).at(iy-1);
      }
    }

    for (int ix=0;ix < nx;ix++) {
      for (int iy=0;iy < ny;iy++) {
	bz.at(0).at(0).at(0) = 0.0;
	for (int iz=1;iz <= nz;iz++) {
	  bz.at(ix).at(iy).at(iz) = bz.at(ix).at(iy).at(iz-1) + fz.at(ix).at(iy).at(iz-1);
	}
      }
    }
  }

protected:
  
  // Total global number of coordinates in the system
  int ncoord_glo;

  // Total number of nodes
  int numnode;

  // This node index (=0...numnode-1)
  int mynode;

  // Homebox index
  int homeix, homeiy, homeiz;

  // Size of the box
  double boxx, boxy, boxz;

  // Size of the neighborlist cut-off radius
  double rnl;

  // Number of sub-boxes in each coordinate direction
  int nx, ny, nz;

 public:

  // Order of zones
  // I,FZ,FY,EX,FX,EZ,EY,C = 0,...7
  enum {I=0,FZ=1,FY=2,EX=3,FX=4,EZ=5,EY=6,C=7};

  Domdec(int ncoord_glo, double boxx, double boxy, double boxz, double rnl,
	 int nx, int ny, int nz, int mynode, MPI_Comm comm);

  // Returns fractional boundaries for nodes relative to this node
  double get_lo_bx(int x=0) {return bx.at((homeix+x+nx)%nx);}
  double get_hi_bx(int x=0) {return bx.at((homeix+x+nx)%nx+1);}
  double get_lo_by(int x=0,int y=0) {return by.at((homeix+x+nx)%nx).at((homeiy+y+ny)%ny);}
  double get_hi_by(int x=0,int y=0) {return by.at((homeix+x+nx)%nx).at((homeiy+y+ny)%ny+1);}
  double get_lo_bz(int x=0,int y=0,int z=0) {
    return bz.at((homeix+x+nx)%nx).at((homeiy+y+ny)%ny).at((homeiz+z+nz)%nz);
  }
  double get_hi_bz(int x=0,int y=0,int z=0) {
    return bz.at((homeix+x+nx)%nx).at((homeiy+y+ny)%ny).at((homeiz+z+nz)%nz+1);
  }
  double get_lo_bx(int x=0) const {return bx.at((homeix+x+nx)%nx);}
  double get_hi_bx(int x=0) const {return bx.at((homeix+x+nx)%nx+1);}
  double get_lo_by(int x=0,int y=0) const {return by.at((homeix+x+nx)%nx).at((homeiy+y+ny)%ny);}
  double get_hi_by(int x=0,int y=0) const {return by.at((homeix+x+nx)%nx).at((homeiy+y+ny)%ny+1);}
  double get_lo_bz(int x=0,int y=0,int z=0) const {
    return bz.at((homeix+x+nx)%nx).at((homeiy+y+ny)%ny).at((homeiz+z+nz)%nz);
  }
  double get_hi_bz(int x=0,int y=0,int z=0) const {
    return bz.at((homeix+x+nx)%nx).at((homeiy+y+ny)%ny).at((homeiz+z+nz)%nz+1);
  }

  /*
  double get_lo_bx() const {return bx.at(homeix);}
  double get_hi_bx() const {return bx.at(homeix+1);}
  double get_lo_by() const {return by.at(homeix).at(homeiy);}
  double get_hi_by() const {return by.at(homeix).at(homeiy+1);}
  double get_lo_bz() const {return bz.at(homeix).at(homeiy).at(homeiz);}
  double get_hi_bz() const {return bz.at(homeix).at(homeiy).at(homeiz+1);}
  */

  // Return the global total number of coordinates
  int get_ncoord_glo() {return ncoord_glo;}
  int get_ncoord_glo() const {return ncoord_glo;}

  // Return the number of coordinates in the homezone
  int get_ncoord() {return zone_ncoord[0];}
  int get_ncoord() const {return zone_ncoord[0];}

  // Return number of nodes
  int get_numnode() {return numnode;}
  int get_numnode() const {return numnode;}

  // Return the cumulative coordinate number
  const int* get_zone_pcoord() {return zone_pcoord;}
  const int* get_zone_pcoord() const {return zone_pcoord;}

  int get_zone_pcoord(const int izone) {
    assert(izone >= 0);
    assert(izone <= 8);
    return zone_pcoord[izone];
  }
  int get_zone_pcoord(const int izone) const {
    assert(izone >= 0);
    assert(izone <= 8);
    return zone_pcoord[izone];
  }

  // Set the number of coordinates on one zone
  void set_zone_ncoord(const int izone, const int ncoord_val) {
    assert(izone >= 0);
    assert(izone < 8);
    zone_ncoord[izone] = ncoord_val;
    // Update zone_pcoord[]
    update_zone_pcoord();
  }

  // Set the number of coordinates to zero on all zones
  void clear_zone_ncoord() {
    for (int i=0;i < 8;i++) zone_ncoord[i] = 0;
    // Update zone_pcoord[]
    update_zone_pcoord();
  }

  // Return the total number of coordinates in all zones
  int get_ncoord_tot() {return zone_pcoord[8];};
  int get_ncoord_tot() const {return zone_pcoord[8];};

  // Return current node ID
  int get_mynode() {return mynode;}
  int get_mynode() const {return mynode;}

  int get_nx() {return nx;}
  int get_ny() {return ny;}
  int get_nz() {return nz;}
  int get_nx() const {return nx;}
  int get_ny() const {return ny;}
  int get_nz() const {return nz;}

  double get_inv_boxx() {return 1.0/boxx;}
  double get_inv_boxy() {return 1.0/boxy;}
  double get_inv_boxz() {return 1.0/boxz;}
  double get_inv_boxx() const {return 1.0/boxx;}
  double get_inv_boxy() const {return 1.0/boxy;}
  double get_inv_boxz() const {return 1.0/boxz;}

  // Return neighborlist cut-off
  double get_rnl() {return rnl;}
  double get_rnl() const {return rnl;}

  int get_homeix() {return homeix;}
  int get_homeiy() {return homeiy;}
  int get_homeiz() {return homeiz;}
  int get_homeix() const {return homeix;}
  int get_homeiy() const {return homeiy;}
  int get_homeiz() const {return homeiz;}

  // Returns the node index for box (ix, iy, iz)
  // NOTE: deals correctly with periodic boundary conditions
  int get_nodeind_pbc(const int ix, const int iy, const int iz);

  //
  // Builds global loc2glo mapping:
  // loc2glo_glo = mapping (size ncoord_glo)
  // nrecv       = number of coordinates we receive from each node     (size numnode)
  // precv       = exclusive cumulative sum of nrecv, used as postiion (size numnode)
  //
  void buildGlobal_loc2glo(int* loc2glo, int* loc2glo_glo, int* nrecv, int* precv);

  //
  // Combines data among all nodes using the global loc2glo mapping
  // xrecvbuf = temporary receive buffer (size ncoord_glo)
  // x        = send buffer (size ncoord)
  // xglo     = final global buffer (size ncoord_glo)
  //
  void combineData(int* loc2glo_glo, int* nrecv, int* precv,
		   double *xrecvbuf, double *x, double *xglo);

  bool checkNumGroups(std::vector<AtomGroupBase*>& atomGroupVector);
  bool checkGroup(AtomGroupBase& atomGroup, const int numTot);

  bool checkHeuristic(const bool heuristic);
  void copy_lohi_buf(double *buf);
};

#endif // DOMDEC_H
