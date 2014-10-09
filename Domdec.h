#ifndef DOMDEC_H
#define DOMDEC_H

#include <cassert>

//
// Base class for atom decompositors
//
class Domdec {

private:
  // Number of coordinates in each zone
  // zone_ncoord[i] = number of coordinates in zone i
  // zone_pcoord[i] = zone i starting position in the coordinate arrays
  int zone_ncoord[8];
  int zone_pcoord[9];

  // Calculate zone_pcoord
  void update_zone_pcoord() {
    zone_pcoord[0] = 0;
    for (int i=1;i <= 8;i++) zone_pcoord[i] = zone_pcoord[i-1] + zone_ncoord[i-1];
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
	 int nx, int ny, int nz, int mynode) : ncoord_glo(ncoord_glo),
    boxx(boxx), boxy(boxy), boxz(boxz),
    rnl(rnl), nx(nx), ny(ny), nz(nz), numnode(nx*ny*nz),
    mynode(mynode) {

      // Setup (homeix, homeiy, homeiz)
      int m = mynode;
      homeiz = m/(nx*ny);
      m -= homeiz*(nx*ny);
      homeiy = m/nx;
      m -= homeiy*nx;
      homeix = m;

    }

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
  int get_nodeind_pbc(const int ix, const int iy, const int iz) {
    // ixt = 0...nx-1
    //int ixt = (ix + (abs(ix)/nx)*nx) % nx;
    //int iyt = (iy + (abs(iy)/ny)*ny) % ny;
    //int izt = (iz + (abs(iz)/nz)*nz) % nz;
    int ixt = ix;
    while (ixt < 0) ixt += nx;
    while (ixt >= nx) ixt -= nx;
    int iyt = iy;
    while (iyt < 0) iyt += ny;
    while (iyt >= ny) iyt -= ny;
    int izt = iz;
    while (izt < 0) izt += nz;
    while (izt >= nz) izt -= nz;

    return ixt + iyt*nx + izt*nx*ny;
  }

};

#endif // DOMDEC_H
