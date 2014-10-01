#ifndef DOMDEC_H
#define DOMDEC_H

//
// Base class for atom decompositors
//
class Domdec {

 protected:
  
  // Total global number of coordinates in the system
  int ncoord_glo;

  // Number of coordinates in each zone
  int zone_ncoord[8];
  int zone_pcoord[8];

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
  int* get_zone_pcoord() {return zone_pcoord;}
  const int* get_zone_pcoord() const {return zone_pcoord;}

  // Return the total number of coordinates in all zones
  int get_ncoord_tot() {return zone_pcoord[7];};
  int get_ncoord_tot() const {return zone_pcoord[7];};

  // Calculate zone_pcoord
  void update_zone_pcoord() {
    zone_pcoord[0] = zone_ncoord[0];
    for (int i=1;i < 8;i++) {
      zone_pcoord[i] = zone_pcoord[i-1] + zone_ncoord[i];
    }
  }

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
