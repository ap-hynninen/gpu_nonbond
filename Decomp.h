#ifndef DECOMP_H
#define DECOMP_H

//
// Abstract base class for decompositors
//
class Decomp {

 protected:
  
  // Total number of coordinates in the system
  int ncoord_tot;

  // Number of coordinates in this node
  int ncoord;

 public:

  int get_ncoord_tot() {
    return ncoord_tot;
  }

};

#endif // DECOMP_H
