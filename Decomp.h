#ifndef DECOMP_H
#define DECOMP_H

//
// Abstract base class for decompositors
//
class Decomp {

 protected:
  
  // Total global number of coordinates in the system
  int ncoord_glo;

  // Number of coordinates in this node
  int ncoord;

 public:

  int get_ncoord_glo() {
    return ncoord_glo;
  }

};

#endif // DECOMP_H
