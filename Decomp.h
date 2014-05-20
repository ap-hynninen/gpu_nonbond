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

  // Total number of nodes
  int numnode;

  // This node index (=0...numnode-1)
  int mynode;

 public:

  // Return the global total number of coordinates
  int get_ncoord_glo() {return ncoord_glo;}

  // Return the number of coordinates in this node
  int get_ncoord() {return ncoord;}

  // Return number of nodes
  int get_numnode() {return numnode;}

};

#endif // DECOMP_H
