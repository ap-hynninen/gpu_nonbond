//
// Domdec groups base class
//
#include <vector>
#include "Bonded_struct.h"

class DomdecGroups {

 public:
  DomdecGroups() {}
  ~DomdecGroups() {}

  std::vector< std::vector<int> > buildMolecules(const int ncoord, const int numBond, const bond_t* bond);
  
};
