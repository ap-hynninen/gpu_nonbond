#include <iostream>
#include "DomdecGroups.h"

std::vector< std::vector<int> > DomdecGroups::buildMolecules(const int ncoord, const int numBond, const bond_t* bond) {

  /*
  
  // For each atom, "bonds" contains a list of bonded atoms
  std::vector< std::vector<int> > bonds(ncoord);
  for (int ibond=0;ibond < numBond;ibond++) {
    bonds.at(bond[ibond].i).push_back(bond[ibond].j);
    bonds.at(bond[ibond].j).push_back(bond[ibond].i);
  }

  // Molecule index for each atom (-1 = no molecule assigned)
  std::vector<int> molIndex(domdec.get_ncoord_glo(), -1);
  int nmol = 0;
  for (int i=0;i < domdec.get_ncoord_glo();i++) {
    if (molIndex.at(i) != -1) {
      // No molecule assigned to this atom => Create a new molecule
      // List of atoms in this molecule
      std::vector<int> atomList;
      // List of 
      std::vector<int> neighList;
      // Put original atom into the list
      atomList.push_back(i);

      while (atomList.size() > 0) {
	// Assign atom j into current molecule
	int j = atomList.back();
	molIndex.at(j) = nmol;
	// Go through atom j neighbors
	if (regGroups.at(j).
      }
      

      nmol++;
    }
  } 

  */

  std::cerr << "DomdecGroups::buildMolecules, not implemented" << std::endl;
  exit(1);
  
  std::vector< std::vector<int> > molecules;
  
  return molecules;
}
