#ifndef CUDADOMDECGROUPS_H
#define CUDADOMDECGROUPS_H

#include <vector>
#include <map>
#include <algorithm>
#include "Bonded_struct.h"
#include "CudaDomdec.h"
#include "CudaAtomGroup.h"

class CudaDomdecGroups {

 private:

  const CudaDomdec& domdec;
  
  // Atom group pointers in a map:
  // <id, atomgroup*>
  std::map<int, AtomGroupBase*> atomGroups;

  // Atom group pointers in a vector:
  std::vector<AtomGroupBase*> atomGroupVector;

  int** groupTable;
  std::vector<int*> h_groupTable;

  int* groupDataStart;
  int* groupData;
  int* groupTablePos;

  int* h_groupTablePos;

  bool tbl_upto_date;

  // Storage vector used for registering groups
  std::vector< std::vector<int> > regGroups;
  
 public:

  CudaDomdecGroups(const CudaDomdec& domdec);
  ~CudaDomdecGroups();

  std::vector<AtomGroupBase*>& get_atomGroupVector() {return atomGroupVector;}

  void beginGroups();

  //
  // Register groups.
  // h_groupList[] is the host version of atomGroup.groupList[]
  //
  template <typename T>
    void insertGroup(int id, CudaAtomGroup<T>& atomGroup, T* h_groupList) {
    assert(regGroups.size() == domdec.get_ncoord_glo());
    int type = atomGroups.size();
    int size = T::size();
    std::pair<std::map<int, AtomGroupBase*>::iterator, bool> ret =
      atomGroups.insert(std::pair<int, AtomGroupBase*>(id, &atomGroup));
    if (ret.second == false) {
      std::cout << "CudaDomdecGroups::insertGroup, group IDs must be unique" << std::endl;
      exit(1);
    }
    // Set group type
    atomGroup.set_type(type);
    // Loop through groups
    for (int i=0;i < atomGroup.get_numGroupList();i++) {
      // Get atoms that are in group
      std::vector<int> atoms;
      h_groupList[i].getAtoms(atoms);
      int t = *std::min_element(atoms.begin(), atoms.end());
      regGroups.at(t).push_back((size << 16) | type );
      regGroups.at(t).push_back(i);
      regGroups.at(t).insert( regGroups.at(t).end(), atoms.begin(), atoms.end() );
    }
  }

  void finishGroups();

  void buildGroupTables(cudaStream_t stream=0);
  void syncGroupTables(cudaStream_t stream=0);

  // Return group list.
  // NOTE: This is constant during the run
  template <typename T>
    T* getGroupList(const int id) {
    std::map<int, AtomGroupBase*>::iterator it = atomGroups.find(id);
    if (it == atomGroups.end()) return NULL;
    CudaAtomGroup<T>* p = dynamic_cast< CudaAtomGroup<T>* >( it->second );
    if (p == NULL) {
      std::cerr << "CudaDomdecGroups::get_group, dynamic_cast failed" << std::endl;
      exit(1);
    }
    return p->get_groupList();
  }

  // Return group table
  // NOTE: This changes at neighborlist update
  int* getGroupTable(const int id) {
    std::map<int, AtomGroupBase*>::iterator it = atomGroups.find(id);
    if (it == atomGroups.end()) return NULL;
    return it->second->get_table();
  }

  // Return number of entries in group table
  // NOTE: This changes at neighborlist update
  int getNumGroupTable(const int id) {
    std::map<int, AtomGroupBase*>::iterator it = atomGroups.find(id);
    if (it == atomGroups.end()) return 0;
    return it->second->get_numTable();
  }

};

#endif // CUDADOMDECGROUPS_H
