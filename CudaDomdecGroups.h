#ifndef CUDADOMDECGROUPS_H
#define CUDADOMDECGROUPS_H

#include <vector>
#include <map>
#include <algorithm>
#include "Bonded_struct.h"
#include "CudaDomdec.h"
#include "AtomGroup.h"

class CudaDomdecGroups {

 private:

  const CudaDomdec& domdec;
  
  // Atom groups
  std::map<int, AtomGroupBase*> atomGroups;

  int** groupTable;
  std::vector<int*> h_groupTable;

  int* groupDataStart;
  int* groupData;
  int* groupTablePos;

  int* h_groupTablePos;

  bool tbl_upto_date;

  // Storage vector used for registering groups
  //std::vector< std::vector<  std::pair<int, AtomGroupBase*> > > regGroups;
  std::vector< std::vector<int> > regGroups;
  
 public:

  CudaDomdecGroups(const CudaDomdec& domdec);
  ~CudaDomdecGroups();

  void beginGroups();

  //
  // Register groups.
  // h_groupList[] is the host version of atomGroup.groupList[]
  //
  template <typename T>
    void insertGroup(int id, AtomGroup<T>& atomGroup, T* h_groupList) {
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
      // Add group to the smallest atom index t
      regGroups.at(t).push_back((size << 16) | type );
      regGroups.at(t).push_back(i);
      regGroups.at(t).insert( regGroups.at(t).end(), atoms.begin(), atoms.end() );
    }
  }

  void finishGroups();

  void build_tbl(cudaStream_t stream=0);

  template <typename T>
    T* getGroupList(const int id) {
    std::map<int, AtomGroupBase*>::iterator it = atomGroups.find(id);
    if (it == atomGroups.end()) return NULL;
    AtomGroup<T>* p = dynamic_cast< AtomGroup<T>* >( it->second );
    if (p == NULL) {
      std::cerr << "CudaDomdecGroups::get_group, dynamic_cast failed" << std::endl;
      exit(1);
    }
    return p->get_groupList();
  }

  int* getGroupTable(const int id) {
    std::map<int, AtomGroupBase*>::iterator it = atomGroups.find(id);
    if (it == atomGroups.end()) return NULL;
    return it->second->get_table();
  }

  int getNumGroupTable(const int id) {
    std::map<int, AtomGroupBase*>::iterator it = atomGroups.find(id);
    if (it == atomGroups.end()) return 0;
    return it->second->get_numTable();
  }

};

#endif // CUDADOMDECGROUPS_H
