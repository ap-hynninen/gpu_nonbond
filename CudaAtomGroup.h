#ifndef CUDAATOMGROUP_H
#define CUDAATOMGROUP_H

#include "AtomGroupBase.h"

//
// Cuda implementation of base class virtual methods
//
class CudaAtomGroupBase : public AtomGroupBase {

 public:

  CudaAtomGroupBase(const int size, const int numGroupList, const char* name) :
    AtomGroupBase(size, numGroupList, name) {}

  // Implementation of virtual methods
  void resizeTable(const int new_numTable) {
    reallocate<int>(&table, &lenTable, new_numTable, 1.2f);
  }

  void getGroupTableVec(std::vector<int>& tableVec) {
    tableVec.resize(this->numTable);
    copy_DtoH_sync<int>(this->table, tableVec.data(), this->numTable);
  }

};

//
// Template class for atom groups
//

template<typename T>
class CudaAtomGroup : public CudaAtomGroupBase {

 private:
  // Global group list, constant
  T* groupList;

 public:

  CudaAtomGroup(const int numGroupList, T* h_groupList, const char* name) : 
    CudaAtomGroupBase(T::size(), numGroupList, name) {
    groupList = NULL;
    if (numGroupList > 0) {
      allocate<T>(&groupList, numGroupList);
      copy_HtoD_sync<T>(h_groupList, groupList, numGroupList);
    }
  }
  
  ~CudaAtomGroup() {
    if (groupList != NULL) deallocate<T>(&groupList);
    if (table != NULL) deallocate<int>(&table);
  }  

  T* get_groupList() {return groupList;}

  void printGroup(const int i) {
    T group;
    copy_DtoH_sync<T>(&groupList[i], &group, 1);
    group.printAtoms();
  }

};

#endif // CUDAATOMGROUP_H
