#ifndef ATOMGROUP_H
#define ATOMGROUP_H

#include "AtomGroupBase.h"

//
// Template class for atom groups
//

template<typename T>
class AtomGroup : public AtomGroupBase {

 private:
  // Global group list, constant
  T* groupList;

 public:

  AtomGroup(const int numGroupList, T* h_groupList, const char* name) : 
    AtomGroupBase(T::size(), numGroupList, name) {
    groupList = NULL;
    if (numGroupList > 0) {
      allocate<T>(&groupList, numGroupList);
      copy_HtoD<T>(h_groupList, groupList, numGroupList);
    }
  }
  
  ~AtomGroup() {
    if (groupList != NULL) deallocate<T>(&groupList);
    if (table != NULL) deallocate<int>(&table);
  }  

  T* get_groupList() {return groupList;}

  void resizeTable(const int new_numTable) {
    reallocate<int>(&table, &lenTable, new_numTable, 1.2f);
  }

};

#endif // ATOMGROUP_H
