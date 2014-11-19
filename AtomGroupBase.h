#ifndef ATOMGROUPBASE_H
#define ATOMGROUPBASE_H

#include <cassert>
#include <vector>
#include <string>

//
// Abstract base class for atom groups
//

class AtomGroupBase {

private:
  // Name, constant
  std::string name;

 protected:
  // Size of the group
  const int size;

  // Type of the group
  int type;

  // Number of entries in group list, constant
  const int numGroupList;

  // Group tables, change at every neighborlist build
  int numTable;
  int lenTable;
  int *table;

 public:

  AtomGroupBase(const int size, const int numGroupList, const char* name) : 
    size(size), numGroupList(numGroupList), name(name) {
    assert(numGroupList > 0);
    numTable = 0;
    lenTable = 0;
    table = NULL;
  }

  void set_numTable(const int numTable) {
    assert(numTable <= lenTable);
    this->numTable = numTable;
  }

  void set_type(const int type) {
    this->type = type;
  }

  int get_type() {return type;}

  int get_size() {return size;}

  int* get_table() {return table;}
  int get_numTable() {return numTable;}
  int get_numGroupList() {return numGroupList;}
  const char* get_name() {return name.c_str();}
  virtual void resizeTable(const int new_numTable) = 0;
  virtual void getGroupTableVec(std::vector<int>& tableVec) = 0;
  virtual void printGroup(const int i)=0;
};

#endif // ATOMGROUPBASE_H
