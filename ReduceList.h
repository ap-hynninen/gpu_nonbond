#ifndef REDUCELIST_H
#define REDUCELIST_H

#include <iostream>
/*
#include <algorithm>
#include <vector>
#include <map>
*/
#include <unordered_map>

template <int n>
struct doublen {
  double x[n];

  // Define == operator for doublen<n>
  bool operator==(const doublen<n>& rhs ) const {
    bool res = true;
    for (int i=0;i < n;i++) res = res && (x[i] == rhs.x[i]);
    return res;
  }

  /*
  // Define < operator for doublen<n>
  bool operator<(const doublen<n>& rhs ) const {

    std::vector<double> vec_this(x, x+n);
    std::vector<double> vec_rhs(rhs.x, rhs.x+n);

    std::sort(vec_this.begin(), vec_this.end());
    std::sort(vec_rhs.begin(), vec_rhs.end());

    bool res = true;
    for (int i=0;i < n;i++) res = res && (vec_this[i] < vec_rhs[i]);
    return res;
  }  
  */

};

// Define std::hash< doublen<n> >() -operator
namespace std {
  template<int n>
  struct hash< doublen<n> > {
    std::size_t operator()(const doublen<n>& k) const {
      std::size_t hash = 0;
      for (int i=0;i < n;i++) hash ^= std::hash<double>()(k.x[i]);
      return hash;
    }
  };
}

//
// Reduces a list of floats or doubles into an indexed list
//

template<typename T>
class ReduceList {

private:

public:

  //
  // Class creator
  //
  ReduceList() {
  }
  
  //
  // Class destructor
  //
  ~ReduceList() {
  }

  //
  //
  //
  void reduce(int nlist, T* list,
	      int *nredlist, T **redlist, int **indlist) {
    
    std::unordered_map<T, int> map;

    std::cout << "nlist=" << nlist << std::endl;

    // Build map
    for (int ilist=0;ilist < nlist;ilist++) {
      map.insert(std::make_pair<T, int>(list[ilist],1));
    }

    *nredlist = map.size();
    *redlist = new T[*nredlist];
    *indlist = new int[nlist];

    std::cout << "reduced size = " << *nredlist << std::endl;

    // Define index for each entry and build reduced list
    int i = 0;
    for (auto it = map.begin(); it != map.end();it++) {
      (*redlist)[i] = it->first;
      it->second = i++;
    }

    // Build index list
    for (int ilist=0;ilist < nlist;ilist++) {
      (*indlist)[ilist] = map.find(list[ilist])->second;
    }

  }

};


#endif //REDUCELIST_H

