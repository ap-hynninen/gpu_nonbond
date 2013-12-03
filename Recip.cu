
template <class T> class Recip {
private:

  // Size of the entire grid in real space
  int nfftx, nffty, nfftz;

  // B-spline order for this reciprocal calculation
  int order;

  // Number of nodes involved in the calculation
  int nnode;

  // Node number for this node: mynode=0...nnode-1
  int mynode;

  // Number of nodes in y and z directions
  int nnode_y, nnode_z;

public:

  void init(int nfftx, int nffty, int nfftz, int order, int nnode, int mynod) {

    

  }

  Recip(int nfftx, int nffty, int nfftz, int order, int nnode=1, int mynod=0) : 
    nfftx(nfftx), nffty(nffty), nfftz(nfftz), nnode(nnode), mynod(mynod)
  {
    
  }

  ~Recip() {
  }

};

//
// Explicit instances of Recip
//
template class Recip<long long int>;
