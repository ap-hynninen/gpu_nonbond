#ifndef DOMDECRECIPCOMM_H
#define DOMDECRECIPCOMM_H

#include <vector>
#include <mpi.h>

class DomdecRecipComm {

 protected:
  // MPI communicators
  MPI_Comm comm_recip;
  MPI_Comm comm_direct_recip;
  
  int mynode;

  // Logical flags for determining the direct/recip role of this node
  bool isDirect;
  bool isRecip;

  // Number of coordinates / forces we are communicating
  std::vector<int> ncomm;

  // Position of coordinates /forces we are communicating
  std::vector<int> pcomm;

  // Recip node for this (direct) node
  std::vector<int> recip_nodes;

  // Direct node for this (recip) node
  std::vector<int> direct_nodes;

  // Number of coordinates stored in this class (zero for pure Direct nodes)
  int ncoord;

 public:
  DomdecRecipComm(MPI_Comm comm_recip, MPI_Comm comm_direct_recip, int mynode,
		  std::vector<int>& direct_nodes, std::vector<int>& recip_nodes);

  void comm_ncoord(const int ncoord_in);

  int get_num_recip() {return recip_nodes.size();}

  //virtual void send_coord(const float4* coord)=0;

  //virtual void send_force(const float4* coord)=0;

};

#endif // DOMDECRECIPCOMM_H
