#undef SEEK_SET
#undef SEEK_CUR
#undef SEEK_END
#include <mpi.h>
#include <iostream>
#include "DomdecRecipComm.h"
#include "mpi_utils.h"

DomdecRecipComm::DomdecRecipComm(MPI_Comm comm_recip, MPI_Comm comm_direct_recip, int mynode,
				 std::vector<int>& direct_nodes, std::vector<int>& recip_nodes) :
  comm_recip(comm_recip), comm_direct_recip(comm_direct_recip),
  mynode(mynode), direct_nodes(direct_nodes), recip_nodes(recip_nodes) {
  
  if (recip_nodes.size() == 0) {
    std::cout << "DomdecRecipComm::DomdecRecipComm, Currently must have #recip == 1" << std::endl;
    exit(1);
  }
    
  if (recip_nodes.size() > 1) {
    std::cout << "DomdecRecipComm::DomdecRecipComm, Currently no support for #recip > 1" << std::endl;
    exit(1);
  }

  isRecip = false;
  for (int i=0;i < recip_nodes.size();i++)
    isRecip |= (mynode == recip_nodes.at(i));

  isDirect = false;
  for (int i=0;i < direct_nodes.size();i++)
    isDirect |= (mynode == direct_nodes.at(i));
    
  // Allocate and zero recv_ncoord
  if (isRecip) {
    ncomm.resize(direct_nodes.size(), 0);
    pcomm.resize(direct_nodes.size()+1, 0);
  } else {
    ncomm.resize(1, 0);
  }

  ncoord = 0;

}

//
// Communicate the number of coordinates
//
void DomdecRecipComm::comm_ncoord(const int ncoord_in) {

  const int TAG = 1;

  if (isDirect && !isRecip) {
    //------------------------------------------------
    // Pure Direct node => Send #coordinates to Recip
    //------------------------------------------------
    MPICheck(MPI_Send((void *)&ncoord_in, 1, MPI_INT, recip_nodes.at(0), TAG, comm_direct_recip));
    ncomm.at(0) = ncoord_in;
  }

  if (isRecip) {
    //------------------------------------------------
    // Recip node => Receive #coordinates from Direct
    //------------------------------------------------
    for (int i=0;i < direct_nodes.size();i++) {
      if (mynode != direct_nodes.at(i)) {
	MPICheck(MPI_Recv(&ncomm.at(i), 1, MPI_INT, direct_nodes.at(i), TAG,
			  comm_direct_recip, MPI_STATUS_IGNORE));
      } else {
	ncomm.at(i) = ncoord_in;
      }
    }
    // Calculate cumulative positions from ncomm
    pcomm.at(0) = 0;
    for (int i=1;i < direct_nodes.size()+1;i++) {
      pcomm.at(i) = pcomm.at(i-1) + ncomm.at(i-i);
    }
    // Total number of coordinates
    ncoord = pcomm.at(direct_nodes.size());
  } else {
    ncoord = 0;
  }


}
