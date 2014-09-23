#include <iostream>
#include <algorithm>
#include "DomdecRecipComm.h"
#include "mpi_utils.h"

DomdecRecipComm::DomdecRecipComm(MPI_Comm comm_recip, MPI_Comm comm_direct_recip, int mynode,
				 std::vector<int>& direct_nodes, std::vector<int>& recip_nodes) :
  comm_recip(comm_recip), comm_direct_recip(comm_direct_recip),
  mynode(mynode), direct_nodes(direct_nodes), recip_nodes(recip_nodes) {
  
  if (direct_nodes.size() == 0 && recip_nodes.size() == 0) {
    std::cout << "DomdecRecipComm::DomdecRecipComm, Currently must have #recip == 1 or #direct == 1"
	      << std::endl;
    exit(1);
  }
    
  if (recip_nodes.size() > 1) {
    std::cout << "DomdecRecipComm::DomdecRecipComm, Currently no support for #recip > 1" << std::endl;
    exit(1);
  }

  isRecip = (std::find(recip_nodes.begin(), recip_nodes.end(), mynode) != recip_nodes.end());
  isDirect = (std::find(direct_nodes.begin(), direct_nodes.end(), mynode) != direct_nodes.end());

  std::vector<int> sorted1 = direct_nodes;
  std::vector<int> sorted2 = recip_nodes;
  std::sort(sorted1.begin(), sorted1.end());
  std::sort(sorted2.begin(), sorted2.end());

  std::vector<int> v(direct_nodes.size());
  std::vector<int>::iterator it = std::set_intersection(sorted1.begin(), sorted1.end(),
							sorted2.begin(), sorted2.end(), v.begin());
  // has Pure Recip node if intersection is zero
  hasPureRecip = ((it - v.begin()) == 0);

  imynode = -1;
  if (isDirect) {
    for (int i=0;i < direct_nodes.size();i++) {
      if (mynode == direct_nodes.at(i)) {
	if (imynode != -1) {
	  std::cout << "DomdecRecipComm::DomdecRecipComm, Error setting imynode (1)" << std::endl;
	  exit(1);
	}
	imynode = i;
      }
    }
    if (imynode == -1) {
      std::cout << "DomdecRecipComm::DomdecRecipComm, Error setting imynode (2)" << std::endl;
      exit(1);
    }
  }

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
// Send header
//
void DomdecRecipComm::send_header(const int ncoord_in, const double inv_boxx, const double inv_boxy,
				  const double inv_boxz, const bool calc_energy,
				  const bool calc_virial) {

  const int TAG = 1;

  //--------------------------------------------------
  // Contains pure Recip node => Full header required
  //--------------------------------------------------
  // Sanity check
  if (isRecip) {
    std::cout << "DomdecRecipComm::send_header, pure recip node should not come here!" << std::endl;
    exit(1);
  }
  header.inv_boxx    = inv_boxx;
  header.inv_boxy    = inv_boxy;
  header.inv_boxz    = inv_boxz;
  header.ncoord      = ncoord_in;
  header.calc_energy = calc_energy;
  header.calc_virial = calc_virial;
  MPICheck(MPI_Send((void *)&header, sizeof(Header_t), MPI_BYTE, recip_nodes.at(0), TAG,
		    comm_direct_recip));

}

//
// Send number of coordinates
//
void DomdecRecipComm::send_ncoord(const int ncoord_in) {

  const int TAG = 1;

  //----------------------------------------------------------------------------
  // All nodes are Direct nodes => #coordinates at neighborlist update required
  //----------------------------------------------------------------------------
  // All nodes are Direct and we have updated neighborlist => send #coordinates to Recip node
  // Send the number of coordinates to the Recip node
  if (mynode != recip_nodes.at(0)) {
    MPICheck(MPI_Send((void *)&ncoord_in, 1, MPI_INT, recip_nodes.at(0), TAG, comm_direct_recip));
    ncomm.at(0) = ncoord_in;
    ncoord = ncoord_in;
  }

}

//
// Send number of coordinates
//
void DomdecRecipComm::recv_ncoord(const int ncoord_in) {

  const int TAG = 1;

  //----------------------------------------------------------------------------
  // All nodes are Direct nodes => #coordinates at neighborlist update required
  //----------------------------------------------------------------------------
  // All nodes are Direct and we have updated neighborlist => send #coordinates to Recip node
  // Receive the number of coordinates from Direct nodes
  for (int i=0;i < direct_nodes.size();i++) {
    if (mynode != direct_nodes.at(i)) {
      MPICheck(MPI_Recv(&ncomm.at(i), 1, MPI_INT, direct_nodes.at(i), TAG,
			comm_direct_recip, MPI_STATUS_IGNORE));
    } else {
      ncomm.at(i) = ncoord_in;
    }
  }
  // Total number of coordinates
  ncoord = calc_pcomm();

}

//
// Send stop signal (negative #coordinates) to Recip node(s)
//
void DomdecRecipComm::send_stop() {
  if (isDirect && hasPureRecip) {
    send_header(-1, 1.0, 1.0, 1.0, false, false);
  }
}

//
// Receive header. Values can be read with get_inv_boxx, etc. functions
// Returns false if STOP signal was received
//
bool DomdecRecipComm::recv_header() {

  const int TAG = 1;

  // Sanity checks
  if (!hasPureRecip) {
    std::cout << "DomdecRecipComm::recv_header, only pure recip nodes can be here!" << std::endl;
    exit(1);
  }
  if (isDirect) {
    std::cout << "DomdecRecipComm::recv_header, direct node should not come here!" << std::endl;
    exit(1);
  }
  if (!isRecip) {
    std::cout << "DomdecRecipComm::recv_header, only recip nodes come here!" << std::endl;
    exit(1);
  }

  // Receive headers from Direct nodes
  for (int i=0;i < direct_nodes.size();i++) {
    MPICheck(MPI_Recv((void *)&header, sizeof(Header_t), MPI_BYTE, direct_nodes.at(i), TAG,
		      comm_direct_recip, MPI_STATUS_IGNORE));
    ncomm.at(i) = header.ncoord;
  }
  // Did we receive STOP from Direct node => return with false
  if (header.ncoord < 0) return false;
  // Total number of coordinates
  ncoord = calc_pcomm();
  
  return true;
}

//
// Calculate cumulative positions from ncomm to pcomm
// Returns the total number of coordinates
//
int DomdecRecipComm::calc_pcomm() {
  pcomm.at(0) = 0;
  for (int i=1;i < direct_nodes.size()+1;i++) {
    pcomm.at(i) = pcomm.at(i-1) + ncomm.at(i-i);
  }
  return pcomm.at(direct_nodes.size());
}

