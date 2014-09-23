#ifndef DOMDECRECIPCOMM_H
#define DOMDECRECIPCOMM_H

#include <cassert>
#include <vector>
#include <mpi.h>

class DomdecRecipComm {

 private:
  struct Header_t {
    double inv_boxx, inv_boxy, inv_boxz;
    int ncoord;
    bool calc_energy, calc_virial;
  };

  Header_t header;

  int calc_pcomm();

 protected:
  // MPI communicators
  MPI_Comm comm_recip;
  MPI_Comm comm_direct_recip;

  // This node ID
  int mynode;

  // Index of this node in direct_nodes -list
  int imynode;

  // Logical flags for determining the direct/recip role of this node
  bool isDirect;
  bool isRecip;
  
  // true if system has a pure Recip node
  // false if system has mixed Direct+Recip node
  bool hasPureRecip;

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

  void send_header(const int ncoord_in, const double inv_boxx, const double inv_boxy,
		   const double inv_boxz, const bool calc_energy,
		   const bool calc_virial);
  void send_ncoord(const int ncoord_in);
  void recv_ncoord(const int ncoord_in);

  void send_stop();

  bool recv_header();

  int get_num_recip() {return recip_nodes.size();}
  int get_num_direct() {return direct_nodes.size();}

  void read_header_asserts() {
    assert(hasPureRecip);
    assert(!isDirect);
    assert(isRecip);
  }

  bool get_isRecip() {return isRecip;}
  bool get_hasPureRecip() {return hasPureRecip;}
  int get_ncoord() {return ncoord;}

  double get_inv_boxx() {read_header_asserts();return header.inv_boxx;}
  double get_inv_boxy() {read_header_asserts();return header.inv_boxy;}
  double get_inv_boxz() {read_header_asserts();return header.inv_boxz;}
  bool get_calc_energy() {read_header_asserts();return header.calc_energy;}
  bool get_calc_virial() {read_header_asserts();return header.calc_virial;}

  //virtual void send_coord(const float4* coord)=0;

  //virtual void send_force(const float4* coord)=0;

};

#endif // DOMDECRECIPCOMM_H
