#ifndef DOMDECD2DCOMM_H
#define DOMDECD2DCOMM_H
#include "DomdecMPI.h"

class DomdecD2DComm {

private:

  // Number of boxes we communicate to in each direction
  int nx_comm, ny_comm, nz_comm;

  int y_recv_ncoord_tot;
  int *y_recv_ncoord;
  int *y_recv_pos;

  int y_send_ncoord_tot;
  int *y_send_ncoord;
  int *y_send_pos;

  int *y_send_node;
  int *y_recv_node;

  int y_recv_atomind_len;
  int *y_recv_atomind;

  int y_recv_buf_len;
  double *y_recv_buf;

  int xyz_tmp_len;
  double *xyz_tmp;

  //
  DomdecMPI *ydir;

public:

  DomdecD2DComm(int nx_comm, int ny_comm, int nz_comm); 
  ~DomdecD2DComm();

  void setup_comm(int *y_recv_node_in, int *y_send_node_in);
  void setup_atomind(int *y_recv_ncoord_in, int *h_y_recv_atomind_in,
		     int *y_send_ncoord_in, int *h_y_send_atomind_in);


};

#endif // DOMDECD2DCOMM_H
