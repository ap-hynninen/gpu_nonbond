#include "CudaDomdecRecipLooper.h"

void CudaDomdecRecipLooper::run() {

  bool done = false;
  while(!done) {
    /*
    recipComm.comm_ncoord();
    recipComm.comm_coord(xyzq.xyzq);
    
    recip.calc(domdec.get_inv_boxx(), domdec.get_inv_boxy(), domdec.get_inv_boxz(),
	       recipComm.get_coord(), recipComm.get_ncoord(),
	       calc_energy, calc_virial, force);
    */

    //recipComm.send_force(force);

  }

}
