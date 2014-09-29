#include "CudaDomdecRecipLooper.h"

void CudaDomdecRecipLooper::run() {

  while(true) {

    // Receive header and stop if the STOP signal was received
    if (!recipComm.recv_header()) break;

    // Resize coordinate array if needed
    xyzq.resize(recipComm.get_ncoord());

    // Resize force array if needed
    reallocate<float3>(&force, &force_len, recipComm.get_ncoord(), 1.0f);

    // Receive coordinates from Direct nodes
    recipComm.recv_coord(xyzq.xyzq);

    // Compute forces
    recip.calc(recipComm.get_inv_boxx(), recipComm.get_inv_boxy(), recipComm.get_inv_boxz(),
	       recipComm.get_coord_ptr(), recipComm.get_ncoord(),
	       recipComm.get_calc_energy(), recipComm.get_calc_virial(), force);

    // Send forces to Direct nodes
    recipComm.send_force(force);

  }

}
