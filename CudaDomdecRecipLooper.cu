#include "CudaDomdecRecipLooper.h"

void CudaDomdecRecipLooper::run() {

  while(true) {

    // Receive header and stop if the STOP signal was received
    if (!recipComm.recv_header()) break;

    // Re-allocate coordinate array if needed
    xyzq.realloc(recipComm.get_ncoord());

    // Re-allocate force array if needed
    reallocate<float3>(&force, &force_len, recipComm.get_ncoord(), 1.0f);

    // Receive coordinates from Direct nodes
    recipComm.recv_coord(xyzq.xyzq, stream);

    //xyzq.save("xyzq_recip.txt");

    assert(xyzq.xyzq == recipComm.get_coord_ptr());

    // Compute forces
    recip.calc(recipComm.get_inv_boxx(), recipComm.get_inv_boxy(), recipComm.get_inv_boxz(),
	       recipComm.get_coord_ptr(), recipComm.get_ncoord(),
	       recipComm.get_calc_energy(), recipComm.get_calc_virial(), force);
    //NOTE: this synchronization is done in recipComm.send_force()
    //cudaCheck(cudaStreamSynchronize(stream));

    //save_float3(recipComm.get_ncoord(), force, "force_recip.txt");

    // Send forces to Direct nodes
    recipComm.send_force(force, stream);

  }

}
