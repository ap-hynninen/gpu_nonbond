#include <stdio.h>
#include <cassert>
#include "CudaPMEForcefield.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

__global__ void heuristic_check_kernel(const int ncoord,
				       const double* __restrict__ x,
				       const double* __restrict__ y,
				       const double* __restrict__ z,
				       const double* __restrict__ ref_x,
				       const double* __restrict__ ref_y,
				       const double* __restrict__ ref_z,
				       const float rsq_limit,
				       int* global_flag) {
  // Required shared memory:
  // blockDim.x/warpsize*sizeof(int)
  extern __shared__ int sh_flag[];
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int sh_flag_size = blockDim.x/warpsize;

  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
  if (tid < ncoord) {
    dx = (float)(x[tid] - ref_x[tid]);
    dy = (float)(y[tid] - ref_y[tid]);
    dz = (float)(z[tid] - ref_z[tid]);
  }

  float rsq = dx*dx + dy*dy + dz*dz;
  // flag = 1 update is needed
  //      = 0 no update needed
  int flag = (rsq > rsq_limit);
  // Reduce flag, packed into bits.
  // NOTE: this assumes that warpsize <= 32
  sh_flag[threadIdx.x/warpsize] = (flag << (threadIdx.x % warpsize));
  __syncthreads();
  if (threadIdx.x < sh_flag_size) {
    for (int d=1;d < sh_flag_size;d *= 2) {
      int t = threadIdx.x + d;
      int flag_val = (t < sh_flag_size) ? sh_flag[t] : 0;
      __syncthreads();
      sh_flag[threadIdx.x] |= flag_val;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      int flag_val = sh_flag[0];
      atomicOr(global_flag, flag_val);
    }
  }

}

//############################################################################################
//############################################################################################
//############################################################################################

//
// Class creator
//
CudaPMEForcefield::CudaPMEForcefield(CudaDomdec& domdec, CudaDomdecGroups& domdecGroups,
				     const CudaTopExcl& topExcl,
				     const int nbondcoef, const float2 *h_bondcoef,
				     const int nureybcoef, const float2 *h_ureybcoef,
				     const int nanglecoef, const float2 *h_anglecoef,
				     const int ndihecoef, const float4 *h_dihecoef,
				     const int nimdihecoef, const float4 *h_imdihecoef,
				     const int ncmapcoef, const float2 *h_cmapcoef,
				     const double roff, const double ron,
				     const double kappa, const double e14fac,
				     const int vdw_model, const int elec_model,
				     const int nvdwparam, const float *h_vdwparam,
				     const float *h_vdwparam14,
				     const int *h_glo_vdwtype, const float *h_glo_q,
				     CudaDomdecRecip* recip, CudaDomdecRecipComm& recipComm) : 
  domdec(domdec), recip(recip), domdecGroups(domdecGroups), 
  nlist(topExcl, domdec.get_nx(), domdec.get_ny(), domdec.get_nz()),
  recipComm(recipComm), kappa(kappa), recip_force_len(0), recip_force(NULL) {

  // Create streams
  cudaCheck(cudaStreamCreateWithPriority(&direct_stream[0], cudaStreamNonBlocking, low_stream_priority()));
  cudaCheck(cudaStreamCreateWithPriority(&direct_stream[1], cudaStreamNonBlocking, high_stream_priority()));
  cudaCheck(cudaStreamCreateWithFlags(&recip_stream, cudaStreamNonBlocking));
  cudaCheck(cudaStreamCreateWithFlags(&in14_stream, cudaStreamNonBlocking));
  cudaCheck(cudaStreamCreateWithFlags(&bonded_stream, cudaStreamNonBlocking));

  // Create events
  cudaCheck(cudaEventCreate(&done_direct_event[0]));
  cudaCheck(cudaEventCreate(&done_direct_event[1]));
  cudaCheck(cudaEventCreate(&done_recip_event));
  cudaCheck(cudaEventCreate(&done_in14_event));
  cudaCheck(cudaEventCreate(&done_bonded_event));
  cudaCheck(cudaEventCreate(&done_force_clear_event));
  cudaCheck(cudaEventCreate(&xyzq_ready_event[0]));
  cudaCheck(cudaEventCreate(&xyzq_ready_event[1]));
  cudaCheck(cudaEventCreate(&recip_coord_ready_event));
  cudaCheck(cudaEventCreate(&setup_bond_done_event));
  cudaCheck(cudaEventCreate(&setup_nonbond_done_event));
  cudaCheck(cudaEventCreate(&setup_14_done_event));

  // Set energy term flags
  calc_bond = true;
  calc_ureyb = true;
  calc_angle = true;
  calc_dihe = true;
  calc_imdihe = true;
  calc_cmap = true;

  // Neighborlists
  std::vector<int> numIntZone(8, 0);
  std::vector< std::vector<int> > intZones(8, std::vector<int>() );
  // Create I vs. I interaction
  numIntZone.at(0) = 1;
  intZones.at(0).push_back(Domdec::I);
  nlist.registerList(numIntZone, intZones);
  if (domdec.get_numnode() > 1) {
    // NOTE: getImportIntZones will clear contents of numIntZone & intZones
    domdec.getImportIntZones(numIntZone, intZones);
    nlist.registerList(numIntZone, intZones);
  }

  // Bonded coefficients
  bonded.setup_coef(nbondcoef, h_bondcoef, nureybcoef, h_ureybcoef,
		    nanglecoef, h_anglecoef, ndihecoef, h_dihecoef,
		    nimdihecoef, h_imdihecoef, ncmapcoef, h_cmapcoef);
  
  // Direct non-bonded interactions
  setup_direct_nonbonded(roff, ron, kappa, e14fac, vdw_model, elec_model,
			 nvdwparam, h_vdwparam, h_vdwparam14, h_glo_vdwtype);

  // Set stream for reciprocal calculation
  if (recip != NULL) recip->set_stream(recip_stream);

  // Copy charges
  allocate<float>(&glo_q, domdec.get_ncoord_glo());
  copy_HtoD<float>(h_glo_q, glo_q, domdec.get_ncoord_glo());

  allocate<int>(&d_heuristic_flag, 1);
  allocate_host<int>(&h_heuristic_flag, 1);

}

//
// Class destructor
//
CudaPMEForcefield::~CudaPMEForcefield() {
  deallocate<int>(&d_heuristic_flag);
  deallocate_host<int>(&h_heuristic_flag);
  deallocate<float>(&glo_q);
  deallocate<int>(&glo_vdwtype);
  if (recip_force != NULL) deallocate<float3>(&recip_force);
  // Destroy streams
  cudaCheck(cudaStreamDestroy(direct_stream[0]));
  cudaCheck(cudaStreamDestroy(direct_stream[1]));
  cudaCheck(cudaStreamDestroy(recip_stream));
  cudaCheck(cudaStreamDestroy(in14_stream));
  cudaCheck(cudaStreamDestroy(bonded_stream));
  // Destroy events
  cudaCheck(cudaEventDestroy(done_direct_event[0]));
  cudaCheck(cudaEventDestroy(done_direct_event[1]));
  cudaCheck(cudaEventDestroy(done_recip_event));
  cudaCheck(cudaEventDestroy(done_in14_event));
  cudaCheck(cudaEventDestroy(done_bonded_event));
  cudaCheck(cudaEventDestroy(done_force_clear_event));
  cudaCheck(cudaEventDestroy(xyzq_ready_event[0]));
  cudaCheck(cudaEventDestroy(xyzq_ready_event[1]));
  cudaCheck(cudaEventDestroy(recip_coord_ready_event));
  cudaCheck(cudaEventDestroy(setup_bond_done_event));
  cudaCheck(cudaEventDestroy(setup_nonbond_done_event));
  cudaCheck(cudaEventDestroy(setup_14_done_event));
}

//
// Setup direct non-bonded interactions.
//
void CudaPMEForcefield::setup_direct_nonbonded(const double roff, const double ron,
					       const double kappa, const double e14fac,
					       const int vdw_model, const int elec_model,
					       const int nvdwparam, const float *h_vdwparam,
					       const float *h_vdwparam14, const int *h_glo_vdwtype) {

  this->roff = roff;
  this->ron = ron;

  dir.setup(domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(), kappa, roff, ron,
	    e14fac, vdw_model, elec_model);

  dir.set_vdwparam(nvdwparam, h_vdwparam);
  dir.set_vdwparam14(nvdwparam, h_vdwparam14);

  allocate<int>(&glo_vdwtype, domdec.get_ncoord_glo());
  copy_HtoD<int>(h_glo_vdwtype, glo_vdwtype, domdec.get_ncoord_glo());
}

int nstep=0;

//
// Calculate forces
//
void CudaPMEForcefield::calc(const bool calc_energy, const bool calc_virial,
			     cudaXYZ<double>& coord, cudaXYZ<double>& prev_step, 
			     Force<long long int>& force, cudaStream_t stream) {

  // Check for neighborlist heuristic update
  if (heuristic_check(coord, stream)) {
    neighborlist_updated = true;

    if (domdec.get_mynode() == 0) fprintf(stderr,"Building neighborlist %d\n",nstep);

    // Update homezone coordinates (coord) and step vector (prev_step)
    // NOTE: Builds domdec.loc2glo
    domdec.update_homezone(coord, prev_step, stream);

    // ... (here we can sort & build neighborlist for local coordinates) ...

    // Communicate coordinates
    // NOTE: Builds rest of domdec.loc2glo and domdec.xyz_shift
    domdec.comm_coord(coord, true, stream);

    fprintf(stderr,"%d: domdec.get_ncoord()=%d domdec.get_ncoord_tot()=%d\n",
	    domdec.get_mynode(),domdec.get_ncoord(),domdec.get_ncoord_tot());

    // Re-allocate ref_coord and force
    ref_coord.realloc(domdec.get_ncoord_tot());
    force.realloc(domdec.get_ncoord_tot());

    // Clear energy and virial variables
    // NOTE: done_force_clear_event also waits for the energy & virial to clear
    if (calc_energy || calc_virial) {
      dir.clear_energy_virial(direct_stream[0]);
    }

    // Clear forces
    force.clear(direct_stream[0]);
    cudaCheck(cudaEventRecord(done_force_clear_event, direct_stream[0]));

    // Re-allocate (xyzq, xyzq_copy)
    xyzq.realloc(domdec.get_ncoord_tot());
    xyzq_copy.realloc(domdec.get_ncoord_tot());

    // Copy: coord => xyzq_copy
    // NOTE: coord and xyz_shift are already in the order determined by domdec.loc2glo,
    //       however, glo_q is in the original global order.
    xyzq_copy.set_xyzq(coord, glo_q, domdec.get_loc2glo_ptr(), domdec.get_xyz_shift(),
		       domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(), stream);

    //nlist.set_test(true);
    // Sort coordinates
    // NOTE: Builds domdec.loc2glo and nlist->glo2loc
    nlist.sort(0, domdec.get_zone_pcoord(), xyzq_copy.xyzq, xyzq.xyzq, domdec.get_loc2glo_ptr(), stream);
    cudaCheck(cudaEventRecord(xyzq_ready_event[0], stream));

    // Build neighborlist
    nlist.build(0, domdec.get_zone_pcoord(), domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(),
		domdec.get_rnl(), xyzq.xyzq, domdec.get_loc2glo_ptr(), stream);

    if (nlist.getNumList() > 1) {
      nlist.sort(1, domdec.get_zone_pcoord(), xyzq_copy.xyzq, xyzq.xyzq, domdec.get_loc2glo_ptr(), stream);
      cudaCheck(cudaEventRecord(xyzq_ready_event[1], stream));
      nlist.build(1, domdec.get_zone_pcoord(), domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(),
		  domdec.get_rnl(), xyzq.xyzq, domdec.get_loc2glo_ptr(), stream);
    }

    // Build bonded tables
    domdecGroups.buildGroupTables(stream);
    domdecGroups.syncGroupTables(stream);

    // Check the total number of groups
    if (!domdec.checkNumGroups(domdecGroups.get_atomGroupVector())) exit(1);
    
    // Setup bonded interaction lists
    bonded.setup_list(xyzq.xyzq, domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(),
		      nlist.get_glo2loc(),
		      domdecGroups.getNumGroupTable(BOND), domdecGroups.getGroupTable(BOND),
		      domdecGroups.getGroupList<bond_t>(BOND),
		      domdecGroups.getNumGroupTable(UREYB), domdecGroups.getGroupTable(UREYB),
		      domdecGroups.getGroupList<bond_t>(UREYB),
		      domdecGroups.getNumGroupTable(ANGLE), domdecGroups.getGroupTable(ANGLE),
		      domdecGroups.getGroupList<angle_t>(ANGLE),
		      domdecGroups.getNumGroupTable(DIHE), domdecGroups.getGroupTable(DIHE),
		      domdecGroups.getGroupList<dihe_t>(DIHE),
		      domdecGroups.getNumGroupTable(IMDIHE), domdecGroups.getGroupTable(IMDIHE),
		      domdecGroups.getGroupList<dihe_t>(IMDIHE),
		      domdecGroups.getNumGroupTable(CMAP), domdecGroups.getGroupTable(CMAP),
		      domdecGroups.getGroupList<cmap_t>(CMAP), stream);
    cudaCheck(cudaEventRecord(setup_bond_done_event, stream));

    // Set vdwtype for Direct non-bonded interactions
    dir.set_vdwtype(domdec.get_ncoord_tot(), glo_vdwtype, domdec.get_loc2glo_ptr(), stream);
    cudaCheck(cudaEventRecord(setup_nonbond_done_event, stream));

    // Setup 1-4 interaction lists
    dir.set_14_list(xyzq.xyzq, domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(),
		    nlist.get_glo2loc(),
		    domdecGroups.getNumGroupTable(IN14), domdecGroups.getGroupTable(IN14),
		    domdecGroups.getGroupList<xx14_t>(IN14),
		    domdecGroups.getNumGroupTable(EX14), domdecGroups.getGroupTable(EX14),
		    domdecGroups.getGroupList<xx14_t>(EX14), stream);
    cudaCheck(cudaEventRecord(setup_14_done_event, stream));

    // Re-order prev_step vector, using ref_coord as temporary storage
    // NOTE: We only re-order up to homezone
    domdec.reorder_coord(domdec.get_ncoord(), prev_step, ref_coord, nlist.get_ind_sorted(), stream);

    // Re-order coordinates (coord), using ref_coord as temporary storage
    // NOTE: We re-order all coordinates, homezone+import. This is done so that coord is
    //       correctly set up e.g. for constraint communication
    domdec.reorder_coord(domdec.get_ncoord_tot(), coord, ref_coord, nlist.get_ind_sorted(), stream);

    // NOTE: Now ref_coord contains the correct current coordinates that are then used in
    //       the heuristic check in the next step

    // Update and re-order communication buffers
    domdec.comm_update(nlist.get_glo2loc(), stream);
    //domdec.test_comm_coord(nlist.get_glo2loc(), coord);

    // Wait for non-bonded setup to finish
    cudaCheck(cudaStreamWaitEvent(direct_stream[0], setup_nonbond_done_event, 0));
    cudaCheck(cudaStreamWaitEvent(direct_stream[0], xyzq_ready_event[0], 0));
    // Direct non-bonded force (homebox)
    dir.calc_force(xyzq.xyzq, nlist.getBuilder(0), calc_energy, calc_virial, force.stride(), force.xyz(),
		   direct_stream[0]);
    cudaCheck(cudaEventRecord(done_direct_event[0], direct_stream[0]));

    if (nlist.getNumList() > 1) {
      // Wait for force clear finish
      cudaCheck(cudaStreamWaitEvent(direct_stream[1], done_force_clear_event, 0));
      // Wait for non-bonded setup to finish
      cudaCheck(cudaStreamWaitEvent(direct_stream[1], setup_nonbond_done_event, 0));
      cudaCheck(cudaStreamWaitEvent(direct_stream[1], xyzq_ready_event[1], 0));
      // Direct non-bonded force (import)
      dir.calc_force(xyzq.xyzq, nlist.getBuilder(1), calc_energy, calc_virial, force.stride(), force.xyz(),
		     direct_stream[1]);
      cudaCheck(cudaEventRecord(done_direct_event[1], direct_stream[1]));
    }

  } else {
    neighborlist_updated = false;

    // Clear energy and virial variables
    // NOTE: done_force_clear_event also waits for the energy & virial to clear
    if (calc_energy || calc_virial) {
      dir.clear_energy_virial(direct_stream[0]);
    }

    // Clear forces
    force.clear(direct_stream[0]);
    cudaCheck(cudaEventRecord(done_force_clear_event, direct_stream[0]));

    // Copy local coordinates to xyzq -array
    xyzq.set_xyz(coord, 0, domdec.get_ncoord()-1, domdec.get_xyz_shift(),
		 domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(), direct_stream[0]);
    // Record event "xyzq local coordinates are ready"
    cudaCheck(cudaEventRecord(xyzq_ready_event[0], direct_stream[0]));
    // Direct non-bonded force (homebox)
    dir.calc_force(xyzq.xyzq, nlist.getBuilder(0), calc_energy, calc_virial, force.stride(), force.xyz(),
		   direct_stream[0]);
    cudaCheck(cudaEventRecord(done_direct_event[0], direct_stream[0]));

    // Communicate coordinates between direct nodes
    domdec.comm_coord(coord, false, direct_stream[1]);

    // Copy import volume coordinates to xyzq -array
    xyzq.set_xyz(coord, domdec.get_ncoord(), domdec.get_ncoord_tot()-1, domdec.get_xyz_shift(),
		 domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(), direct_stream[1]);
    // Record event "xyzq array is ready"
    cudaCheck(cudaEventRecord(xyzq_ready_event[1], direct_stream[1]));
    if (nlist.getNumList() > 1) {
      // Wait for force clear finish
      cudaCheck(cudaStreamWaitEvent(direct_stream[1], done_force_clear_event, 0));
      // Direct non-bonded force (import)
      dir.calc_force(xyzq.xyzq, nlist.getBuilder(1), calc_energy, calc_virial, force.stride(), force.xyz(),
		     direct_stream[1]);
      cudaCheck(cudaEventRecord(done_direct_event[1], direct_stream[1]));
    }
  }

  nstep++;

  //-------------------------------------------------------------------------------

  bool do_recipcomm = recipComm.get_hasPureRecip() || 
    (recipComm.get_num_recip() > 0 && recipComm.get_num_direct() > 1);

  if (do_recipcomm) {
    if (recipComm.get_isRecip() && recip == NULL) {
      std::cout << "CudaPMEForcefield::calc, missing recip object" << std::endl;
      exit(1);
    }
    //-------------------------------------
    // Send coordinates to recip node(s)
    //-------------------------------------
    // Send header
    if (recipComm.get_hasPureRecip()) {
      recipComm.send_header(domdec.get_ncoord(), domdec.get_inv_boxx(), domdec.get_inv_boxy(),
			    domdec.get_inv_boxz(), calc_energy, calc_virial);
    } else if (neighborlist_updated) {
      if (recipComm.get_isRecip()) {
	recipComm.recv_ncoord(domdec.get_ncoord());
      } else {
	recipComm.send_ncoord(domdec.get_ncoord());
      }
    }
    if (recipComm.get_isRecip() && recipComm.get_num_recip() == 1) {
      assert(recipComm.get_ncoord() == domdec.get_ncoord_glo());
    }
    // Re-allocate recip_xyzq and recip_force if needed
    if (recipComm.get_isRecip() && recipComm.get_num_direct() > 1) {
      recip_xyzq.realloc(recipComm.get_ncoord());
    }
    reallocate<float3>(&recip_force, &recip_force_len, recipComm.get_ncoord(), 1.0f);
    // Send coordinates
    cudaCheck(cudaStreamWaitEvent(stream, xyzq_ready_event[0], 0));
    recipComm.send_coord(xyzq.xyzq, stream);
    // Receive coordinates
    if (recipComm.get_isRecip()) recipComm.recv_coord(recip_xyzq.xyzq, stream);
    //-------------------------------------
    cudaCheck(cudaEventRecord(recip_coord_ready_event, stream));
  }

  // --------------- Use of stream "stream" stops here --------------

  // Clear energy and virial variables
  if (calc_energy || calc_virial) {
    bonded.clear_energy_virial(bonded_stream);
    if (recipComm.get_isRecip()) recip->clear_energy_virial();
  }

  // Wait for xyzq coordinates to be ready
  //cudaCheck(cudaStreamWaitEvent(direct_stream[0], xyzq_ready_event[0], 0));
  //cudaCheck(cudaStreamWaitEvent(direct_stream[0], setup_nonbond_done_event, 0));
  // .... (here we'll calculate the local forces) ...
  //cudaCheck(cudaStreamWaitEvent(direct_stream[1], xyzq_ready_event[1], 0));
  //cudaCheck(cudaStreamWaitEvent(direct_stream[1], setup_nonbond_done_event, 0));

  /*
  // Direct non-bonded force
  dir.calc_force(xyzq.xyzq, nlist.getBuilder(0), calc_energy, calc_virial, force.stride(), force.xyz(),
  		 direct_stream[0]);
  cudaCheck(cudaEventRecord(done_direct_event[0], direct_stream[0]));

  if (nlist.getNumList() > 1) {
    dir.calc_force(xyzq.xyzq, nlist.getBuilder(1), calc_energy, calc_virial, force.stride(), force.xyz(),
    		   direct_stream[1]);
    cudaCheck(cudaEventRecord(done_direct_event[1], direct_stream[1]));
  }
  */

  // 1-4 interactions
  // NOTE: we make GPU wait until force.cleap() is done
  cudaCheck(cudaStreamWaitEvent(in14_stream, xyzq_ready_event[1], 0));
  cudaCheck(cudaStreamWaitEvent(in14_stream, done_force_clear_event, 0));
  cudaCheck(cudaStreamWaitEvent(in14_stream, setup_14_done_event, 0));
  dir.calc_14_force(xyzq.xyzq, calc_energy, calc_virial, force.stride(), force.xyz(),
  		    in14_stream);
  cudaCheck(cudaEventRecord(done_in14_event, in14_stream));

  // Bonded forces
  cudaCheck(cudaStreamWaitEvent(bonded_stream, xyzq_ready_event[1], 0));
  cudaCheck(cudaStreamWaitEvent(bonded_stream, done_force_clear_event, 0));
  cudaCheck(cudaStreamWaitEvent(bonded_stream, setup_bond_done_event, 0));
  bonded.calc_force(xyzq.xyzq, domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(),
    		    calc_energy, calc_virial, force.stride(), force.xyz(),
  		    calc_bond, calc_ureyb, calc_angle, calc_dihe, calc_imdihe, calc_cmap,
  		    bonded_stream);
  cudaCheck(cudaEventRecord(done_bonded_event, bonded_stream));

  // Reciprocal force (Only reciprocal nodes calculate this)
  if (recipComm.get_isRecip()) {
    if (recipComm.get_num_recip() == 1) {
      if (recipComm.get_num_direct() == 1) {
	// Single node that is a Direct+Recip node => add to total force and be done
	cudaCheck(cudaStreamWaitEvent(recip_stream, xyzq_ready_event[0], 0));
	cudaCheck(cudaStreamWaitEvent(recip_stream, done_force_clear_event, 0));
	recip->calc(domdec.get_inv_boxx(), domdec.get_inv_boxy(), domdec.get_inv_boxz(),
		    xyzq.xyzq, xyzq.ncoord,
		    calc_energy, calc_virial, force);
      } else {
	cudaCheck(cudaStreamWaitEvent(recip_stream, recip_coord_ready_event, 0));
	recip->calc(domdec.get_inv_boxx(), domdec.get_inv_boxy(), domdec.get_inv_boxz(),
		    recipComm.get_coord_ptr(), recipComm.get_ncoord(),
		    calc_energy, calc_virial, recip_force);
      }
    } else if (recipComm.get_num_recip() > 1) {
      // For #recip > 1, we need another force buffer (force_recip) and then need to combine results
      // to the total force
      std::cout << "CudaPMEForcefield::calc, #recip > 1 not implemented yet" << std::endl;
      exit(1);
    } else {
      std::cout << "CudaPMEForcefield::calc, #nrecip = 0, but recip defined should not end up here"
		<< std::endl;
      exit(1);
    }
  }
  cudaCheck(cudaEventRecord(done_recip_event, recip_stream));

  // --------------- stream "stream" picks up here --------------

  // Make integrator stream "stream" wait until all computation is done
  // i.e. we are pinning all computation to this stream
  cudaCheck(cudaStreamWaitEvent(stream, done_in14_event, 0));
  cudaCheck(cudaStreamWaitEvent(stream, done_bonded_event, 0));
  cudaCheck(cudaStreamWaitEvent(stream, done_recip_event, 0));
  cudaCheck(cudaStreamWaitEvent(stream, done_direct_event[0], 0));
  cudaCheck(cudaStreamWaitEvent(stream, done_direct_event[1], 0));

  // Convert forces from FP to DP
  force.convert<double>(stream);

  // If energy or virial is requested, make CPU wait here for computation to finish
  if (calc_energy || calc_virial) {
    cudaCheck(cudaEventSynchronize(done_in14_event));
    cudaCheck(cudaEventSynchronize(done_bonded_event));
    if (recipComm.get_isRecip()) cudaCheck(cudaEventSynchronize(done_recip_event));
    cudaCheck(cudaEventSynchronize(done_direct_event[0]));
    cudaCheck(cudaEventSynchronize(done_direct_event[1]));
  }

  bonded.get_energy_virial(calc_energy, calc_virial,
			   &energy_bond, &energy_ureyb,
			   &energy_angle,
			   &energy_dihe, &energy_imdihe,
			   &energy_cmap,
			   sforce);

  dir.get_energy_virial(calc_energy, calc_virial,
			&energy_vdw, &energy_elec,
			&energy_excl, vir);

  if (recipComm.get_isRecip()) {
    recip->get_energy_virial(calc_energy, calc_virial, energy_ewksum, energy_ewself, vir);
  }

  // Communicate Direct-Direct
  // NOTE: Synchronization on stream is done in comm_force
  // NOTE2: Due to the GPU waits above, this implies that all the other streams have
  //        finished their computation as well
  domdec.comm_force(force, stream);

  if (do_recipcomm) {
    cudaCheck(cudaStreamSynchronize(recip_stream));
    // Communicate Direct-Recip forces
    if (recipComm.get_isRecip()) recipComm.send_force(recip_force, stream);
    recipComm.recv_force(recip_force, stream);
    // Add Recip force to the total force
    force.add<double>(recipComm.get_force_ptr(), domdec.get_ncoord(), stream);
  }

}

//
// Post-process force calculation. Used for array re-ordering after neighborlist search
// Updates holonomic constraint tables if neccessary
//
void CudaPMEForcefield::post_calc(const float *global_mass, float *mass, HoloConst* holoconst,
				  cudaStream_t stream) {

  if (neighborlist_updated) {
    // Re-order xyz_shift
    domdec.reorder_xyz_shift(nlist.get_ind_sorted(), stream);

    // Re-do mass array
    map_to_local_array<float>(domdec.get_ncoord(), domdec.get_loc2glo_ptr(), global_mass, mass, stream);

    if (holoconst != NULL) {
      holoconst->setup_list(nlist.get_glo2loc(),
			    domdecGroups.getNumGroupTable(PAIR), domdecGroups.getGroupTable(PAIR),
			    domdecGroups.getGroupList<bond_t>(PAIR),
			    domdecGroups.getNumGroupTable(TRIP), domdecGroups.getGroupTable(TRIP),
			    domdecGroups.getGroupList<angle_t>(TRIP),
			    domdecGroups.getNumGroupTable(QUAD), domdecGroups.getGroupTable(QUAD),
			    domdecGroups.getGroupList<dihe_t>(QUAD),
			    domdecGroups.getNumGroupTable(SOLVENT), domdecGroups.getGroupTable(SOLVENT),
			    domdecGroups.getGroupList<solvent_t>(SOLVENT), stream);
      domdec.constCommSetup(domdecGroups.getNeighPos(), domdecGroups.getCoordInd(),
			    nlist.get_glo2loc(), stream);
    }
  }

}

//
// Communicate constraint coordinates in direction "dir" =-1 or +1
//
void CudaPMEForcefield::constComm(const int dir, cudaXYZ<double>& coord, cudaStream_t stream) {
  domdec.constCommDo(dir, coord, stream);
}

//
// Assigns coordinates to nodes
// NOTE: All nodes receive all coordinates here. Domdec distributes them across the nodes
//
void CudaPMEForcefield::assignCoordToNodes(hostXYZ<double>& coord, std::vector<int>& h_loc2glo) {
  // Build loc2glo for the homezone, we now know the number of coordinates at the homezone
  domdec.build_homezone(coord);
  // Copy loc2glo to h_loc2glo
  h_loc2glo.resize(domdec.get_ncoord());
  copy_DtoH<int>(domdec.get_loc2glo_ptr(), h_loc2glo.data(), domdec.get_ncoord());
  // Re-allocate coordinate arrays to the new homezone size
  ref_coord.realloc(domdec.get_ncoord());
  ref_coord.clear();
  xyzq.realloc(domdec.get_ncoord());
  xyzq_copy.realloc(domdec.get_ncoord());
}

//
// Checks if non-bonded list needs to be updated
// Returns true if update is needed
//
bool CudaPMEForcefield::heuristic_check(const cudaXYZ<double>& coord, cudaStream_t stream) {
  assert(domdec.get_ncoord() <= ref_coord.size());
  assert(domdec.get_ncoord() <= coord.size());
  assert(warpsize <= 32);

  double rsq_limit_dbl = fabs(domdec.get_rnl() - roff)/2.0;
  rsq_limit_dbl *= rsq_limit_dbl;
  float rsq_limit = (float)rsq_limit_dbl;

  int nthread = 512;
  int nblock = (domdec.get_ncoord() - 1)/nthread + 1;
  int shmem_size = (nthread/warpsize)*sizeof(int);

  *h_heuristic_flag = 0;
  copy_HtoD<int>(h_heuristic_flag, d_heuristic_flag, 1, stream);

  heuristic_check_kernel<<< nblock, nthread, shmem_size, stream >>>
    (domdec.get_ncoord(), coord.x(), coord.y(), coord.z(),
     ref_coord.x(), ref_coord.y(), ref_coord.z(),
     rsq_limit, d_heuristic_flag);
  cudaCheck(cudaGetLastError());

  copy_DtoH<int>(d_heuristic_flag, h_heuristic_flag, 1, stream);

  // Make sure above kernel calls finish
  cudaCheck(cudaStreamSynchronize(stream));

  return domdec.checkHeuristic((*h_heuristic_flag != 0));
}

//
// Print energies and virials on screen
//
void CudaPMEForcefield::print_energy_virial(int step, const double energy_kin) {
  double tol = 0.0;

  double energy = energy_bond + energy_angle + energy_ureyb + energy_dihe + energy_imdihe +
    energy_vdw + energy_elec + energy_ewksum + energy_ewself + energy_excl;
  double energy_tot = energy + energy_kin;
  double temp = 0.0;

  printf("DYNA>     %d %lf %lf %lf %lf\n",step, energy_tot, energy_kin, energy, temp);

  if (fabs(energy_bond) >= tol || fabs(energy_angle) >= tol || fabs(energy_ureyb) >= tol ||
      fabs(energy_dihe) >= tol || fabs(energy_imdihe) >= tol) {
    printf("DYNA INTERN> %lf %lf %lf %lf %lf\n",
	   energy_bond, energy_angle, energy_ureyb, energy_dihe, energy_imdihe);
  }

  if (fabs(energy_vdw) >= tol || fabs(energy_elec) >= tol) {
    printf("DYNA EXTERN> %lf %lf\n",energy_vdw, energy_elec);
  }

  if (fabs(energy_ewksum) >= tol || fabs(energy_ewself) >= tol || fabs(energy_excl) >= tol) {
    printf("DYNA EWALD> %lf %lf %lf\n",energy_ewksum, energy_ewself, energy_excl);
  }

}

//
// Copies restart data into host buffers
//
void CudaPMEForcefield::get_restart_data(cudaXYZ<double>& coord, cudaXYZ<double>& step,
					 Force<long long int>& force,
					 double *x, double *y, double *z,
					 double *dx, double *dy, double *dz,
					 double *fx, double *fy, double *fz) {

  int* loc2glo = new int[domdec.get_ncoord()];
  int* loc2glo_glo = new int[domdec.get_ncoord_glo()];
  int* nrecv = new int[domdec.get_numnode()];
  int* precv = new int[domdec.get_numnode()];
  double* recvbuf = new double[domdec.get_ncoord_glo()];
  hostXYZ<double> hXYZ(domdec.get_ncoord(), NON_PINNED);

  copy_DtoH_sync<int>(domdec.get_loc2glo_ptr(), loc2glo, domdec.get_ncoord());
  domdec.buildGlobal_loc2glo(loc2glo, loc2glo_glo, nrecv, precv);

  hXYZ.set_data_sync(domdec.get_ncoord(), coord);
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, hXYZ.x(), x);
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, hXYZ.y(), y);
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, hXYZ.z(), z);
  hXYZ.set_data_sync(domdec.get_ncoord(), step);
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, hXYZ.x(), dx);
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, hXYZ.y(), dy);
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, hXYZ.z(), dz);
  hXYZ.set_data_sync(domdec.get_ncoord(), (double *)force.x(), (double *)force.y(), (double *)force.z());
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, hXYZ.x(), fx);
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, hXYZ.y(), fy);
  domdec.combineData(loc2glo_glo, nrecv, precv, recvbuf, hXYZ.z(), fz);

  delete [] loc2glo;
  delete [] loc2glo_glo;
  delete [] nrecv;
  delete [] precv;
  delete [] recvbuf;

}
