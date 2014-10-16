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
				     NeighborList<32>& nlist,
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
  domdec(domdec), recip(recip), domdecGroups(domdecGroups), nlist(nlist), recipComm(recipComm),
  kappa(kappa), recip_force_len(0), recip_force(NULL) {

  // Create streams
  cudaCheck(cudaStreamCreate(&direct_stream[0]));
  cudaCheck(cudaStreamCreate(&direct_stream[1]));
  cudaCheck(cudaStreamCreate(&recip_stream));
  cudaCheck(cudaStreamCreate(&in14_stream));
  cudaCheck(cudaStreamCreate(&bonded_stream));

  // Create events
  cudaCheck(cudaEventCreate(&done_direct_event));
  cudaCheck(cudaEventCreate(&done_recip_event));
  cudaCheck(cudaEventCreate(&done_in14_event));
  cudaCheck(cudaEventCreate(&done_bonded_event));
  cudaCheck(cudaEventCreate(&done_calc_event));
  cudaCheck(cudaEventCreate(&done_force_clear_event));

  // Set energy term flags
  calc_bond = true;
  calc_ureyb = true;
  calc_angle = true;
  calc_dihe = true;
  calc_imdihe = true;
  calc_cmap = true;

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
  cudaCheck(cudaEventDestroy(done_direct_event));
  cudaCheck(cudaEventDestroy(done_recip_event));
  cudaCheck(cudaEventDestroy(done_in14_event));
  cudaCheck(cudaEventDestroy(done_bonded_event));
  cudaCheck(cudaEventDestroy(done_calc_event));
  cudaCheck(cudaEventDestroy(done_force_clear_event));
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


//
// Pre-process force calculation
//
void CudaPMEForcefield::pre_calc(cudaXYZ<double>& coord, cudaXYZ<double>& prev_step) {

  // Check for neighborlist heuristic update
  if (heuristic_check(coord, direct_stream[0])) {
    neighborlist_updated = true;

    fprintf(stderr,"Building neighborlist\n");

    // Update homezone coordinates (coord) and step vector (prev_step)
    // NOTE: Builds domdec.loc2glo
    domdec.update_homezone(coord, prev_step);
    ref_coord.realloc(domdec.get_ncoord());

    // Communicate coordinates
    // NOTE: Builds rest of domdec.loc2glo and domdec.xyz_shift
    domdec.comm_coord(coord, true);

    fprintf(stderr,"%d: domdec.get_ncoord()=%d domdec.get_ncoord_tot()=%d\n",
	    domdec.get_mynode(),domdec.get_ncoord(),domdec.get_ncoord_tot());

    // Re-allocate (xyzq, xyzq_copy)
    xyzq.realloc(domdec.get_ncoord_tot());
    xyzq_copy.realloc(domdec.get_ncoord_tot());

    // Copy: coord => xyzq_copy
    // NOTE: coord and xyz_shift are already in the order determined by domdec.loc2glo,
    //       however, glo_q is in the original global order.
    xyzq_copy.set_xyzq(coord, glo_q, domdec.get_loc2glo_ptr(), domdec.get_xyz_shift(),
		       domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz());

    //nlist.set_test(true);

    // Sort coordinates
    // NOTE: Builds domdec.loc2glo and nlist->glo2loc
    nlist.sort(domdec.get_zone_pcoord(), xyzq_copy.xyzq, xyzq.xyzq, domdec.get_loc2glo_ptr());

    // Build neighborlist
    nlist.build(domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(), domdec.get_rnl(),
		xyzq.xyzq, domdec.get_loc2glo_ptr());

    // Build bonded tables
    domdecGroups.buildGroupTables();
    domdecGroups.syncGroupTables();

    // Check the total number of groups
    domdec.checkNumGroups(domdecGroups.get_atomGroupVector());
    
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
		      domdecGroups.getGroupList<cmap_t>(CMAP));

    // Set vdwtype for Direct non-bonded interactions
    dir.set_vdwtype(domdec.get_ncoord_tot(), glo_vdwtype, domdec.get_loc2glo_ptr());

    // Setup 1-4 interaction lists
    dir.set_14_list(xyzq.xyzq, domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(),
		    nlist.get_glo2loc(),
		    domdecGroups.getNumGroupTable(IN14), domdecGroups.getGroupTable(IN14),
		    domdecGroups.getGroupList<xx14_t>(IN14),
		    domdecGroups.getNumGroupTable(EX14), domdecGroups.getGroupTable(EX14),
		    domdecGroups.getGroupList<xx14_t>(EX14));

    // Re-order prev_step vector, using ref_coord as temporary storage
    domdec.reorder_homezone_coord(prev_step, ref_coord, nlist.get_ind_sorted());

    // Re-order coordinates (coord), using ref_coord as temporary storage
    domdec.reorder_homezone_coord(coord, ref_coord, nlist.get_ind_sorted());

    // Update and re-order communication buffers
    domdec.comm_update(nlist.get_glo2loc(), coord);

  } else {
    neighborlist_updated = false;
    // Copy local coordinates to xyzq -array
    xyzq.set_xyz(coord, 0, domdec.get_ncoord()-1, domdec.get_xyz_shift(),
		 domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(), direct_stream[0]);
    // Communicate coordinates between direct nodes
    domdec.comm_coord(coord, false);
    // Copy import volume coordinates to xyzq -array
    xyzq.set_xyz(coord, domdec.get_ncoord(), domdec.get_ncoord_tot()-1, domdec.get_xyz_shift(),
		 domdec.get_boxx(), domdec.get_boxy(), domdec.get_boxz(), direct_stream[0]);

  }

}

//
// Calculate forces
//
void CudaPMEForcefield::calc(const bool calc_energy, const bool calc_virial, Force<long long int>& force) {

  // Re-allocate force
  force.realloc(domdec.get_ncoord_tot());

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
    recipComm.send_coord(xyzq.xyzq);
    // Receive coordinates
    if (recipComm.get_isRecip()) recipComm.recv_coord(recip_xyzq.xyzq);
    //-------------------------------------
  }

  force.clear(direct_stream[0]);
  cudaCheck(cudaEventRecord(done_force_clear_event, direct_stream[0]));

  // Clear energy and virial variables
  if (calc_energy || calc_virial) {
    dir.clear_energy_virial();
    bonded.clear_energy_virial();
    if (recipComm.get_isRecip()) recip->clear_energy_virial();
  }

  // Direct non-bonded force
  dir.calc_force(xyzq.xyzq, nlist, calc_energy, calc_virial, force.stride(), force.xyz(),
  		 direct_stream[0]);
  cudaCheck(cudaEventRecord(done_direct_event, direct_stream[0]));

  // 1-4 interactions
  // NOTE: we make GPU wait until force.cleap() is done
  cudaCheck(cudaStreamWaitEvent(in14_stream, done_force_clear_event, 0));
  dir.calc_14_force(xyzq.xyzq, calc_energy, calc_virial, force.stride(), force.xyz(),
  		    in14_stream);
  cudaCheck(cudaEventRecord(done_in14_event, in14_stream));

  // Bonded forces
  cudaCheck(cudaStreamWaitEvent(bonded_stream, done_force_clear_event, 0));
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
	cudaCheck(cudaStreamWaitEvent(recip_stream, done_force_clear_event, 0));
	recip->calc(domdec.get_inv_boxx(), domdec.get_inv_boxy(), domdec.get_inv_boxz(),
		    xyzq.xyzq, xyzq.ncoord,
		    calc_energy, calc_virial, force);
      } else {
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

  // Make GPU wait until all computation is done
  cudaCheck(cudaStreamWaitEvent(direct_stream[0], done_in14_event, 0));
  cudaCheck(cudaStreamWaitEvent(direct_stream[0], done_bonded_event, 0));
  cudaCheck(cudaStreamWaitEvent(direct_stream[0], done_recip_event, 0));
  cudaCheck(cudaStreamWaitEvent(direct_stream[0], done_direct_event, 0));

  // Convert forces from FP to DP
  force.convert<double>(direct_stream[0]);

  bonded.get_energy_virial(calc_energy, calc_virial,
			   &energy_bond, &energy_ureyb,
			   &energy_angle,
			   &energy_dihe, &energy_imdihe,
			   &energy_cmap,
			   sforcex, sforcey, sforcez);

  dir.get_energy_virial(calc_energy, calc_virial,
			&energy_vdw, &energy_elec,
			&energy_excl, vir);

  if (recipComm.get_isRecip()) {
    recip->get_energy_virial(calc_energy, calc_virial, energy_ewksum, energy_ewself, vir);
  }

  // Communicate Direct-Direct
  domdec.comm_force(force);

  if (do_recipcomm) {
    // Communicate Direct-Recip forces
    if (recipComm.get_isRecip()) recipComm.send_force(recip_force);
    recipComm.recv_force(recip_force);
    // Add Recip force to the total force
    force.add<double>(recipComm.get_force_ptr(), domdec.get_ncoord(), direct_stream[0]);
  }

}

//
// Post-process force calculation. Used for array re-ordering after neighborlist search
// Updates holonomic constraint tables if neccessary
//
void CudaPMEForcefield::post_calc(const float *global_mass, float *mass, HoloConst* holoconst) {

  if (neighborlist_updated) {
    // Re-order xyz_shift
    domdec.reorder_xyz_shift(nlist.get_ind_sorted());

    // Re-do mass array
    map_to_local_array<float>(domdec.get_ncoord(), domdec.get_loc2glo_ptr(), global_mass, mass);

    if (holoconst != NULL) {
      holoconst->setup_list(nlist.get_glo2loc(),
			    domdecGroups.getNumGroupTable(PAIR), domdecGroups.getGroupTable(PAIR),
			    domdecGroups.getGroupList<bond_t>(PAIR),
			    domdecGroups.getNumGroupTable(TRIP), domdecGroups.getGroupTable(TRIP),
			    domdecGroups.getGroupList<angle_t>(TRIP),
			    domdecGroups.getNumGroupTable(QUAD), domdecGroups.getGroupTable(QUAD),
			    domdecGroups.getGroupList<dihe_t>(QUAD),
			    domdecGroups.getNumGroupTable(SOLVENT), domdecGroups.getGroupTable(SOLVENT),
			    domdecGroups.getGroupList<solvent_t>(SOLVENT));
    }
  }

  cudaCheck(cudaEventRecord(done_calc_event, direct_stream[0]));
}

//
// Make stream "stream" wait until calc - routine is done
//
void CudaPMEForcefield::wait_calc(cudaStream_t stream) {
  cudaCheck(cudaStreamWaitEvent(stream, done_calc_event, 0));
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
  assert(ref_coord.size() == domdec.get_ncoord());
  assert(coord.size() == domdec.get_ncoord_tot());
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

  copy_DtoH_sync<int>(d_heuristic_flag, h_heuristic_flag, 1);

  return domdec.checkHeuristic((*h_heuristic_flag != 0));
}

//
// Print energies and virials on screen
//
void CudaPMEForcefield::print_energy_virial(int step) {
  double tol = 0.0;

  double energy_kin = 0.0;
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
