#include <stdio.h>
#include <cassert>
#include "CudaPMEForcefield.h"
#include "cuda_utils.h"
#include "gpu_utils.h"

__global__ void heuristic_check_kernel(const int ncoord, const int stride,
				       const double* __restrict__ coord,
				       const double* __restrict__ ref_coord,
				       const float rsq_limit,
				       int* global_flag) {
  // Required shared memory:
  // blockDim.x/warpsize*sizeof(int)
  extern __shared__ int sh_flag[];
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int stride2 = stride*2;
  const int sh_flag_size = blockDim.x/warpsize;

  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
  if (tid < ncoord) {
    dx = (float)(coord[tid]         - ref_coord[tid]);
    dy = (float)(coord[tid+stride]  - ref_coord[tid+stride]);
    dz = (float)(coord[tid+stride2] - ref_coord[tid+stride2]);
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
CudaPMEForcefield::CudaPMEForcefield(CudaDomdec *domdec, CudaDomdecBonded *domdec_bonded,
				     NeighborList<32> *nlist,
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
				     const int *h_glo_vdwtype, const float *h_q,
				     const int nfftx, const int nffty, const int nfftz,
				     const int order) {

  // Domain decomposition
  this->domdec = domdec;
  this->domdec_bonded = domdec_bonded;

  // Neighborlist
  this->nlist = nlist;

  // Bonded coefficients
  bonded.setup_coef(nbondcoef, h_bondcoef, nureybcoef, h_ureybcoef,
		    nanglecoef, h_anglecoef, ndihecoef, h_dihecoef,
		    nimdihecoef, h_imdihecoef, ncmapcoef, h_cmapcoef);
  
  // Direct non-bonded interactions
  setup_direct_nonbonded(roff, ron, kappa, e14fac, vdw_model, elec_model,
			 nvdwparam, h_vdwparam, h_vdwparam14, h_glo_vdwtype);

  // Copy charges
  allocate<float>(&q, domdec->get_ncoord_glo());
  copy_HtoD<float>(h_q, q, domdec->get_ncoord_glo());

  // Recip non-bonded interactions
  setup_recip_nonbonded(kappa, nfftx, nffty, nfftz, order);

  allocate<int>(&d_heuristic_flag, 1);
  allocate_host<int>(&h_heuristic_flag, 1);
}

//
// Class destructor
//
CudaPMEForcefield::~CudaPMEForcefield() {
  deallocate<int>(&d_heuristic_flag);
  deallocate_host<int>(&h_heuristic_flag);
  deallocate<float>(&q);
  if (grid != NULL) delete grid;
}

//
// Setup direct non-bonded interactions.
//
void CudaPMEForcefield::setup_direct_nonbonded(const double roff, const double ron,
					       const double kappa, const double e14fac,
					       const int vdw_model, const int elec_model,
					       const int nvdwparam, const float *h_vdwparam,
					       const float *h_vdwparam14, const int *h_glo_vdwtype) {

  dir.setup(domdec->get_boxx(), domdec->get_boxy(), domdec->get_boxz(), kappa, roff, ron,
	    e14fac, vdw_model, elec_model);

  dir.set_vdwparam(nvdwparam, h_vdwparam);
  dir.set_vdwparam14(nvdwparam, h_vdwparam14);

  allocate<int>(&glo_vdwtype, domdec->get_ncoord_glo());
  copy_HtoD<int>(h_glo_vdwtype, glo_vdwtype, domdec->get_ncoord_glo());
}

//
// Setup recip non-bonded interactions.
//
void CudaPMEForcefield::setup_recip_nonbonded(const double kappa,
					      const int nfftx, const int nffty, const int nfftz,
					      const int order) {

  this->kappa = kappa;

  if (nfftx > 0 && nffty > 0 && nfftz > 0 && order > 0) {
    const FFTtype fft_type = BOX;
    grid = new Grid<int, float, float2>(nfftx, nffty, nfftz, order, fft_type, 1, 0);
  } else {
    grid = NULL;
  }

}

//
// Calculate forces
//
void CudaPMEForcefield::calc(cudaXYZ<double> *coord,
			     const bool calc_energy, const bool calc_virial,
			     Force<long long int> *force) {


  std::cout << "CudaPMEForcefield::calc" << std::endl;

  // Check for neighborlist heuristic update
  if (heuristic_check(coord)) {
    std::cout << "  Building neighborlist" << std::endl;

    // Update homezone coordinates
    // NOTE: Builds domdec->loc2glo
    domdec->update_homezone(coord);

    // Communicate coordinates
    // NOTE: Builds rest of domdec->loc2glo and domdec->xyz_shift
    domdec->comm_coord(coord, true);

    // Copy coord => xyzq_copy
    xyzq_copy.set_xyzq(coord, q, domdec->get_xyz_shift(),
		       domdec->get_boxx(), domdec->get_boxy(), domdec->get_boxz());

    // Update neighborlist
    nlist->sort(domdec->get_zone_pcoord(), xyzq_copy.xyzq, xyzq.xyzq, domdec->get_loc2glo());
    nlist->build(domdec->get_boxx(), domdec->get_boxy(), domdec->get_boxz(), domdec->get_rnl(),
		 xyzq.xyzq, domdec->get_loc2glo());

    /*
    nlist->test_build(domdec->get_zone_pcoord(), domdec->get_boxx(), domdec->get_boxy(),
		      domdec->get_boxz(), domdec->get_rnl(), xyzq.xyzq, domdec->get_loc2glo());
    */

    // Build bonded tables
    domdec_bonded->build_tbl(domdec, domdec->get_zone_pcoord());

    // Setup bonded interaction lists
    bonded.setup_list(xyzq.xyzq, domdec->get_boxx(), domdec->get_boxy(), domdec->get_boxz(),
		      nlist->get_glo2loc(),
		      domdec_bonded->get_nbond_tbl(), domdec_bonded->get_bond_tbl(),
		      domdec_bonded->get_bond(),
		      domdec_bonded->get_nureyb_tbl(), domdec_bonded->get_ureyb_tbl(),
		      domdec_bonded->get_ureyb(),
		      domdec_bonded->get_nangle_tbl(), domdec_bonded->get_angle_tbl(),
		      domdec_bonded->get_angle(),
		      domdec_bonded->get_ndihe_tbl(), domdec_bonded->get_dihe_tbl(),
		      domdec_bonded->get_dihe(),
		      domdec_bonded->get_nimdihe_tbl(), domdec_bonded->get_imdihe_tbl(),
		      domdec_bonded->get_imdihe(),
		      domdec_bonded->get_ncmap_tbl(), domdec_bonded->get_cmap_tbl(),
		      domdec_bonded->get_cmap());

    // Set vdwtype for Direct non-bonded interactions
    dir.set_vdwtype(domdec->get_ncoord_tot(), glo_vdwtype, domdec->get_loc2glo());

    // Setup 1-4 interaction lists
    dir.set_14_list(xyzq.xyzq, domdec->get_boxx(), domdec->get_boxy(), domdec->get_boxz(),
		    nlist->get_glo2loc(),
		    domdec_bonded->get_nin14_tbl(), domdec_bonded->get_in14_tbl(),
		    domdec_bonded->get_in14(),
		    domdec_bonded->get_nex14_tbl(), domdec_bonded->get_ex14_tbl(),
		    domdec_bonded->get_ex14());

    // Update reference coordinates
    ref_coord.set_data(coord);

  } else {
    std::cout << "  Not building neighborlist" << std::endl;
    // Communicate coordinates
    domdec->comm_coord(coord, false);
    // Copy coordinates to xyzq -array
    xyzq.set_xyz(coord);
  }

  // Clear energy and virial variables
  if (calc_energy || calc_virial) {
    dir.clear_energy_virial();
    bonded.clear_energy_virial();
  }

  // Direct non-bonded force
  dir.calc_force(xyzq.xyzq, nlist, calc_energy, calc_virial, force->xyz.stride, force->xyz.data);

  // 1-4 interactions
  dir.calc_14_force(xyzq.xyzq, calc_energy, calc_virial, force->xyz.stride, force->xyz.data);

  // Bonded forces
  bonded.calc_force(xyzq.xyzq, domdec->get_boxx(), domdec->get_boxy(), domdec->get_boxz(),
		    calc_energy, calc_virial, force->xyz.stride, force->xyz.data);

  /*
  // Reciprocal forces (Only reciprocal nodes calculate these)
  if (grid != NULL) {
    double recip[9];
    for (int i=0;i < 9;i++) recip[i] = 0;
    recip[0] = 1.0/domdec->get_boxx();
    recip[4] = 1.0/domdec->get_boxy();
    recip[8] = 1.0/domdec->get_boxz();
    grid->spread_charge(xyzq.xyzq, xyzq.ncoord, recip);
    grid->r2c_fft();
    grid->scalar_sum(recip, kappa, calc_energy, calc_virial);
    grid->c2r_fft();
    grid->gather_force(xyzq.xyzq, xyzq.ncoord, recip, recip_force.xyz.stride, recip_force.xyz.data);
  }
  */

  bonded.get_energy_virial(calc_energy, calc_virial,
			   &energy_bond, &energy_ureyb,
			   &energy_angle,
			   &energy_dihe, &energy_imdihe,
			   &energy_cmap,
			   sforcex, sforcey, sforcez);

  dir.get_energy_virial(calc_energy, calc_virial,
			&energy_vdw, &energy_elec,
			&energy_excl, vir);

  // Communicate forces (After this all nodes have their correct total force)
  domdec->comm_force(force);

}

//
// Initializes coordinates.
// NOTE: All nodes receive all coordinates here. Domdec distributes them across the nodes
//
void CudaPMEForcefield::init_coord(cudaXYZ<double> *coord) {
  domdec->build_homezone(coord);
  ref_coord.resize(coord->n);
  ref_coord.clear();
  xyzq.set_ncoord(coord->n);
  xyzq_copy.set_ncoord(coord->n);
}

//
// Checks if non-bonded list needs to be updated
// Returns true if update is needed
//
bool CudaPMEForcefield::heuristic_check(const cudaXYZ<double> *coord) {
  assert(ref_coord.match(coord));
  assert(warpsize <= 32);

  double rsq_limit_dbl = fabs(domdec->get_rnl() - roff)/2.0;
  rsq_limit_dbl *= rsq_limit_dbl;
  float rsq_limit = (float)rsq_limit_dbl;

  int ncoord = ref_coord.n;
  int stride = ref_coord.stride;
  int nthread = 512;
  int nblock = (ncoord - 1)/nthread + 1;

  int shmem_size = (nthread/warpsize)*sizeof(int);

  *h_heuristic_flag = 0;
  copy_HtoD<int>(h_heuristic_flag, d_heuristic_flag, 1, 0);

  heuristic_check_kernel<<< nblock, nthread, shmem_size, 0 >>>
    (ncoord, stride, coord->data, ref_coord.data, rsq_limit, d_heuristic_flag);

  cudaCheck(cudaGetLastError());

  copy_DtoH_sync<int>(d_heuristic_flag, h_heuristic_flag, 1);
  
  return (*h_heuristic_flag != 0);
}

//
// Print energies and virials on screen
//
void CudaPMEForcefield::print_energy_virial() {
  double tol = 0.0;

  if (fabs(energy_bond) >= tol || fabs(energy_angle) >= tol || fabs(energy_ureyb) >= tol ||
      fabs(energy_dihe) >= tol || fabs(energy_imdihe) >= tol) {
    printf("DYNA INTERN> %lf %lf %lf %lf %lf\n",
	   energy_bond, energy_angle, energy_ureyb, energy_dihe, energy_imdihe);
  }

  if (fabs(energy_vdw) >= tol || fabs(energy_elec) >= tol || fabs(energy_excl) >= tol) {
    printf("DYNA EXTERN> %lf %lf %lf\n",energy_vdw, energy_elec, energy_excl);
  }

}
