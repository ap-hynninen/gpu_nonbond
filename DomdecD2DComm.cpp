#include <iostream>
#include <cmath>             // ceil
#include "DomdecD2DComm.h"

//
// Class creator
//
DomdecD2DComm::DomdecD2DComm(Domdec& domdec) : domdec(domdec) {

  setup_subboxes();

  setup_comm_nodes();

  // Send
  z_nsend.resize(nz_comm);
  z_psend.resize(nz_comm+1);

  // Recv
  z_nrecv.resize(nz_comm);
  z_precv.resize(nz_comm+1);

}

//
// Setup communication node IDs
//
void DomdecD2DComm::setup_comm_nodes() {

  int homeix = domdec.get_homeix();
  int homeiy = domdec.get_homeiy();
  int homeiz = domdec.get_homeiz();
  int nx = domdec.get_nx();
  int ny = domdec.get_ny();
  int nz = domdec.get_nz();

  x_send_node.resize(nx_comm);
  x_recv_node.resize(nx_comm);

  y_send_node.resize(ny_comm);
  y_recv_node.resize(ny_comm);

  z_send_node.resize(nz_comm);
  z_recv_node.resize(nz_comm);

  for (int i=0;i < nx_comm;i++) {
    x_send_node[i] = domdec.get_nodeind_pbc(homeix-(i+1), homeiy, homeiz);
    x_recv_node[i] = domdec.get_nodeind_pbc(homeix+(i+1), homeiy, homeiz);
  }

  for (int i=0;i < ny_comm;i++) {
    y_send_node[i] = domdec.get_nodeind_pbc(homeix, homeiy-(i+1), homeiz);
    y_recv_node[i] = domdec.get_nodeind_pbc(homeix, homeiy+(i+1), homeiz);
  }

  for (int i=0;i < nz_comm;i++) {
    z_send_node[i] = domdec.get_nodeind_pbc(homeix, homeiy, homeiz-(i+1));
    z_recv_node[i] = domdec.get_nodeind_pbc(homeix, homeiy, homeiz+(i+1));
  }

}

//
// Setup sub-box configuration
//
void DomdecD2DComm::setup_subboxes() {
  // Initial setup
  int nx = domdec.get_nx();
  int ny = domdec.get_ny();
  int nz = domdec.get_nz();
  double rnl = domdec.get_rnl();
  double inv_boxx = domdec.get_inv_boxx();
  double inv_boxy = domdec.get_inv_boxy();
  double inv_boxz = domdec.get_inv_boxz();
  double fx_init = 1.0/(double)nx;
  double fy_init = 1.0/(double)ny;
  double fz_init = 1.0/(double)nz;
  nx_comm = std::min(nx, (int)ceil(rnl*inv_boxx/fx_init));
  ny_comm = std::min(ny, (int)ceil(rnl*inv_boxy/fy_init));
  nz_comm = std::min(nz, (int)ceil(rnl*inv_boxz/fz_init));

  if (nx == 1) nx_comm = 0;
  if (ny == 1) ny_comm = 0;
  if (nz == 1) nz_comm = 0;
  
  load_balance = false;

  if (load_balance) {
    std::cerr << "DomdecD2DComm::setup_subboxes(), load balancing not yet supported" << std::endl;
    exit(1);
    load_balance_x = false;
    load_balance_y = false;
    load_balance_z = false;
    load_balance = load_balance_x || load_balance_y || load_balance_z;
  } else {
    load_balance_x = false;
    load_balance_y = false;
    load_balance_z = false;
    max_fx = 1.0 - 2.0*rnl*inv_boxx;
    max_fy = 1.0 - 2.0*rnl*inv_boxy;
    max_fz = 1.0 - 2.0*rnl*inv_boxz;
  }

  // Allocate and initialize (fx, fy, fz) and (bx, by, bz)
  fx.resize(nx);
  bx.resize(nx+1);
  for (int ix=0;ix < nx;ix++) {
    fx[ix] = fx_init;
  }

  fy.resize(2*nx_comm+1);
  by.resize(2*nx_comm+1);
  for (int ix=0;ix < 2*nx_comm+1;ix++) {
    fy[ix].resize(ny);
    by[ix].resize(ny+1);
    for (int iy=0;iy < ny;iy++) {
      fy[ix][iy] = fy_init;
    }
  }

  fz.resize(2*nx_comm+1);
  bz.resize(2*nx_comm+1);
  for (int ix=0;ix < 2*nx_comm+1;ix++) {
    fz[ix].resize(2*ny_comm+1);
    bz[ix].resize(2*ny_comm+1);
    for (int iy=0;iy < 2*ny_comm+1;iy++) {
      fz[ix][iy].resize(nz);
      bz[ix][iy].resize(nz+1);
      for (int iz=0;iz < nz;iz++) {
	fz[ix][iy][iz] = fz_init;
      }
    }
  }

  fill_bx_by_bz();

  double fx_maxval = fx_init;
  double fy_maxval = fy_init;
  double fz_maxval = fz_init;

  if (nx > 1 && fx_maxval > max_fx) {
    std::cout << "box size too small in x direction, try setting larger ndir value" << std::endl;
    exit(1);
  }

  if (ny > 1 && fy_maxval > max_fy) {
    std::cout << "box size too small in y direction, try setting larger ndir value" << std::endl;
    exit(1);
  }

  if (nz > 1 && fz_maxval > max_fz) {
    std::cout << "box size too small in z direction, try setting larger ndir value" << std::endl;
    exit(1);
  }
 
}

//
// Fills nodebx, nodeby, nodebz:
// bx(i) = sum(nodefx(1:i))
//
void DomdecD2DComm::fill_bx_by_bz() {

  int nx = domdec.get_nx();
  int ny = domdec.get_ny();
  int nz = domdec.get_nz();

  bx[0] = 0.0;
  for (int i=0;i < nx-1;i++) bx[i+1] = bx[i] + fx[i];
  bx[nx] = 1.0;

  for (int ix=0;ix < 2*nx_comm+1;ix++) {
    by[ix][0] = 0.0;
    for (int i=0;i < ny-1;i++) by[ix][i+1] = by[ix][i] + fy[ix][i];
    by[ix][ny] = 1.0;
  }

  for (int ix=0;ix < 2*nx_comm+1;ix++) {
    for (int iy=0;iy < 2*ny_comm+1;iy++) {
      bz[ix][iy][0] = 0.0;
      for (int i=0;i < nz-1;i++) bz[ix][iy][i+1] = bz[ix][iy][i] + fz[ix][iy][i];
      bz[ix][iy][nz] = 1.0;
    }
  }

}

//
// Calculates node upper z-boundary from nodefrz, where (iz, iy, ix) are in absolute coordinates:
// (iz, iy, ix) in (0:nz-1, -ny_comm+homeiy:ny_comm+homeiy, -nx_comm+homeix:nx_comm+homeix)
//
double DomdecD2DComm::get_bz(const int iz, const int iy, const int ix) {
  int nz = domdec.get_nz();

  int izt = iz + 1;       // +1 is for upper boundary

  while (izt < 0) izt = izt + nz;
  while (izt > nz) izt = izt - nz;

  int iyt = iy - domdec.get_homeiy() + ny_comm;
  int ixt = ix - domdec.get_homeix() + nx_comm;

  if (iyt < 0 || iyt > 2*ny_comm) {
    std::cout << "DomdecD2DComm::get_bz, iy out of range" << std::endl;
  }
  if (ixt < 0 || ixt > 2*nx_comm) {
    std::cout << "DomdecD2DComm::get_bz, ix out of range" << std::endl;
  }

  return bz[ixt][iyt][izt];
}

//
// Calculates node upper y-boundary from nodefry, where (iy, ix) are in absolute coordinates:
// (iy, ix) in (0:ny-1, -nx_comm+homeix:nx_comm+homeix)
//
double DomdecD2DComm::get_by(const int iy, const int ix) {
  int ny = domdec.get_ny();

  int iyt = iy + 1;       // +1 is for upper boundary

  while (iyt < 0) iyt = iyt + ny;
  while (iyt > ny) iyt = iyt - ny;

  int ixt = ix - domdec.get_homeix() + nx_comm;

  if (ixt < 0 || ixt > 2*nx_comm) {
    std::cout << "DomdecD2DComm::get_by, ix out of range" << std::endl;
  }

  return by[ixt][iyt];
}

//
// Calculates node upper x-boundary from nodefrx, where (ix) is in absolute coordinates:
// (ix) in (0:nx-1)
//
double DomdecD2DComm::get_bx(const int ix) {
  int nx = domdec.get_nx();

  int ixt = ix + 1;       // +1 is for upper boundary

  while (ixt < 0) ixt = ixt + nx;
  while (ixt > nx) ixt = ixt - nx;

  return bx[ixt];
}

//
// Returns the z communication boundary for the node that is at (ix, iy, iz)
//
void DomdecD2DComm::get_fz_boundary(const int ix, const int iy, const int iz,
				    const double cut, const double cut_grouped, double& fz_z) {
  /*
    integer ixl, iyl, j, k
    real(chm_real) fz_z_top
    real(chm_real) ez_z_top, fx_z_top, fy_z_top
    real(chm_real) ex_y, ex_z, ey_x, ey_z, group_z, group_z_top
    real(chm_real) xd, yd, zd, dist, y0, ez_y, shy
    real(chm_real) cutsq, cutz, cutz_grouped, cutsq_grouped
    logical q_checkgrouped
  */

  fz_z = get_bz(iz, iy, ix);

  /*
    if (q_load_balance_z) then
       cutsq = cut*cut
       cutz = cut/boxz
       cutsq_grouped = cut_grouped*cut_grouped
       cutz_grouped = cut_grouped/boxz
       ! These loops expand the FZ zone to include FZ-FX, FZ-FY, 
       ! and FZ-EZ interactions correctly.
       ! NOTE: Although they could be written as a single loop, 
       ! they are written out separately for clarity sake
       !
       ! This loop expands the FZ zone to include all FZ-EZ interactions correctly.
       ! We loop through: ix+1:ix+nx_comm, iy+1:iy+ny_comm
       !
       ! fz_z     = z coordinate of the FZ zone bottom
       ! fz_z_top = z coordinate of the FZ zone top
       ! ez_z_top = z coordinate of the EZ zone top
       !
       ! y0   = y coordinate of the node (iy,ix) top
       ! ez_y = bottom y coordinate of the cell (iyl,ixl) in EZ
       ! xd = x distance from the node top (x,y) -corner
       ! yd = positive y distance (if yd < 0, y-distance must be considered to be 0)
       xd = zero
       y0 = get_nodeby(iy,ix)
       ! Loop through all cells in the EZ zone
       do ixl=ix+1,ix+nx_comm
          ez_y = get_nodeby(iy, ixl)
          do iyl=iy+1,iy+ny_comm
             yd = ez_y - y0
             yd = max(zero, yd)              ! Enforces yd >= 0
             fz_z_top = fz_z + cutz
             ez_z_top = get_nodebz(iz, iyl, ixl)
             zd = fz_z_top - ez_z_top
             zd = max(zero, zd)              ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                fz_z = sqrt(cutsq - (yd*boxy)**2 - (xd*boxx)**2)/boxz - cutz + ez_z_top
             endif
             ez_y = ez_y + nodefry_pbc(iyl, ixl)
          enddo
          xd = xd + nodefrx_pbc(ixl)
       enddo

       ! This loop expands the FZ zone to include all FZ-FX interactions correctly.
       xd = zero
       ! Loop through all cells in the FX zone
       do ixl=ix+1,ix+nx_comm
          fz_z_top = fz_z + cutz
          fx_z_top = get_nodebz(iz, iy, ixl)
          if (ixl == ix+1) then
             ! Treat special case of neighboring cell separately
             if (fz_z_top - fx_z_top < cutz) fz_z = fx_z_top
          else
             zd = fz_z_top - fx_z_top
             zd = max(zero, zd)              ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                fz_z = sqrt(cutsq - (xd*boxx)**2)/boxz - cutz + fx_z_top
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

       ! This loop expands the FZ zone to include all FZ-FY interactions correctly.
       yd = zero
       ! Loop through all cells in the FY zone
       do iyl=iy+1,iy+ny_comm
          fz_z_top = fz_z + cutz
          fy_z_top = get_nodebz(iz, iyl, ix)
          if (iyl == iy+1) then
             ! Treat special case of neighboring cell separately
             if (fz_z_top - fy_z_top < cutz) fz_z = fy_z_top
          else
             zd = fz_z_top - fy_z_top
             zd = max(zero, zd)              ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (yd*boxy)**2
             if (dist < cutsq) then
                fz_z = sqrt(cutsq - (yd*boxy)**2)/boxz - cutz + fy_z_top
             endif
          endif
          yd = yd + nodefry_pbc(iyl, ix)
       enddo

       ! Coordinates from the FZ zone will be further communicated in the y and x
       ! directions to zones EX, EY, and C. Therefore, we have to expand FZ zone
       ! to include all coordinates that are needed in this future communications

       ! This loop expands the FZ zone to include coordinates in the EX zones of the
       ! nodes in iy-1:iy-ny_comm
       ! Also takes care of the expansion due to future communication to C zones
       ! (calls are made from get_ex_boundary to get_c_boundary)
       !
       do iyl=iy-1,iy-ny_comm,-1
          call get_ex_boundary(ix, iyl, iz, ex_y, ex_z, &
               group_z, q_checkgrouped, cut, cut_grouped)
          if (iyl == iy-1) then
             ! Treat neighboring cell case separately
             fz_z = max(fz_z, ex_z)
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fz_z_top = fz_z + cutz
                group_z_top = group_z + cutz_grouped
                if (group_z_top > fz_z_top) fz_z = group_z_top - cutz
             endif
          else
             fz_z_top = fz_z + cutz
             zd = fz_z_top - ex_z
             zd = max(zero, zd)           ! Ensures zd >= 0
             ! Shift factor shy = 1 if we have crossed the box border in y-direction
             shy = zero
             if (iy-1 > 0 .and. iyl <= 0) shy = one
             !
             yd = get_nodeby(iy-1, ix) + shy - ex_y
             yd = max(zero, yd)           ! Ensures yd >= 0
             dist = (zd*boxz)**2 + (yd*boxy)**2
             if (dist < cutsq) then
                fz_z = sqrt(cutsq - (yd*boxy)**2)/boxz - cutz + ex_z
             endif
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fz_z_top = fz_z + cutz
                zd = fz_z_top - group_z
                zd = max(zero, zd)           ! Ensures zd >= 0
                dist = (zd*boxz)**2 + (yd*boxy)**2
                if (dist < cutsq_grouped) then
                   fz_z = sqrt(cutsq_grouped - (yd*boxy)**2)/boxz - cutz + group_z
                endif
             endif
          endif
       enddo

       ! This loop expands the FZ zone to include coordinates in the EY zones of the
       ! nodes in ix-1:ix-nx_comm
       xd = zero
       do ixl=ix-1,ix-nx_comm,-1
          call get_ey_boundary(ixl, iy, iz, ey_x, ey_z, group_z, q_checkgrouped, cut)
          if (ixl == ix-1) then
             ! Treat neighboring cell case separately
             fz_z = max(fz_z, ey_z)
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fz_z_top = fz_z + cutz
                group_z_top = group_z + cutz_grouped
                if (group_z_top > fz_z_top) fz_z = group_z_top - cutz
             endif
          else
             fz_z_top = fz_z + cutz          
             zd = fz_z_top - ey_z
             zd = max(zero, zd)           ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                fz_z = sqrt(cutsq - (xd*boxx)**2)/boxz - cutz + ey_z
             endif
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fz_z_top = fz_z + cutz
                zd = fz_z_top - group_z
                zd = max(zero, zd)           ! Ensures zd >= 0
                dist = (zd*boxz)**2 + (xd*boxx)**2
                if (dist < cutsq_grouped) then
                   fz_z = sqrt(cutsq_grouped - (xd*boxx)**2)/boxz - cutz + group_z
                endif
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

    endif
  */

}


//
// Returns the y communication boundary for the node that is at iy
//
void DomdecD2DComm::get_fy_boundary(const int ix, const int iy, const int iz,
				    const double cut, const double cut_grouped, double& fy_y) {
  /*
    integer ixl, izl
    real(chm_real) fy_y_top
    real(chm_real) ey_y_top, fx_y_top, ez_x, ez_y, group_y, group_y_top
    real(chm_real) xd, yd, zd, dist, z0, z1
    real(chm_real) cuty, cutsq, cuty_grouped, cutsq_grouped
    logical q_checkgrouped
  */

  fy_y = get_by(iy, ix);

  /*
    if (q_load_balance_y) then
       cutsq = cut*cut
       cuty = cut/boxy
       cutsq_grouped = cut_grouped*cut_grouped
       cuty_grouped = cut_grouped/boxy
       ! These loops expand the FY zone to include FY-FX and FY-EY interactions correctly
       !
       ! This loop expands the FY zone to include FY-EY interactions correctly
       !
       ! fy_y     = y coordinate of the FY zone bottom
       ! fy_y_top = y coordinate of the FY zone top
       ! ey_y_top = y coordinate of the EY zone top
       !
       ! z0 = z coordinate of the node (iz,iy,ix) top
       ! z1 = z coordinate of the node (izl,iy,ixl) bottom
       !
       ! xd, zd   = x and z distance from node top (x,z) -corner
       xd = zero
       z0 = get_nodebz(iz,iy,ix)
       ! Loop through all cells in the EY zone
       do ixl=ix+1,ix+nx_comm
          z1 = get_nodebz(iz,iy,ixl)
          do izl=iz+1,iz+nz_comm
             zd = z1 - z0
             zd = max(zero, zd)               ! Ensures zd >= 0
             ey_y_top = get_nodeby(iy,ixl)
             fy_y_top = fy_y + cuty
             yd = fy_y_top - ey_y_top
             yd = max(zero, yd)               ! Ensures yd >= 0
             dist = (yd*boxy)**2 + (xd*boxx)**2 + (zd*boxz)**2
             if (dist < cutsq) then
                fy_y = sqrt(cutsq - (xd*boxx)**2 - (zd*boxz)**2)/boxy - cuty + ey_y_top
             endif
             z1 = z1 + nodefrz_pbc(izl,iy,ixl)
          enddo
          xd = xd + nodefrx_pbc(ixl)
       enddo

       ! This loop expands the FY zone to include FY-FX interactions correctly
       !
       ! fx_y_top = y coordinate of the FX zone top
       ! xd = x distance between FY and FX
       xd = zero
       do ixl=ix+1,ix+nx_comm
          fx_y_top = get_nodeby(iy,ixl)
          fy_y_top = fy_y + cuty
          if (ixl == ix+1) then
             ! Treat special case of neighboring cell separately (xd = 0)
             if (fy_y_top - fx_y_top < cuty) fy_y = fx_y_top
          else
             yd = fy_y_top - fx_y_top
             yd = max(zero, yd)              ! Ensures yd >= 0
             dist = (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                fy_y = sqrt(cutsq - (xd*boxx)**2)/boxy - cuty + fx_y_top
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

       ! Coordinates in FY will be further communicated in x direction to zone EZ.
       ! Therefore, we have to expand FY zone to include all coordinates that are needed
       ! in this future communication

       ! This loop expands the FY zone to include coordinates in the EZ zones of the nodes
       ! ix-1:ix-nx_comm
       xd = zero
       do ixl=ix-1,ix-nx_comm,-1
          call get_ez_boundary(ixl, iy, ez_x, ez_y, group_y, q_checkgrouped)
          if (ixl == ix-1) then
             ! Treat neighboring cell case (xd = 0) separately
             fy_y = max(fy_y, ez_y)
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fy_y_top = fy_y + cuty
                group_y_top = group_y + cuty_grouped
                if (group_y_top > fy_y_top) fy_y = group_y_top - cuty
             endif
          else
             fy_y_top = fy_y + cuty
             yd = fy_y_top - ez_y
             yd = max(zero, yd)          ! Ensures yd >= 0
             dist = (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                fy_y = sqrt(cutsq - (xd*boxx)**2)/boxy - cuty + ez_y
             endif
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fy_y_top = fy_y + cuty
                yd = fy_y_top - group_y
                yd = max(zero, yd)          ! Ensures yd >= 0
                dist = (yd*boxy)**2 + (xd*boxx)**2
                if (dist < cutsq_grouped) then
                   fy_y = sqrt(cutsq_grouped - (xd*boxx)**2)/boxy - cuty + group_y
                endif
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

    endif
  */

}

//
// Returns the (ex_y, ex_z) communication origin for the node that is at (ix,iy,iz)
//
void DomdecD2DComm::get_ex_boundary(const int ix, const int iy, const int iz,
				    const double cut, const double cut_grouped,
				    double& ex_y, double& ex_z,
				    double& group_z, bool& q_checkgrouped) {
  /*
    use domdec_grouped,only:q_grouped
    implicit none
    ! Input / Output
    integer, intent(in) :: ix, iy, iz
    real(chm_real), intent(out) :: ex_y, ex_z, group_z
    logical, intent(out) :: q_checkgrouped
    real(chm_real), intent(in) :: cut, cut_grouped
    ! Variables
    integer ixl
    real(chm_real) ex_y_top, ex_z_top
    real(chm_real) fx_y_top, fx_z_top
    real(chm_real) xd, zd, yd, dist
    real(chm_real) cuty, cutz, cutsq, cutsq_grouped, cutz_grouped, cuty_grouped
    real(chm_real) c_x, c_y, c_z, c_group_y, c_group_z, c_group_y_top, group_z_top
  */

  ex_y = get_by(iy, ix);
  ex_z = get_bz(iz, iy, ix);
  q_checkgrouped = false;

  /*
    if (q_load_balance) then
       cutsq = cut*cut
       cuty = cut/boxy
       cutz = cut/boxz
       cutsq_grouped = cut_grouped*cut_grouped
       cuty_grouped = cut_grouped/boxy
       cutz_grouped = cut_grouped/boxz
       ! Determine grouped interaction origin z-coordinate
       group_z = ex_z
       ! This checks for the neighboring cell in the FY zone
       if (ny_comm > 0) group_z = max(group_z, get_nodebz(iz,iy+1,ix))
       ! This checks for the neighboring cell in the FX zone
       if (nx_comm > 0) group_z = max(group_z, get_nodebz(iz,iy,ix+1))
       ! This checks for the neighboring cell in the EZ zone
       if (nx_comm > 0 .and. ny_comm > 0) group_z = max(group_z, get_nodebz(iz,iy+1,ix+1))
       if (q_grouped .and. group_z > ex_z) q_checkgrouped = .true.

       ! This loop expands the EX zone to include all FX-EX interactions
       ! (ex_y, ex_z) = y and z coordinates of the EX zone bottom
       ! (fx_y_top, fx_z_top) = y and z coordinates of the FX zone top
       ! xd = x distance between EX and FX boundary
       xd = zero
       ! Loop through all cells in the FX zone
       do ixl=ix+1,ix+nx_comm
          fx_y_top = get_nodeby(iy, ixl)
          fx_z_top = get_nodebz(iz, iy, ixl)
          ! In order to include all FX-EX interactions, the distance between FX
          ! zone top and the EX zone top must be smaller than cut
          ex_y_top = ex_y + cuty
          ex_z_top = ex_z + cutz
          if (ixl == ix + 1) then
             ! Treat special case xd == 0 separately
             if (ex_y_top - fx_y_top < cuty) ex_y = fx_y_top
             if (ex_z_top - fx_z_top < cutz) ex_z = fx_z_top
          else
             ! Check y-distance
             yd = ex_y_top - fx_y_top
             yd = max(zero, yd)             ! Ensures yd >= 0
             dist = (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                ! Expand EX zone in y-direction
                ex_y = sqrt(cutsq - (xd*boxx)**2)/boxy - cuty + fx_y_top
             endif
             ! Check z-distance
             zd = ex_z_top - fx_z_top
             zd = max(zero, zd)             ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                ! Expand EX zone in z-direction
                ex_z = sqrt(cutsq - (xd*boxx)**2)/boxz - cutz + fx_z_top
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

       ! Part of the EX zone will be communicated further in the x-direction to 
       ! nodes at ix-1:ix-nx_comm as C zone
       ! This loop expands the EX zone so that enough is communicated
       !
       ! xd = x distance between EX and C
       xd = zero
       do ixl=ix-1,ix-nx_comm,-1
          ! We are at node (ixl, iy, iz)
          call get_c_boundary(ixl, iy, iz, c_x, c_y, c_z, c_group_y, c_group_z, &
               q_c_checkgrouped)
          if (ixl == ix-1) then
             ! Treat special case xd = 0 separately
             ex_y = max(ex_y, c_y)
             ex_z = max(ex_z, c_z)
             if (q_c_checkgrouped) then
                ! Increase EX grouped interaction zone if needed
                if (group_z < c_group_z) then
                   group_z = c_group_z
                   q_checkgrouped = .true.
                endif
                ! Increase EX in y-direction if needed
                ex_y_top = ex_y + cuty
                c_group_y_top = c_group_y + cuty_grouped
                if (c_group_y_top > ex_y_top) ex_y = c_group_y_top - cuty
             endif
          else
             ! If EX top z falls within C zone => expand EX in z-direction
             ex_z_top = ex_z + cutz
             zd = ex_z_top - c_z
             zd = max(zero,zd)                    ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                ex_z = sqrt(cutsq - (xd*boxx)**2)/boxz - cutz + c_z
             endif
             ! If EX top y falls within C zone => expand EX in y-direction
             ex_y_top = ex_y + cuty
             yd = ex_y_top - c_y                  ! Ensures yd >= 0
             yd = max(zero, yd)
             dist = (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                ex_y = sqrt(cutsq - (xd*boxx)**2)/boxy - cuty + c_y
             endif
             !
             if (q_c_checkgrouped) then
                ! Increase EX grouped interaction zone if needed
                group_z_top = group_z + cutz_grouped
                zd = group_z_top - c_group_z
                zd = max(zero,zd)                  ! Ensures zd >= 0
                dist = (zd*boxz)**2 + (xd*boxx)**2
                if (dist < cutsq_grouped) then
                   group_z = sqrt(cutsq_grouped - (xd*boxx)**2)/boxz - cutz_grouped + c_group_z
                   q_checkgrouped = .true.
                endif
                ! Increase EX in y-direction if needed
                ex_y_top = ex_y + cuty
                yd = ex_y_top - c_group_y
                yd = max(zero,yd)                  ! Ensures yd >= 0
                dist = (yd*boxy)**2 + (xd*boxx)**2
                if (dist < cutsq_grouped) then
                   ex_y = sqrt(cutsq_grouped - (xd*boxx)**2)/boxy - cuty + c_group_y
                endif
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

    endif
  */
}

//
// Returns the (x,y,z) communication origin for the node that is at (ix,iy,iz)
//
void DomdecD2DComm::get_fx_boundary(const int ix, double& fx_x) {
  fx_x = get_bx(ix);
}

//
// Returns the (ez_x, ez_y) communication origin for the node that is at (ix,iy,iz)
//
void DomdecD2DComm::get_ez_boundary(const int ix, const int iy,
				    double& ez_x, double& ez_y, double& group_y, bool& q_checkgrouped) {

  ez_x = get_bx(ix);
  ez_y = get_by(iy, ix);
  q_checkgrouped = false;

  /*
  if (q_load_balance) {
    group_y = ez_y;
    if (nx_comm > 0) group_y = max(group_y, get_nodeby(iy, ix+1));
    if (q_grouped .and. group_y > ez_y) q_checkgrouped = .true.;
  }
  */

}

//
// Returns the (x,y,z) communication origin for the node that is at (ix,iy,iz)
//
void DomdecD2DComm::get_ey_boundary(const int ix, const int iy, const int iz, const double cut,
				    double& ey_x, double& ey_z, double& group_z, bool& q_checkgrouped) {
  /*
    integer ixl, iyl
    real(chm_real) ey_z_top, ey_y_top
    real(chm_real) fy_z_top, fy_y
    real(chm_real) xd, yd, zd, dist, cutz, cutsq
  */

  ey_x = get_bx(ix);
  ey_z = get_bz(iz, iy, ix);
  q_checkgrouped = false;

  /*
    if (q_load_balance) then
       cutsq = cut*cut
       cutz = cut/boxz
       group_z = ey_z
       ! This checks if the neighboring node (iy,ix+1) in the EY zone is higher
       if (nx_comm > 0) group_z = max(group_z, get_nodebz(iz, iy, ix+1))
       ! This checks if the neighboring node (iy+1,ix) in the FY zone is higher
       ! NOTE: only done if EY and FY overlap
       ! nodeby(iy,ix)   = low y coordinate of FY
       ! nodeby(iy,ix+1) = high y coordinate of EY
       if (nx_comm > 0 .and. ny_comm > 0) then
          if (get_nodeby(iy,ix) < get_nodeby(iy,ix+1)) then
             group_z = max(group_z, get_nodebz(iz, iy+1, ix))
          endif
       endif
       ! This checks if the corner node (iy+1, ix+1) in the C zone is higher
       if (nx_comm > 0 .and. ny_comm > 0) group_z = max(group_z, get_nodebz(iz, iy+1, ix+1))
       if (q_grouped .and. group_z > ey_z) q_checkgrouped = .true.
       
       ! This loop expands the EY zone in z-direction to include all FY-EY interactions
       !
       ! ey_z     = z coordinate of the EY zone bottom
       ! ey_z_top = z coordinate of the EY zone top
       ! ey_y_top  = y coordinate of the EY zone top
       !
       ! fy_z_top = z coordinate of the FY zone top
       ! fy_y     = y coordinate of the FY zone bottom
       !
       ! yd = y distance between FY and EY boundary
       yd = zero
       fy_y = get_nodeby(iy, ix)
       ! Loop through all cells in the FY zone, from iy+1 to iy+ny_comm
       do iyl=iy+1,iy+ny_comm
          fy_z_top = get_nodebz(iz, iyl, ix)
          ! Loop through all x rows of EY zone
          ! (this has to be done because the rows have different y coordinates)
          xd = zero
          do ixl=ix+1,ix+nx_comm
             ey_y_top = get_nodeby(iy, ixl)
             ey_z_top = ey_z + cutz
             ! In order to include all FY-EY interactions, the distance between FY
             ! zone top and EY zone top must be smaller than cut
             yd = fy_y - ey_y_top
             yd = max(zero, yd)               ! Ensures yd >= 0
             zd = ey_z_top - fy_z_top
             zd = max(zero, zd)               ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                ! Expand EY zone in z-direction
                ey_z = sqrt(cutsq - (yd*boxy)**2 - (xd*boxx)**2)/boxz - cutz + fy_z_top
             endif
             xd = xd + nodefrx_pbc(ixl)
          enddo
          fy_y = fy_y + nodefry_pbc(iyl,ix)
       enddo
    endif    
  */
}
  
//
// Returns the (x,y,z) communication origin for the node that is at (ix,iy,iz)
//
void DomdecD2DComm::get_c_boundary(const int ix, const int iy, const int iz,
				   double& c_x, double& c_y, double& c_z,
				   double& group_y, double& group_z, bool& q_checkgrouped) {

  c_x = get_bx(ix);
  c_y = get_by(iy, ix);
  c_z = get_bz(iz, iy, ix);
  q_checkgrouped = false;

  /*
    if (q_load_balance) then
       group_y = c_y
       group_z = c_z
       ! group_y = max between (iy,ix) and (iy,ix+1)
       if (nx_comm > 0) group_y = max(group_y, get_nodeby(iy,ix+1) )
       ! group_z = max between (iy,ix), (iy,ix+1), (iy+1,ix), (iy+1,ix+1)
       if (nx_comm > 0) group_z = max(group_z, get_nodebz(iz,iy,ix+1))
       if (ny_comm > 0) group_z = max(group_z, get_nodebz(iz,iy+1,ix))
       if (nx_comm > 0 .and. ny_comm > 0) group_z = max(group_z, get_nodebz(iz,iy+1,ix+1))
       if (q_grouped .and. (group_y > c_y .or. group_z > c_z)) q_checkgrouped = .true.
    endif
  */
}

//
// Returns the z0 (lowest z coordinate cell within the c zone)
//
void DomdecD2DComm::get_z0_for_c(const int ix, const int iy, const int iz, double& z0) {

  /*
  z0 = get_bz(iz, iy, ix);

  if (q_load_balance) {
    for (int ixl=ix;ixl <= ix+nx_comm;ixl++) {
      for (int iyl=iy;iyl <= iy+ny_comm;iyl++) {
	z0 = std::min(z0, get_bz(iz, iyl, ixl));
      }
    }
  }
  */

}
