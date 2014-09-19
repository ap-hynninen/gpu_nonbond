#ifndef DOMDECD2DCOMM_H
#define DOMDECD2DCOMM_H

#include <vector>
#include "Domdec.h"

class DomdecD2DComm {

 private:

  // Number of sub-boxes we need to communicate with in each direction
  int nx_comm, ny_comm, nz_comm;

  // True for load balancing runs
  bool load_balance;
  // Load balancing for each coordinate direction
  bool load_balance_x, load_balance_y, load_balance_z;

  // Fractional [0...1] sizes of sub-boxes
  // fx[nx]
  // fy[2*nx_comm+1][ny]
  // fy[2*nx_comm+1][2*ny_comm+1][nz]
  std::vector<double> fx;
  std::vector<std::vector<double> > fy;
  std::vector<std::vector<std::vector<double> > > fz;

  // Fractional [0...1] sizes of sub-boxes boundaries
  // bx[nx+1]
  // by[2*nx_comm+1][ny+1]
  // by[2*nx_comm+1][2*ny_comm+1][nz+1]
  std::vector<double> bx;
  std::vector<std::vector<double> > by;
  std::vector<std::vector<std::vector<double> > > bz;

  // Maximum fractional sub-box sizes allowed by the domain decomposition
  double max_fx, max_fy, max_fz;

  void fill_bx_by_bz();
  void setup_subboxes();

  double get_bz(const int iz, const int iy, const int ix);
  double get_by(const int iy, const int ix);
  double get_bx(const int ix);

 protected:

  // Domdec definition
  Domdec& domdec;

  int get_nx_comm() {return nx_comm;}
  int get_ny_comm() {return ny_comm;}
  int get_nz_comm() {return nz_comm;}

  void get_fz_boundary(const int ix, const int iy, const int iz,
		       const double cut, const double cut_grouped, double& fz_z);
  void get_fy_boundary(const int ix, const int iy, const int iz,
		       const double cut, const double cut_grouped, double& fy_y);

  void get_ex_boundary(const int ix, const int iy, const int iz,
		       const double cut, const double cut_grouped,
		       double& ex_y, double& ex_z,
		       double& group_z, bool& q_checkgrouped);
  void get_fx_boundary(const int ix, double& fx_x);
  void get_ez_boundary(const int ix, const int iy,
		       double& ez_x, double& ez_y, double& group_y, bool& q_checkgrouped);
  void get_ey_boundary(const int ix, const int iy, const int iz, const double cut,
		       double& ey_x, double& ey_z, double& group_z, bool& q_checkgrouped);
  void get_c_boundary(const int ix, const int iy, const int iz,
		      double& c_x, double& c_y, double& c_z,
		      double& group_y, double& group_z, bool& q_checkgrouped);
  void get_z0_for_c(const int ix, const int iy, const int iz, double& z0);

 public:
  
 DomdecD2DComm(Domdec& domdec) : domdec(domdec) {setup_subboxes();}
  ~DomdecD2DComm() {}

};

#endif // DOMDECD2DCOMM_H
