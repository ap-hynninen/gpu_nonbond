#include <iostream>
#include <fstream>
#include <cuda.h>
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "HoloConst.h"
#include "hostXYZ.h"
#include "Bonded_struct.h"

void test();

//
// Main
//
int main(int argc, char *argv[]) {

  int numnode = 1;
  int mynode = 0;

  std::vector<int> devices;
  start_gpu(numnode, mynode, devices);
  
  test();

  return 0;
}

//
// Loads (x, y, z) coordinates from file
//
void load_coord(const char *filename, const int n, double *x, double *y, double *z) {

  std::ifstream file(filename);
  if (file.is_open()) {

    int i = 0;
    while (file >> x[i] >> y[i] >> z[i]) i++;

    if (i > n) {
      std::cerr<<"Too many lines in file "<<filename<<std::endl;
      exit(1);
    }

  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }

}

//
// Loads (x) coordinates from file
//
void load_coord(const char *filename, const int n, double *x) {

  std::ifstream file(filename);
  if (file.is_open()) {

    int i = 0;
    while (file >> x[i]) i++;

    if (i > n) {
      std::cerr<<"Too many lines in file "<<filename<<std::endl;
      exit(1);
    }

  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }

}

//
// Loads vector from file
//
template <typename T>
void load_vec(const int nind, const char *filename, const int n, T *ind) {
  std::ifstream file(filename);
  if (file.is_open()) {

    for (int i=0;i < n;i++) {
      for (int k=0;k < nind;k++) {
	if (!(file >> ind[i*nind+k])) {
	  std::cerr<<"Error reading file "<<filename<<std::endl;
	  exit(1);
	}
      }
    }

  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }

}

//
// Loads constraints and masses from file
//
void load_constr_mass(const int nconstr, const int nmass, const char *filename, const int n,
		      double *constr, double *mass) {

  std::ifstream file(filename);
  if (file.is_open()) {

    for (int i=0;i < n;i++) {
      for (int k=0;k < nconstr;k++) {
	if (!(file >> constr[i*nconstr+k])) {
	  std::cerr<<"Error reading file "<<filename<<std::endl;
	  exit(1);
	}
      }
      for (int k=0;k < nmass;k++) {
	if (!(file >> mass[i*nmass+k])) {
	  std::cerr<<"Error reading file "<<filename<<std::endl;
	  exit(1);
	}
      }
    }

  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }

}

//
// Checks SETTLE and SHAKE results
//
bool check_result(const int nind, const int n, const int *ind,
		  const double *x, const double *y, const double *z,
		  const double *x_ref, const double *y_ref, const double *z_ref,
		  const double tol, double &max_diff) {

  double x1, y1, z1;
  double x2, y2, z2;
  double diff;
  int imol, j, i;

  for (imol=0;imol < n;imol++) {
    for (j=0;j < nind;j++) {
      i = ind[imol*nind+j];
      x1 = x[i];
      y1 = y[i];
      z1 = z[i];
      x2 = x_ref[i];
      y2 = y_ref[i];
      z2 = z_ref[i];
      bool ok = true;
      if (isnan(x1) || isnan(y1) || isnan(z1) || isnan(x2) || isnan(y2) || isnan(z2)) ok = false;
      if (ok) {
	diff = max(fabs(x1-x2), max(fabs(y1-y2), fabs(z1-z2)));
	max_diff = max(diff, max_diff);
	if (diff > tol) ok = false;
      }
      if (!ok) {
	std::cout << "comparison FAILED, imol=" << imol << " diff=" << diff << std::endl;
	std::cout << "ind =";
	for (j=0;j < nind;j++) {
	  std::cout << " " << ind[imol*nind+j];
	}
	std::cout << std::endl;
	std::cout << "computed:  " << x1 << " "<< y1 << " "<< z1 << std::endl;
	std::cout << "reference: " << x2 << " "<< y2 << " "<< z2 << std::endl;
	return false;
      }
    }
  }

  return true;
}

//
// Check results
//
void check_results(cudaXYZ<double>& xyz_res, hostXYZ<double>& h_xyz_cor,
		   const int nsolvent, const solvent_t* h_solvent_ind,
		   const int npair, const int* h_pair_ind,
		   const int ntrip, const int* h_trip_ind,
		   const int nquad, const int* h_quad_ind) {

  hostXYZ<double> h_xyz_res(xyz_res);

  double max_diff;
  double tol;

  max_diff = 0.0;
  tol = 5.0e-13;
  if (check_result(3, nsolvent, (int *)h_solvent_ind, h_xyz_res.x(), h_xyz_res.y(), h_xyz_res.z(),
		   h_xyz_cor.x(), h_xyz_cor.y(), h_xyz_cor.z(), tol, max_diff)) {
    std::cout<<"solvent SETTLE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

  max_diff = 0.0;
  tol = 5.0e-14;
  if (check_result(2, npair, h_pair_ind, h_xyz_res.x(), h_xyz_res.y(), h_xyz_res.z(),
		   h_xyz_cor.x(), h_xyz_cor.y(), h_xyz_cor.z(), tol, max_diff)) {
    std::cout<<"pair SHAKE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

  max_diff = 0.0;
  tol = 5.0e-10;
  if (check_result(3, ntrip, h_trip_ind, h_xyz_res.x(), h_xyz_res.y(), h_xyz_res.z(),
		   h_xyz_cor.x(), h_xyz_cor.y(), h_xyz_cor.z(), tol, max_diff)) {
    std::cout<<"trip SHAKE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

  max_diff = 0.0;
  tol = 5.0e-10;
  if (check_result(4, nquad, h_quad_ind, h_xyz_res.x(), h_xyz_res.y(), h_xyz_res.z(),
		   h_xyz_cor.x(), h_xyz_cor.y(), h_xyz_cor.z(), tol, max_diff)) {
    std::cout<<"quad SHAKE OK (tolerance " << tol << " max difference " << 
      max_diff << ")" << std::endl;
  }

}

//
// Test parametric version
//
void test_parametric(const double mO, const double mH, const double rOHsq, const double rHHsq,
		     const int npair, const int* h_pair_ind,
		     const int ntrip, const int* h_trip_ind,
		     const int nquad, const int* h_quad_ind,
		     const int nsolvent, const solvent_t* h_solvent_ind,
		     cudaXYZ<double>& xyz_ref, cudaXYZ<double>& xyz_res, hostXYZ<double>& h_xyz_start) {
  //---------------------------------------------------------------------------
  // Load constraint distances and masses
  double *h_pair_constr = (double *)malloc(npair*sizeof(double));
  double *h_pair_mass = (double *)malloc(npair*2*sizeof(double));
  load_constr_mass(1, 2, "test_data/pair_constr_mass.txt", npair, h_pair_constr, h_pair_mass);

  double *h_trip_constr = (double *)malloc(ntrip*2*sizeof(double));
  double *h_trip_mass = (double *)malloc(ntrip*5*sizeof(double));
  load_constr_mass(2, 5, "test_data/trip_constr_mass.txt", ntrip, h_trip_constr, h_trip_mass);

  double *h_quad_constr = (double *)malloc(nquad*3*sizeof(double));
  double *h_quad_mass = (double *)malloc(nquad*7*sizeof(double));
  load_constr_mass(3, 7, "test_data/quad_constr_mass.txt", nquad, h_quad_constr, h_quad_mass);
  //---------------------------------------------------------------------------

  HoloConst holoconst;

  // Setup
  holoconst.setup_solvent_parameters(mO, mH, rOHsq, rHHsq);
  holoconst.setup_ind_mass_constr(npair, (int2 *)h_pair_ind, h_pair_constr, h_pair_mass,
				  ntrip, (int3 *)h_trip_ind, h_trip_constr, h_trip_mass,
				  nquad, (int4 *)h_quad_ind, h_quad_constr, h_quad_mass,
				  nsolvent, h_solvent_ind);

  // Apply holonomic constraints, result is in xyz_res
  xyz_res.set_data_sync(h_xyz_start);
  holoconst.apply(xyz_ref, xyz_res);
  cudaCheck(cudaDeviceSynchronize());

  free(h_pair_constr);
  free(h_pair_mass);
  free(h_trip_constr);
  free(h_trip_mass);
  free(h_quad_constr);
  free(h_quad_mass);
}

//
// Test indexed version
//
void test_indexed(const double mO, const double mH, const double rOHsq, const double rHHsq,
		  const int npair, const int ntrip, const int nquad,
		  const int nsolvent, const solvent_t* h_solvent_ind,
		  cudaXYZ<double>& xyz_ref, cudaXYZ<double>& xyz_res, hostXYZ<double>& h_xyz_start) {

  const int npair_type = 9;
  const int ntrip_type = 3;
  const int nquad_type = 2;

  double *h_pair_constr = new double[npair_type];
  double *h_pair_mass = new double[npair_type*2];
  load_constr_mass(1, 2, "test_data/pair_types.txt", npair_type, h_pair_constr, h_pair_mass);
  bond_t* h_pair_indtype = new bond_t[npair];
  load_vec<int>(3, "test_data/pair_indtype.txt", npair, (int *)h_pair_indtype);

  double *h_trip_constr = new double[ntrip_type*2];
  double *h_trip_mass = new double[ntrip_type*5];
  load_constr_mass(2, 5, "test_data/trip_types.txt", ntrip_type, h_trip_constr, h_trip_mass);
  angle_t* h_trip_indtype = new angle_t[ntrip];
  load_vec<int>(4, "test_data/trip_indtype.txt", ntrip, (int *)h_trip_indtype);

  double *h_quad_constr = new double[nquad_type*3];
  double *h_quad_mass = new double[nquad_type*7];
  load_constr_mass(3, 7, "test_data/quad_types.txt", nquad_type, h_quad_constr, h_quad_mass);
  dihe_t* h_quad_indtype = new dihe_t[nquad];
  load_vec<int>(5, "test_data/quad_indtype.txt", nquad, (int *)h_quad_indtype);

  // Setup
  HoloConst holoconst;
  holoconst.setup_solvent_parameters(mO, mH, rOHsq, rHHsq);
  holoconst.setup_indexed(npair, h_pair_indtype, npair_type, h_pair_constr, h_pair_mass,
			  ntrip, h_trip_indtype, ntrip_type, h_trip_constr, h_trip_mass,
			  nquad, h_quad_indtype, nquad_type, h_quad_constr, h_quad_mass,
			  nsolvent, h_solvent_ind);

  // Apply holonomic constraints
  xyz_res.set_data_sync(h_xyz_start);
  holoconst.apply(xyz_ref, xyz_res);
  cudaCheck(cudaDeviceSynchronize());

  delete [] h_pair_indtype;
  delete [] h_trip_indtype;
  delete [] h_quad_indtype;

  delete [] h_pair_constr;
  delete [] h_pair_mass;
  delete [] h_trip_constr;
  delete [] h_trip_mass;
  delete [] h_quad_constr;
  delete [] h_quad_mass;
}

/*
//
// Test indexed version with SETTLE for triplets
//
void test_indexed_settle(const double mO, const double mH, const double rOHsq, const double rHHsq,
			 const int npair, const int ntrip, const int nquad,
			 const int nsolvent, const int3* h_solvent_ind,
			 cudaXYZ<double>& xyz0, cudaXYZ<double>& xyz1,
			 hostXYZ<double>& h_xyz0, hostXYZ<double>& h_xyz1) {

  const int npair_type = 9;
  const int ntrip_type = 3;
  const int nquad_type = 2;

  double *h_pair_constr = new double[npair_type];
  double *h_pair_mass = new double[npair_type*2];
  load_constr_mass(1, 2, "test_data/pair_types.txt", npair_type, h_pair_constr, h_pair_mass);
  bond_t* h_pair_indtype = new bond_t[npair];
  load_vec<int>(3, "test_data/pair_indtype.txt", npair, (int *)h_pair_indtype);

  double *h_trip_constr = new double[ntrip_type*2];
  double *h_trip_mass = new double[ntrip_type*5];
  load_constr_mass(2, 5, "test_data/trip_types.txt", ntrip_type, h_trip_constr, h_trip_mass);
  angle_t* h_trip_indtype = new angle_t[ntrip];
  load_vec<int>(4, "test_data/trip_indtype.txt", ntrip, (int *)h_trip_indtype);
  // Merge triplets with solvent
  angle_t* h_settle_ind = new angle_t[nsolvent + ntrip];
  for (int i=0;i < nsolvent;i++) {
    h_settle_ind[i].i     = h_solvent_ind[i].x;
    h_settle_ind[i].j     = h_solvent_ind[i].y;
    h_settle_ind[i].k     = h_solvent_ind[i].z;
    h_settle_ind[i].itype = ntrip_type;
  }
  for (int i=nsolvent;i < nsolvent+ntrip;i++) {
    h_settle_ind[i].i     = h_trip_indtype[i-nsolvent].i;
    h_settle_ind[i].j     = h_trip_indtype[i-nsolvent].j;
    h_settle_ind[i].k     = h_trip_indtype[i-nsolvent].k;
    h_settle_ind[i].itype = h_trip_indtype[i-nsolvent].itype;
  }
  double* h_massP = new double[ntrip_type+1];
  double* h_massH = new double[ntrip_type+1];
  double* h_rPHsq = new double[ntrip_type+1];
  double* h_rHHsq = new double[ntrip_type+1];
  for (int i=0;i < ntrip_type;i++) {
    h_massP[i] = 1.0/h_trip_mass[i*5];
    h_massH[i] = 1.0/h_trip_mass[i*5+1];
    h_rPHsq[i] = h_trip_constr[i*2];
    int j;
    for (j=0;j < ntrip;j++) if (h_trip_indtype[j].itype == i) break;
    int jj = h_trip_indtype[j].j;
    int kk = h_trip_indtype[j].k;
    double xjk = h_xyz0.x()[jj] - h_xyz0.x()[kk];
    double yjk = h_xyz0.y()[jj] - h_xyz0.y()[kk];
    double zjk = h_xyz0.z()[jj] - h_xyz0.z()[kk];
    h_rHHsq[i] = xjk*xjk + yjk*yjk + zjk*zjk;
  }
  h_massP[ntrip_type] = mO;
  h_massH[ntrip_type] = mH;
  h_rPHsq[ntrip_type] = rOHsq;
  h_rHHsq[ntrip_type] = rHHsq;

  double *h_quad_constr = new double[nquad_type*3];
  double *h_quad_mass = new double[nquad_type*7];
  load_constr_mass(3, 7, "test_data/quad_types.txt", nquad_type, h_quad_constr, h_quad_mass);
  dihe_t* h_quad_indtype = new dihe_t[nquad];
  load_vec<int>(5, "test_data/quad_indtype.txt", nquad, (int *)h_quad_indtype);

  // Setup
  HoloConst holoconst;
  holoconst.setup_settle_parameters(ntrip_type+1, h_massP, h_massH, h_rPHsq, h_rHHsq);
  holoconst.setup_indexed(npair, h_pair_indtype, npair_type, h_pair_constr, h_pair_mass,
			  nquad, h_quad_indtype, nquad_type, h_quad_constr, h_quad_mass,
			  nsolvent+ntrip, h_settle_ind);

  // Apply holonomic constraints
  xyz1.set_data_sync(h_xyz1);
  holoconst.apply(xyz0, xyz1);
  xyz1.set_data_sync(h_xyz1);
  holoconst.apply(xyz0, xyz1);
  xyz1.set_data_sync(h_xyz1);
  holoconst.apply(xyz0, xyz1);
  cudaCheck(cudaDeviceSynchronize());

  delete [] h_pair_indtype;
  delete [] h_trip_indtype;
  delete [] h_quad_indtype;

  delete [] h_settle_ind;
  delete [] h_massP;
  delete [] h_massH;
  delete [] h_rPHsq;
  delete [] h_rHHsq;

  delete [] h_pair_constr;
  delete [] h_pair_mass;
  delete [] h_trip_constr;
  delete [] h_trip_mass;
  delete [] h_quad_constr;
  delete [] h_quad_mass;
}
*/

//
// Test the code using data in test_data/ -directory
//
void test() {

  // Settings for the data:
  const double mO = 15.9994;
  const double mH = 1.008;
  const double rOHsq = 0.91623184;
  const double rHHsq = 2.29189321;
  const int ncoord = 23558;
  const int nsolvent = 7023;
  const int npair = 458;
  const int ntrip = 233;
  const int nquad = 99;

  cudaXYZ<double> xyz_ref(ncoord);
  cudaXYZ<double> xyz_res(ncoord);

  // Load coordinates
  hostXYZ<double> h_xyz_ref(ncoord, NON_PINNED);
  hostXYZ<double> h_xyz_start(ncoord, NON_PINNED);
  hostXYZ<double> h_xyz_cor(ncoord, NON_PINNED);
  // Reference coordinates
  load_coord("test_data/xref.txt", h_xyz_ref.size(), h_xyz_ref.x());
  load_coord("test_data/yref.txt", h_xyz_ref.size(), h_xyz_ref.y());
  load_coord("test_data/zref.txt", h_xyz_ref.size(), h_xyz_ref.z());
  // Starting coordinates
  load_coord("test_data/xstart.txt", h_xyz_start.size(), h_xyz_start.x());
  load_coord("test_data/ystart.txt", h_xyz_start.size(), h_xyz_start.y());
  load_coord("test_data/zstart.txt", h_xyz_start.size(), h_xyz_start.z());
  // Correct result coordinates
  load_coord("test_data/xcor.txt", h_xyz_cor.size(), h_xyz_cor.x());
  load_coord("test_data/ycor.txt", h_xyz_cor.size(), h_xyz_cor.y());
  load_coord("test_data/zcor.txt", h_xyz_cor.size(), h_xyz_cor.z());

  // Set reference data, this never changes
  xyz_ref.set_data_sync(h_xyz_ref);

  // Load constraint indices
  solvent_t *h_solvent_ind = new solvent_t[nsolvent];
  load_vec<int>(3, "test_data/solvent_ind.txt", nsolvent, (int *)h_solvent_ind);

  int *h_pair_ind = (int *)malloc(npair*2*sizeof(int));
  load_vec<int>(2, "test_data/pair_ind.txt", npair, h_pair_ind);

  int *h_trip_ind = (int *)malloc(ntrip*3*sizeof(int));
  load_vec<int>(3, "test_data/trip_ind.txt", ntrip, h_trip_ind);

  int *h_quad_ind = (int *)malloc(nquad*4*sizeof(int));
  load_vec<int>(4, "test_data/quad_ind.txt", nquad, h_quad_ind);

  //-------------------------
  // Test parametric
  //-------------------------
  test_parametric(mO, mH, rOHsq, rHHsq, npair, h_pair_ind, ntrip, h_trip_ind, nquad, h_quad_ind,
		  nsolvent, h_solvent_ind, xyz_ref, xyz_res, h_xyz_start);
  check_results(xyz_res, h_xyz_cor, nsolvent, h_solvent_ind, npair, h_pair_ind,
		ntrip, h_trip_ind, nquad, h_quad_ind);

  //-------------------------
  // Test indexed
  //-------------------------
  test_indexed(mO, mH, rOHsq, rHHsq, npair, ntrip, nquad, nsolvent, h_solvent_ind,
	       xyz_ref, xyz_res, h_xyz_start);
  check_results(xyz_res, h_xyz_cor, nsolvent, h_solvent_ind, npair, h_pair_ind,
  		ntrip, h_trip_ind, nquad, h_quad_ind);

  delete [] h_solvent_ind;

  free(h_pair_ind);
  free(h_trip_ind);
  free(h_quad_ind);


}
