#include <iostream>
#include <fstream>
#include <cuda.h>
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "XYZQ.h"
#include "Force.h"
#include "BondedForce.h"
#include "VirialPressure.h"

void test();

int numnode = 1;
int mynode = 0;

int main(int argc, char *argv[]) {

  start_gpu(numnode, mynode);
  
  test();

  return 0;
}

//
// Loads indices from file
//
template <typename T>
void load_ind(const int nind, const char *filename, const int n, T *ind) {
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
// Test the code using data in test_data/ -directory
//
void test() {

  // Settings for the data:
  const double boxx = 62.23;
  const double boxy = 62.23;
  const double boxz = 62.23;
  const int ncoord = 23558;

  const double energy_bond_ref = 715.08289;
  const double energy_ureyb_ref = 167.39536;
  const double energy_angle_ref = 1228.72913;
  const double energy_dihe_ref = 921.88694;
  const double energy_imdihe_ref = 102.07776;

  double sforce_ref[81];
  load_ind<double>(81, "test_data/sforce_bonded.txt", 1, sforce_ref);

  double sforcex[27], sforcey[27], sforcez[27];
  double energy_bond, energy_ureyb, energy_angle, energy_dihe, energy_imdihe, energy_cmap;

  Force<double> force_bonded("test_data/force_bonded.txt");
  Force<long long int> force_fp(ncoord);
  Force<double> force(ncoord);

  // Load coordinates
  XYZQ xyzq("test_data/xyzq.txt", 32);

  const int nbondlist = 23592;
  const int nbondcoef = 129;

  const int nureyblist = 11584;
  const int nureybcoef = 327;

  const int nanglelist = 11584;
  const int nanglecoef = 327;

  const int ndihelist = 6701;
  const int ndihecoef = 438;

  const int nimdihelist = 418;
  const int nimdihecoef = 40;

  const int ncmaplist = 0;
  const int ncmapcoef = 0;

  bondlist_t *h_bondlist = new bondlist_t[nbondlist];
  float2 *h_bondcoef = new float2[nbondcoef];
  load_ind<int>(4, "test_data/bondlist.txt", nbondlist, (int *)h_bondlist);
  load_ind<float>(2, "test_data/bondcoef.txt", nbondcoef, (float *)h_bondcoef);

  bondlist_t *h_ureyblist = new bondlist_t[nureyblist];
  float2 *h_ureybcoef = new float2[nureybcoef];
  load_ind<int>(4, "test_data/ureyblist.txt", nureyblist, (int *)h_ureyblist);
  load_ind<float>(2, "test_data/ureybcoef.txt", nureybcoef, (float *)h_ureybcoef);

  anglelist_t *h_anglelist = new anglelist_t[nanglelist];
  float2 *h_anglecoef = new float2[nanglecoef];
  load_ind<int>(6, "test_data/anglelist.txt", nanglelist, (int *)h_anglelist);
  load_ind<float>(2, "test_data/anglecoef.txt", nanglecoef, (float *)h_anglecoef);

  dihelist_t *h_dihelist = new dihelist_t[ndihelist];
  float4 *h_dihecoef = new float4[ndihecoef];
  load_ind<int>(8, "test_data/dihelist.txt", ndihelist, (int *)h_dihelist);
  load_ind<float>(4, "test_data/dihecoef.txt", ndihecoef, (float *)h_dihecoef);

  dihelist_t *h_imdihelist = new dihelist_t[nimdihelist];
  float4 *h_imdihecoef = new float4[nimdihecoef];
  load_ind<int>(8, "test_data/imdihelist.txt", nimdihelist, (int *)h_imdihelist);
  load_ind<float>(4, "test_data/imdihecoef.txt", nimdihecoef, (float *)h_imdihecoef);

  cmaplist_t *h_cmaplist = NULL; //new cmaplist_t[ncmaplist];
  float2 *h_cmapcoef = NULL;//new float2[ncmaplist];
  //load_ind<int>(8, "test_data/cmaplist_176k.txt", ncmaplist, (int *)h_cmaplist);
  //load_ind<float>(2, "test_data/cmapcoef_176k.txt", ncmaplist, (float *)h_cmapcoef);

  force_fp.clear();

  // Single precision
  {
    force_fp.clear();
    BondedForce<long long int, float> bondedforce;
    bondedforce.clear_energy_virial();
    bondedforce.setup_coef(nbondcoef, h_bondcoef,
			   nureybcoef, h_ureybcoef,
			   nanglecoef, h_anglecoef,
			   ndihecoef, h_dihecoef,
			   nimdihecoef, h_imdihecoef,
			   ncmapcoef, h_cmapcoef);
    bondedforce.setup_list(nbondlist, h_bondlist, 
			   nureyblist, h_ureyblist, 
			   nanglelist, h_anglelist, 
			   ndihelist, h_dihelist, 
			   nimdihelist, h_imdihelist, 
			   ncmaplist, h_cmaplist);

    bondedforce.calc_force(xyzq.xyzq, boxx, boxy, boxz, true, false,
			   force_fp.xyz.stride, force_fp.xyz.data,
			   true, true, true, true, true, true);
    bondedforce.get_energy_virial(true, false,
				  &energy_bond, &energy_ureyb,
				  &energy_angle,
				  &energy_dihe, &energy_imdihe,
				  &energy_cmap,
				  sforcex, sforcey, sforcez);
    force_fp.convert(&force);

    double max_diff;
    double tol = 0.0057;
    if (!force_bonded.compare(&force, tol, max_diff)) {
      std::cout << "(SP) Bonded force comparison FAILED " << std::endl;
    } else {
      std::cout<<"(SP) Bonded force comparison OK (tolerance " << tol << " max difference " 
	       << max_diff << ")" << std::endl;
    }

    max_diff = fabs(energy_bond_ref - energy_bond);
    if (max_diff > tol) {
      std::cout << "(SP) energy_bond comparison FAILED: ref = " << energy_bond_ref 
		<< " energy = " << energy_bond << std::endl;
    } else {
      std::cout << "(SP) energy_bond comparison OK (tolerance " << tol << " difference " 
		<< max_diff << ")" << std::endl;
    }

    max_diff = fabs(energy_ureyb_ref - energy_ureyb);
    if (max_diff > tol) {
      std::cout << "(SP) energy_ureyb comparison FAILED: ref = " << energy_ureyb_ref 
		<< " energy = " << energy_ureyb << std::endl;
    } else {
      std::cout << "(SP) energy_ureyb comparison OK (tolerance " << tol << " difference " 
		<< max_diff << ")" << std::endl;
    }

    max_diff = fabs(energy_angle_ref - energy_angle);
    if (max_diff > tol) {
      std::cout << "(SP) energy_angle comparison FAILED: ref = " << energy_angle_ref 
		<< " energy = " << energy_angle << std::endl;
    } else {
      std::cout << "(SP) energy_angle comparison OK (tolerance " << tol << " difference " 
		<< max_diff << ")" << std::endl;
    }

    max_diff = fabs(energy_dihe_ref - energy_dihe);
    if (max_diff > tol) {
      std::cout << "(SP) energy_dihe comparison FAILED: ref = " << energy_dihe_ref 
		<< " energy = " << energy_dihe << std::endl;
    } else {
      std::cout << "(SP) energy_dihe comparison OK (tolerance " << tol << " difference " 
		<< max_diff << ")" << std::endl;
    }

    max_diff = fabs(energy_imdihe_ref - energy_imdihe);
    if (max_diff > tol) {
      std::cout << "(SP) energy_imdihe comparison FAILED: ref = " << energy_imdihe_ref 
		<< " energy = " << energy_imdihe << std::endl;
    } else {
      std::cout << "(SP) energy_imdihe comparison OK (tolerance " << tol << " difference " 
		<< max_diff << ")" << std::endl;
    }

    force_fp.clear();
    bondedforce.calc_force(xyzq.xyzq, boxx, boxy, boxz, false, false,
			   force_fp.xyz.stride, force_fp.xyz.data,
			   true, true, true, true, true, true);
    force_fp.convert(&force);

    tol = 0.0057;
    if (!force_bonded.compare(&force, tol, max_diff)) {
      std::cout << "(SP) Bonded force comparison FAILED " << std::endl;
    } else {
      std::cout<<"(SP) Bonded force comparison OK (tolerance " << tol << " max difference " 
	       << max_diff << ")" << std::endl;
    }

  }

  // Double precision
  {
    force_fp.clear();
    BondedForce<long long int, double> bondedforce;
    bondedforce.clear_energy_virial();
    bondedforce.setup_coef(nbondcoef, h_bondcoef,
			   nureybcoef, h_ureybcoef,
			   nanglecoef, h_anglecoef,
			   ndihecoef, h_dihecoef,
			   nimdihecoef, h_imdihecoef,
			   ncmapcoef, h_cmapcoef);
    bondedforce.setup_list(nbondlist, h_bondlist, 
			   nureyblist, h_ureyblist, 
			   nanglelist, h_anglelist, 
			   ndihelist, h_dihelist, 
			   nimdihelist, h_imdihelist, 
			   ncmaplist, h_cmaplist);
    bondedforce.calc_force(xyzq.xyzq, boxx, boxy, boxz, true, false,
			   force_fp.xyz.stride, force_fp.xyz.data,
			   true, true, true, true, true, true);
    force_fp.convert(&force);

    double max_diff;
    double tol = 0.0058;
    if (!force_bonded.compare(&force, tol, max_diff)) {
      std::cout<<"(DP) Bonded force comparison FAILED"<<std::endl;
    } else {
      std::cout<<"(DP) Bonded force comparison OK (tolerance " << tol << " max difference " 
	       << max_diff << ")" << std::endl;
    }
  }

  delete [] h_bondlist;
  delete [] h_bondcoef;
  
  delete [] h_ureyblist;
  delete [] h_ureybcoef;
  
  delete [] h_anglelist;
  delete [] h_anglecoef;

  delete [] h_dihelist;
  delete [] h_dihecoef;
  
  delete [] h_imdihelist;
  delete [] h_imdihecoef;

  delete [] h_cmaplist;
  delete [] h_cmapcoef;
  
}
