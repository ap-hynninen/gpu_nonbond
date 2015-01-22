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

  std::vector<int> devices;
  start_gpu(numnode, mynode, devices);
  
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

struct results_t {
  double energy_bond;
  double energy_ureyb;
  double energy_angle;
  double energy_dihe;
  double energy_imdihe;
  double energy_cmap;
  double sforce[27*3];
};

void check_results(const bool calc_energy, const bool calc_virial,
		   const results_t &results_ref, const results_t &results,
		   Force<double> &force_ref, Force<double> &force) {
  double max_diff;
  double tol = 0.0058;
  if (!force_ref.compare(force, tol, max_diff)) {
    std::cout << "Bonded force comparison FAILED " << std::endl;
  } else {
    std::cout<<"Bonded force comparison OK (tolerance " << tol << " max difference " 
	     << max_diff << ")" << std::endl;
  }

  if (calc_energy) {
    tol = 0.0007;
    max_diff = fabs(results_ref.energy_bond - results.energy_bond);
    if (max_diff > tol) {
      std::cout << "energy_bond comparison FAILED: ref = " << results_ref.energy_bond
		<< " energy = " << results.energy_bond << std::endl;
    } else {
      std::cout << "energy_bond comparison OK (tolerance " << tol << " difference " 
		<< max_diff << ")" << std::endl;
    }

    max_diff = fabs(results_ref.energy_ureyb - results.energy_ureyb);
    if (max_diff > tol) {
      std::cout << "energy_ureyb comparison FAILED: ref = " << results_ref.energy_ureyb
		<< " energy = " << results.energy_ureyb << std::endl;
    } else {
      std::cout << "energy_ureyb comparison OK (tolerance " << tol << " difference " 
		<< max_diff << ")" << std::endl;
    }

    max_diff = fabs(results_ref.energy_angle - results.energy_angle);
    if (max_diff > tol) {
      std::cout << "energy_angle comparison FAILED: ref = " << results_ref.energy_angle
		<< " energy = " << results.energy_angle << std::endl;
    } else {
      std::cout << "energy_angle comparison OK (tolerance " << tol << " difference " 
		<< max_diff << ")" << std::endl;
    }

    max_diff = fabs(results_ref.energy_dihe - results.energy_dihe);
    if (max_diff > tol) {
      std::cout << "energy_dihe comparison FAILED: ref = " << results_ref.energy_dihe
		<< " energy = " << results.energy_dihe << std::endl;
    } else {
      std::cout << "energy_dihe comparison OK (tolerance " << tol << " difference " 
		<< max_diff << ")" << std::endl;
    }

    max_diff = fabs(results_ref.energy_imdihe - results.energy_imdihe);
    if (max_diff > tol) {
      std::cout << "energy_imdihe comparison FAILED: ref = " << results_ref.energy_imdihe
		<< " energy = " << results.energy_imdihe << std::endl;
    } else {
      std::cout << "energy_imdihe comparison OK (tolerance " << tol << " difference " 
		<< max_diff << ")" << std::endl;
    }
  }

  if (calc_virial) {
    max_diff = 0.0;
    tol = 0.16;
    for (int i=0;i < 27*3;i++) {
      max_diff = max(max_diff, fabs(results_ref.sforce[i] - results.sforce[i]));
      if (max_diff > tol) {
	printf("sforce_ref[%d] = %e\n",i,results_ref.sforce[i]);
	printf("sforce[%d]     = %e\n",i,results.sforce[i]);
	break;
      }
    }
    if (max_diff > tol) {
      std::cout << "sforce comparison FAILED" << std::endl;
    } else {
      std::cout << "sforce comparison OK (tolerance " << tol << " difference " 
		<< max_diff << ")" << std::endl;
    }
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

  results_t results_ref;
  results_t results;
  
  results_ref.energy_bond = 715.08289;
  results_ref.energy_ureyb = 167.39536;
  results_ref.energy_angle = 1228.72913;
  results_ref.energy_dihe = 921.88694;
  results_ref.energy_imdihe = 102.07776;

  // Load reference virial
  load_ind<double>(27*3, "test_data/sforce_bonded.txt", 1, results_ref.sforce);
  // Zero the center element of sforce that doesn't contribute to the virial
  results_ref.sforce[39] = 0.0;
  results_ref.sforce[40] = 0.0;
  results_ref.sforce[41] = 0.0;
   
  Force<double> force_ref("test_data/force_bonded.txt");
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

  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << " Single Precision - Fixed Precision" << std::endl;
  std::cout << "--------------------------------------------------" << std::endl;

  // Single precision
  {
    BondedForce<long long int, float> bondedforce;
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

    // Loop through all four possibilities
    for (int bb=0;bb <= 3;bb++) {
      bool calc_energy = ((bb & 1) == 1);
      bool calc_virial = ((bb & 2) == 2);
      std::cout << "calc_energy, calc_virial = " << calc_energy << " " << calc_virial << std::endl;
      force_fp.clear();
      bondedforce.clear_energy_virial();
      bondedforce.calc_force(xyzq.xyzq, boxx, boxy, boxz, calc_energy, calc_virial,
			     force_fp.stride(), force_fp.xyz(),
			     true, true, true, true, true, true);
      bondedforce.get_energy_virial(calc_energy, calc_virial,
				    &results.energy_bond, &results.energy_ureyb,
				    &results.energy_angle,
				    &results.energy_dihe, &results.energy_imdihe,
				    &results.energy_cmap,
				    results.sforce);
      force_fp.convert(force);
      check_results(calc_energy, calc_virial, results_ref, results, force_ref, force);
    }
    
  }

  std::cout << "--------------------------------------------------" << std::endl;
  std::cout << " Double Precision - Fixed Precision" << std::endl;
  std::cout << "--------------------------------------------------" << std::endl;
  
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
    
    // Loop through all four possibilities
    for (int bb=0;bb <= 3;bb++) {
      bool calc_energy = ((bb & 1) == 1);
      bool calc_virial = ((bb & 2) == 2);
      std::cout << "calc_energy, calc_virial = " << calc_energy << " " << calc_virial << std::endl;
      force_fp.clear();
      bondedforce.clear_energy_virial();    
      bondedforce.calc_force(xyzq.xyzq, boxx, boxy, boxz, calc_energy, calc_virial,
			     force_fp.stride(), force_fp.xyz(),
			     true, true, true, true, true, true);
      bondedforce.get_energy_virial(calc_energy, calc_virial,
				    &results.energy_bond, &results.energy_ureyb,
				    &results.energy_angle,
				    &results.energy_dihe, &results.energy_imdihe,
				    &results.energy_cmap,
				    results.sforce);
      force_fp.convert(force);
      check_results(calc_energy, calc_virial, results_ref, results, force_ref, force);
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
