#ifndef NEIGHBORLIST_H
#define NEIGHBORLIST_H

#include <iostream>
#include <cuda.h>

// Constants
const int n_jlist_max = 64;
const int n_jlist_max_shift = 6;
const int n_jlist_max_mask = (1<<6) - 1;
const int max_ncoord = (1 << (32-n_jlist_max_shift));

template <int tilesize>
struct num_excl {
  static const int val = ((tilesize*tilesize-1)/32 + 1);
};

template <int tilesize>
struct tile_excl_t {
  unsigned int excl[num_excl<tilesize>::val]; // Exclusion mask
};

struct ientry_t {
  int indi;
  int ish;
  int startj;
  int endj;
};

template <int tilesize>
struct pairs_t {
  int i[tilesize];
};

//
// Neighborlist parameter structure
// NOTE: This is used to avoid GPU-CPU-GPU communication
//
struct NeighborListParam_t {

  // Image boundaries: -1, 0, 1
  int imx_lo, imx_hi;
  int imy_lo, imy_hi;
  int imz_lo, imz_hi;

  // Total number of cells
  int ncell;

  // Minimum x and y for each zone
  float3 min_xyz[8];

  // Maximum x and y for each zone
  float3 max_xyz[8];

  // Number of cells for each zone (NOTE: ncellz_max[i] is the maximum value for zone i)
  int ncellx[8];
  int ncelly[8];
  int ncellz_max[8];

  // ncol[izone] = exclusive cumulative sum: sum(ncellx[j]*ncelly[j], j=0...izone-1)
  int ncol[9];

  // xy cell sizes
  float celldx[8];
  float celldy[8];
  float celldz_min[8];

  // Inverse of xy cell sizes
  float inv_celldx[8];
  float inv_celldy[8];

  // Interaction zones
  int zone_patom[9];
  int n_int_zone[8];
  int int_zone[8][8];

  // Starting column for every zone
  int zone_col[8];

  // Starting cell for every zone
  int zone_cell[8];

  // Number of entries in ientry -table
  int n_ientry;

  // Number of tiles
  int n_tile;

  // Number of cell-cell exclusions
  int nexcl;

  // temporary variable, REMOVE AFTER DEBUGGING
  //int tmp;
};

//
// Bounding box structure
//
struct bb_t {
  float x, y, z;      // Center
  float wx, wy, wz;   // Half-width
  friend std::ostream& operator<<(std::ostream &o, const bb_t& b);
};

template<typename AT, typename CT> class CudaPMEDirectForce;
template<typename AT, typename CT> class CudaPMEDirectForceBlock;

template <int tilesize>
class NeighborList {
  friend class CudaPMEDirectForce<long long int, float>;
  friend class CudaPMEDirectForceBlock<long long int, float>;
private:

  // Global total number of atoms in this system
  int ncoord_glo;

  // Number of i tiles
  int n_ientry;

  // Total number of tiles
  int n_tile;

  int tile_excl_len;
  tile_excl_t<tilesize> *tile_excl;

  int ientry_len;
  ientry_t *ientry;

  int tile_indj_len;
  int *tile_indj;

  // Sparse:
  int n_ientry_sparse;
  int n_tile_sparse;

  int pairs_len;
  pairs_t<tilesize> *pairs;
  
  int ientry_sparse_len;
  ientry_t *ientry_sparse;

  int tile_indj_sparse_len;
  int *tile_indj_sparse;

  // ----------------------------------
  // For building neighbor list on GPU
  // ----------------------------------

  // Number of atoms in each column
  int col_natom_len;
  int *col_natom;

  // Cumulative number of atoms in each column
  int col_patom_len;
  int *col_patom;

  // Number of z-cells in each column
  int col_ncellz_len;
  int *col_ncellz;

  // x and y coordinates and zone of each column
  int col_xy_zone_len;
  int3 *col_xy_zone;

  // Starting cell index for each column
  int col_cell_len;
  int* col_cell;

  // Column index of each atom
  int atom_icol_len;
  int *atom_icol;

  // Index sorting performed by this class
  // Usage: ind_sorted[i] = j : Atom j belongs to position i
  int ind_sorted_len;
  int *ind_sorted;

  // Global -> local mapping
  int *glo2loc;

  // Atom indices where each cell start
  int cell_patom_len;
  int *cell_patom;

  // Cell indices for each atom
  int atom_pcell_len;
  int *atom_pcell;

  // (icellx, icelly, icellz, izone) for each cell
  int cell_xyz_zone_len;
  int4 *cell_xyz_zone;

  // Cell z-boundaries
  int cell_bz_len;
  float *cell_bz;

  // Approximate upper bound for number of cells
  int ncell_max;

  // Host copy of parameters
  NeighborListParam_t *h_nlist_param;

  // Maximum value of n_int_zone[]
  int n_int_zone_max;

  // Bounding boxes
  int bb_len;
  bb_t *bb;

  // Cell-cell exclusions:
  int cell_excl_pos_len;
  int *cell_excl_pos;

  int cell_excl_len;
  int *cell_excl;

  // Atom-atom exclusions:
  // For global atom index i, excluded atoms are in
  // atom_excl[ atom_excl_pos[i] ... atom_excl_pos[i+1]-1 ]
  int atom_excl_pos_len;
  int *atom_excl_pos;

  int atom_excl_len;
  int *atom_excl;

  int excl_atom_heap_len;
  int* excl_atom_heap;

  // Maximum number of atom-atom exclusions
  int max_nexcl;

  // Flag for testing neighborlist build
  bool test;

  void get_tile_ientry_est(int *n_int_zone, int int_zone[][8],
			   int *ncellx, int *ncelly, int *ncellz_max,
			   float *celldx, float *celldy, float *celldz_min,
			   float rcut, int &n_tile_est, int &n_ientry_est);

  double calc_volume_overlap(double Ax0, double Ay0, double Az0, 
			     double Ax1, double Ay1, double Az1, double rcut,
			     double Bx0, double By0, double Bz0, 
			     double Bx1, double By1, double Bz1,
			     double& dx, double& dy, double& dz);

  void set_int_zone(const int *zone_natom, int *n_int_zone, int int_zone[][8]);

  void set_cell_sizes(const int *zone_natom,
		      const float3 *max_xyz, const float3 *min_xyz,
		      int *ncellx, int *ncelly, int *ncellz_max,
		      float *celldx, float *celldy, float *celldz_min);

  bool test_z_columns(const int* zone_patom,
		      const int* ncellx, const int* ncelly,
		      const int ncol_tot,
		      const float3* min_xyz,
		      const float* inv_celldx, const float* inv_celldy,
		      const float4* xyzq, const float4* xyzq_sorted,
		      const int* col_patom);

  bool test_sort(const int* zone_patom,
		 const int* ncellx, const int* ncelly,
		 const int ncol_tot, const int ncell_max,
		 const float3* min_xyz,
		 const float* inv_celldx, const float* inv_celldy,
		 const float4* xyzq, const float4* xyzq_sorted,
		 const int* col_patom, const int* cell_patom);

  void test_build(const double boxx, const double boxy, const double boxz,
		  const double rcut, const float4 *xyzq, const int *loc2glo);

  template<typename T>
  int calc_gpu_pairlist(const int n_ientry, const ientry_t* ientry, const int* tile_indj,
			const tile_excl_t<tilesize>* tile_excl,
			const float4* xyzq, const double boxx, const double boxy,
			const double boxz, const double rcut);

  template<typename T>
  int calc_cpu_pairlist(const int* zone_patom, const float4* xyzq,
			const int* loc2glo, const int* atom_excl_pos,
			const int* atom_excl, const double boxx,
			const double boxy, const double boxz,
			const double rcut);
  
  void set_nlist_param(cudaStream_t stream);
  void get_nlist_param();

  void sort_setup(const int *zone_patom,
		  const float3 *max_xyz, const float3 *min_xyz,
		  int &ncol_tot, cudaStream_t stream);

  void sort_alloc_realloc(const int ncol_tot, const int ncoord);

  void sort_build_indices(const int ncoord, float4 *xyzq, int *loc2glo, cudaStream_t stream);

  void sort_core(const int ncol_tot, const int ncoord,
		 float4 *xyzq,
		 float4 *xyzq_sorted,
		 cudaStream_t stream);

  void setup_top_excl(const int ncoord_glo, const int *iblo14, const int *inb14);

  void init(const int nx, const int ny, const int nz);
  void load(const char *filename);

public:
  NeighborList(const int ncoord_glo, const int *iblo14, const int *inb14,
	       const int nx, const int ny, const int nz);
  NeighborList(const int ncoord_glo, const char *filename,
	       const int nx, const int ny, const int nz);
  ~NeighborList();

  void sort(const int *zone_patom,
	    const float3 *max_xyz, const float3 *min_xyz,
	    float4 *xyzq,
	    float4 *xyzq_sorted,
	    int *loc2glo,
	    cudaStream_t stream=0);

  void sort(const int *zone_patom,
	    float4 *xyzq,
	    float4 *xyzq_sorted,
	    int *loc2glo,
	    cudaStream_t stream=0);

  void build(const float boxx, const float boxy, const float boxz,
	     const float rcut,
	     const float4 *xyzq, const int *loc2glo,
	     cudaStream_t stream=0);

  void reset();
  
  void build_excl(const float boxx, const float boxy, const float boxz,
		  const float rcut,
		  const int n_ijlist, const int3 *ijlist,
		  const int *cell_patom,
		  const float4 *xyzq,
		  cudaStream_t stream=0);
  
  void add_tile_top(const int ntile_top, const int *tile_ind_top,
		    const tile_excl_t<tilesize> *tile_excl_top,
		    cudaStream_t stream=0);

  void set_ientry(int n_ientry, ientry_t *h_ientry);

  void split_dense_sparse(int npair_cutoff);
  void remove_empty_tiles();
  void analyze();

  int *get_glo2loc() {return glo2loc;}

  int *get_ind_sorted() {return ind_sorted;}

  void set_test(bool test_in) {this->test = test_in;}

};

#endif // NEIGHBORLIST_H
