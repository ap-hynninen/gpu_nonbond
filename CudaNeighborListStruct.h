#ifndef CUDANEIGHBORLISTSTRUCT_H
#define CUDANEIGHBORLISTSTRUCT_H

// Constants
const int n_jlist_max = 64;
const int n_jlist_max_shift = 6;
const int n_jlist_max_mask = (1<<6) - 1;
const int max_ncoord = (1 << (32-n_jlist_max_shift));
// Maximum number of interacting zones for each zone
const int maxNumIntZone = 4;
// Maximum number of zones
const int maxNumZone = 8;

//
// Zone parameters
//
struct ZoneParam_t {
  // Minimum x, y, and z
  float3 min_xyz;

  // Maximum x, y, and z
  float3 max_xyz;

  // Number of cells (NOTE: ncellz_max is the maximum value for this zone)
  int ncellx;
  int ncelly;
  int ncellz_max;

  // cell sizes
  float celldx;
  float celldy;
  float celldz_min;

  // Inverse of xy cell sizes
  float inv_celldx;
  float inv_celldy;

  // Number of zones this zone interacts with
  int n_int_zone;
  // List of zones this zone interacts with:
  // int_zone[0..n_int_zone[izone]-1]
  int int_zone[maxNumIntZone];

  // Starting column for this zone
  int zone_col;

  // Number of coordinates in this zone
  int ncoord;
};

//
// Neighborlist parameter structure
// NOTE: This is used to avoid multiple GPU-CPU-GPU communications
//
struct NlistParam_t {

  // Image boundaries: -1, 0, 1
  int imx_lo, imx_hi;
  int imy_lo, imy_hi;
  int imz_lo, imz_hi;

  // ----------------------------------------------------------------------------------
  // NOTE: ncell and col_max_natom must be kept together, hence this union-struct here
  union {
    int2 ncell_col_max_natom;
    struct {
      // Total number of cells
      int ncell;
      // Maximum number of atoms in all columns
      int col_max_natom;
    };
  };
  // ----------------------------------------------------------------------------------

  // Number of entries in ientry -table
  int n_ientry;

  // Number of tiles
  int n_tile;

  // Number of cell-cell exclusions
  int nexcl;
};

//
// Bounding box structure
//
struct bb_t {
  float x, y, z;      // Center
  float wx, wy, wz;   // Half-width
  friend std::ostream& operator<<(std::ostream &o, const bb_t& b);
};

#endif // CUDANEIGHBORLISTSTRUCT_H
