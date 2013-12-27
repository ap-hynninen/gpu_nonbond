#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "NeighborList.h"

//
// Class creator
//
template <int tilesize>
NeighborList<tilesize>::NeighborList() {
  ni = 0;
  ntot = 0;

  tile_excl = NULL;
  tile_excl_len = 0;

  ientry = NULL;
  ientry_len = 0;

  tile_indj = NULL;
  tile_indj_len = 0;

  // Sparse
  ni_sparse = 0;
  ntot_sparse = 0;

  pairs_len = 0;
  pairs = NULL;
  
  ientry_sparse_len = 0;
  ientry_sparse = NULL;

  tile_indj_sparse_len = NULL;
  tile_indj_sparse = NULL;
}

//
// Class destructor
//
template <int tilesize>
NeighborList<tilesize>::~NeighborList() {
  if (tile_excl != NULL) deallocate< tile_excl_t<tilesize> > (&tile_excl);
  if (ientry != NULL) deallocate<ientry_t>(&ientry);
  if (tile_indj != NULL) deallocate<int>(&tile_indj);
  // Sparse
  if (pairs != NULL) deallocate< pairs_t<tilesize> > (&pairs);
  if (ientry_sparse != NULL) deallocate<ientry_t>(&ientry_sparse);
  if (tile_indj_sparse != NULL) deallocate<int>(&tile_indj_sparse);
}

/*
static unsigned int count_1bits(unsigned int xin)
{
  unsigned int x = xin;

  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = x + (x >> 8);
  x = x + (x >> 16);
  return x & 0x0000003F;
}
*/

static int BitCount(unsigned int u)
 {
         unsigned int uCount;

         uCount = u
                  - ((u >> 1) & 033333333333)
                  - ((u >> 2) & 011111111111);
         return
           ((uCount + (uCount >> 3))
            & 030707070707) % 63;
 }

/*
static int BitCount_ref(unsigned int u) {
  unsigned int x = u;
  int res = 0;
  while (x != 0) {
    res += (x & 1);
    x >>= 1;
  }
  return res;
}
*/

//
// Splits neighbor list into dense and sparse parts
//
template <int tilesize>
void NeighborList<tilesize>::split_dense_sparse(int npair_cutoff) {

  ientry_t *h_ientry = new ientry_t[ni];
  int *h_tile_indj = new int[ntot];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[ntot];

  ientry_t *h_ientry_dense = new ientry_t[ni];
  int *h_tile_indj_dense = new int[ntot];
  tile_excl_t<tilesize> *h_tile_excl_dense = new tile_excl_t<tilesize>[ntot];

  ientry_t *h_ientry_sparse = new ientry_t[ni];
  int *h_tile_indj_sparse = new int[ntot];
  pairs_t<tilesize> *h_pairs = new pairs_t<tilesize>[ntot];

  copy_DtoH<ientry_t>(ientry, h_ientry, ni);
  copy_DtoH<int>(tile_indj, h_tile_indj, ntot);
  copy_DtoH< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, ntot);

  int ni_dense = 0;
  int ntot_dense = 0;
  ni_sparse = 0;
  ntot_sparse = 0;
  for (int i=0;i < ni;i++) {
    bool sparse_i_tiles = true;
    int startj_dense = ntot_dense;
    for (int j=h_ientry[i].startj;j <= h_ientry[i].endj;j++) {
      int npair = 0;
      for (int k=0;k < num_excl;k++) {
	unsigned int n1bit = BitCount(h_tile_excl[j].excl[k]);
	npair += 32 - n1bit;
      }

      if (npair <= npair_cutoff) {
	// Sparse
	for (int k=0;k < num_excl;k++) {
	  
	}
	h_tile_indj_sparse[ntot_sparse] = h_tile_indj[j];
	ntot_sparse++;
      } else {
	// Dense
	for (int k=0;k < num_excl;k++) {
	  h_tile_excl_dense[ntot_dense].excl[k] = h_tile_excl[j].excl[k];
	}
	h_tile_indj_dense[ntot_dense] = h_tile_indj[j];
	ntot_dense++;
	sparse_i_tiles = false;
      }

    }

    if (sparse_i_tiles) {
      // Sparse
    } else {
      h_ientry_dense[ni_dense] = h_ientry[i];
      h_ientry_dense[ni_dense].startj = startj_dense;
      h_ientry_dense[ni_dense].endj = ntot_dense - 1;
      ni_dense++;
    }
  }

  ni = ni_dense;
  ntot = ntot_dense;

  copy_HtoD<ientry_t>(h_ientry_dense, ientry, ni);
  copy_HtoD<int>(h_tile_indj_dense, tile_indj, ntot);
  copy_HtoD< tile_excl_t<tilesize> >(h_tile_excl_dense, tile_excl, ntot);

  allocate<ientry_t>(&ientry_sparse, ni_sparse);
  allocate<int>(&tile_indj_sparse, ntot_sparse);
  allocate< pairs_t<tilesize> >(&pairs, ntot_sparse);
  ientry_sparse_len = ni_sparse;
  tile_indj_sparse_len = ntot_sparse;
  pairs_len = ntot_sparse;

  copy_HtoD<ientry_t>(h_ientry_sparse, ientry_sparse, ni_sparse);
  copy_HtoD<int>(h_tile_indj_sparse, tile_indj_sparse, ntot_sparse);
  copy_HtoD< pairs_t<tilesize> >(h_pairs, pairs, ntot_sparse);

  delete [] h_ientry;
  delete [] h_tile_indj;
  delete [] h_tile_excl;

  delete [] h_ientry_dense;
  delete [] h_tile_indj_dense;
  delete [] h_tile_excl_dense;

  delete [] h_ientry_sparse;
  delete [] h_tile_indj_sparse;
  delete [] h_pairs;

}

//
// Removes empty tiles
//
template <int tilesize>
void NeighborList<tilesize>::remove_empty_tiles() {

  ientry_t *h_ientry = new ientry_t[ni];
  int *h_tile_indj = new int[ntot];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[ntot];

  ientry_t *h_ientry_noempty = new ientry_t[ni];
  int *h_tile_indj_noempty = new int[ntot];
  tile_excl_t<tilesize> *h_tile_excl_noempty = new tile_excl_t<tilesize>[ntot];

  copy_DtoH<ientry_t>(ientry, h_ientry, ni);
  copy_DtoH<int>(tile_indj, h_tile_indj, ntot);
  copy_DtoH< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, ntot);

  int ni_noempty = 0;
  int ntot_noempty = 0;
  for (int i=0;i < ni;i++) {
    bool empty_i_tiles = true;
    int startj_noempty = ntot_noempty;
    for (int j=h_ientry[i].startj;j <= h_ientry[i].endj;j++) {
      bool empty_tile = true;
      for (int k=0;k < num_excl;k++) {
	unsigned int n1bit = BitCount(h_tile_excl[j].excl[k]);
	if (n1bit != 32) empty_tile = false;
      }

      if (!empty_tile) {
	for (int k=0;k < num_excl;k++) {
	  h_tile_excl_noempty[ntot_noempty].excl[k] = h_tile_excl[j].excl[k];
	}
	h_tile_indj_noempty[ntot_noempty] = h_tile_indj[j];
	ntot_noempty++;
	empty_i_tiles = false;
      }
    }

    if (!empty_i_tiles) {
      h_ientry_noempty[ni_noempty] = h_ientry[i];
      h_ientry_noempty[ni_noempty].startj = startj_noempty;
      h_ientry_noempty[ni_noempty].endj = ntot_noempty - 1;
      ni_noempty++;
    }
  }

  ni = ni_noempty;
  ntot = ntot_noempty;

  copy_HtoD<ientry_t>(h_ientry_noempty, ientry, ni);
  copy_HtoD<int>(h_tile_indj_noempty, tile_indj, ntot);
  copy_HtoD< tile_excl_t<tilesize> >(h_tile_excl_noempty, tile_excl, ntot);

  delete [] h_ientry;
  delete [] h_tile_indj;
  delete [] h_tile_excl;

  delete [] h_ientry_noempty;
  delete [] h_tile_indj_noempty;
  delete [] h_tile_excl_noempty;

}

//
// Analyzes the neighbor list and prints info
//
template <int tilesize>
void NeighborList<tilesize>::analyze() {

  ientry_t *h_ientry = new ientry_t[ni];
  int *h_tile_indj = new int[ntot];
  tile_excl_t<tilesize> *h_tile_excl = new tile_excl_t<tilesize>[ntot];

  copy_DtoH<ientry_t>(ientry, h_ientry, ni);
  copy_DtoH<int>(tile_indj, h_tile_indj, ntot);
  copy_DtoH< tile_excl_t<tilesize> >(tile_excl, h_tile_excl, ntot);

  std::cout << "Number of i-tiles = " << ni << ", total number of tiles = " << ntot << std::endl;

  std::ofstream file_npair("npair.txt", std::ofstream::out);
  std::ofstream file_nj("nj.txt", std::ofstream::out);

  unsigned int nexcl_bit = 0;
  unsigned int nexcl_bit_self = 0;
  unsigned int nempty_tile = 0;
  unsigned int nempty_line = 0;
  for (int i=0;i < ni;i++) {
    file_nj << h_ientry[i].endj - h_ientry[i].startj + 1 << std::endl;
    for (int j=h_ientry[i].startj;j <= h_ientry[i].endj;j++) {
      int npair = 0;
      bool empty_tile = true;
      for (int k=0;k < num_excl;k++) {
	unsigned int n1bit = BitCount(h_tile_excl[j].excl[k]);

	if (n1bit > 32) {
	  std::cerr << n1bit << " " << std::hex << h_tile_excl[j].excl[k] << std::endl;
	  exit(1);
	}

	if (n1bit == 32)
	  nempty_line++;
	else
	  empty_tile = false;

	nexcl_bit += n1bit;
	npair += 32 - n1bit;

	if (h_ientry[i].indi == h_tile_indj[j]) nexcl_bit_self += n1bit;
      }
      if (empty_tile) nempty_tile++;
      file_npair << npair << std::endl;
    }
  }

  file_npair.close();
  file_nj.close();

  unsigned int ntot_pairs = ntot*tilesize*tilesize;
  std::cout << "Total number of pairs = " << ntot_pairs << std::endl;
  std::cout << "Number of excluded pairs = " << nexcl_bit << " (" << 
    ((double)nexcl_bit*100)/(double)ntot_pairs << "%)" << std::endl;
  std::cout << "Number of excluded pairs in self (i==j) tiles = " << nexcl_bit_self << " (" << 
    ((double)nexcl_bit_self*100)/(double)ntot_pairs << "%)" << std::endl;
  std::cout << "Number of empty lines = " << nempty_line << " (" <<
    ((double)nempty_line*100)/((double)(ntot*tilesize)) << "%)" << std::endl;
  std::cout << "Number of empty tiles = " << nempty_tile << " (" <<
    ((double)nempty_tile*100)/(double)ntot << "%)" << std::endl;

  delete [] h_ientry;
  delete [] h_tile_indj;
  delete [] h_tile_excl;

}

//
// Load neighbor list from file
//
template <int tilesize>
void NeighborList<tilesize>::load(const char *filename) {

  ientry_t *h_ientry;
  int *h_tile_indj;
  tile_excl_t<tilesize> *h_tile_excl;

  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(filename);

    file >> ni >> ntot;

    h_ientry = new ientry_t[ni];
    h_tile_indj = new int[ntot];
    h_tile_excl = new tile_excl_t<tilesize>[ntot];

    for (int i=0;i < ni;i++) {
      file >> std::dec >> h_ientry[i].indi >> h_ientry[i].ish >> 
	h_ientry[i].startj >> h_ientry[i].endj;
      for (int j=h_ientry[i].startj;j <= h_ientry[i].endj;j++) {
	file >> std::dec >> h_tile_indj[j];
	for (int k=0;k < num_excl;k++) {
	  file >> std::hex >> h_tile_excl[j].excl[k];
	}
      }
    }

    file.close();
  }
  catch(std::ifstream::failure e) {
    std::cerr << "Error opening/reading/closing file " << filename << std::endl;
    exit(1);
  }

  reallocate<ientry_t>(&ientry, &ientry_len, ni, 1.2f);
  reallocate<int>(&tile_indj, &tile_indj_len, ntot, 1.2f);
  reallocate< tile_excl_t<tilesize> >(&tile_excl, &tile_excl_len, ntot, 1.2f);

  copy_HtoD<ientry_t>(h_ientry, ientry, ni);
  copy_HtoD<int>(h_tile_indj, tile_indj, ntot);
  copy_HtoD< tile_excl_t<tilesize> >(h_tile_excl, tile_excl, ntot);

  delete [] h_ientry;
  delete [] h_tile_indj;
  delete [] h_tile_excl;
}

//
// Explicit instances of DirectForce
//
template class NeighborList<16>;
template class NeighborList<32>;
