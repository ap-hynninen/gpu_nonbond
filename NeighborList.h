#ifndef NEIGHBORLIST_H
#define NEIGHBORLIST_H

template <int tilesize>
struct tile_excl_t {
  unsigned int excl[((tilesize*tilesize-1)/32 + 1)]; // Exclusion mask
};

struct ientry_t {
  int indi;
  int ish;
  int startj;
  int endj;
};

template<typename AT, typename CT> class DirectForce;

template <int tilesize>
class NeighborList {
  friend class DirectForce<long long int, float>;
private:

  const static int num_excl = ((tilesize*tilesize-1)/32 + 1);

  int ni;

  int ntot;

  int tile_excl_len;
  tile_excl_t<tilesize> *tile_excl;

  int ientry_len;
  ientry_t *ientry;

  int tile_indj_len;
  int *tile_indj;

public:
  NeighborList();
  ~NeighborList();
  void analyze();
  void load(const char *filename);
};

#endif // NEIGHBORLIST_H
