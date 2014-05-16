
#ifndef BONDED_STRUCT_H
#define BONDED_STRUCT_H

struct bond_t {
  int i, j, itype;
};

struct angle_t {
  int i, j, k, itype;
};

struct dihe_t {
  int i, j, k, l, itype;
};

struct cmap_t {
  int i1, j1, k1, l1, i2, j2, k2, l2, itype;
};

struct xx14_t {
  int i, j;
};

// Data structures for bonds, angles, dihedrals, and cmap
struct bondlist_t {
  int i, j, itype, ishift;
};

struct anglelist_t {
  int i, j, k, itype, ishift1, ishift2;
};

struct dihelist_t {
  int i, j, k, l, itype, ishift1, ishift2, ishift3;
};

struct cmaplist_t {
  int i1, j1, k1, l1, i2, j2, k2, l2, itype, ishift1, ishift2, ishift3;
};

struct xx14list_t {
  int i, j, ishift;
};

#endif // BONDED_STRUCT_H
