
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

#endif // BONDED_STRUCT_H
