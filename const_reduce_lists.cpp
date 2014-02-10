#include <iostream>
#include "ReduceList.h"

void reduce_lists(int npair, double *pair_constr_in, double *pair_mass_in,
		  int ntrip, double *trip_constr_in, double *trip_mass_in,
		  int nquad, double *quad_constr_in, double *quad_mass_in,
		  int *npair_constr, double **pair_constr, int **pair_constr_indlist,
		  int *npair_mass, double **pair_mass, int **pair_mass_indlist,
		  int *ntrip_constr, double **trip_constr, int **trip_constr_indlist,
		  int *ntrip_mass, double **trip_mass, int **trip_mass_indlist,
		  int *nquad_constr, double **quad_constr, int **quad_constr_indlist,
		  int *nquad_mass, double **quad_mass, int **quad_mass_indlist) {

  ReduceList<double> rl1;
  ReduceList< doublen<2> > rl2;
  ReduceList< doublen<3> > rl3;
  ReduceList< doublen<5> > rl5;
  ReduceList< doublen<7> > rl7;

  rl1.reduce(npair, pair_constr_in,
	     npair_constr, pair_constr, pair_constr_indlist);

  rl2.reduce(npair, (doublen<2> *)pair_mass_in,
	     npair_mass, (doublen<2> **)pair_mass, pair_mass_indlist);

  rl2.reduce(ntrip, (doublen<2> *)trip_constr_in,
	     ntrip_constr, (doublen<2> **)trip_constr, trip_constr_indlist);

  rl5.reduce(ntrip, (doublen<5> *)trip_mass_in,
	     ntrip_mass, (doublen<5> **)trip_mass, trip_mass_indlist);

  rl3.reduce(nquad, (doublen<3> *)quad_constr_in,
	     nquad_constr, (doublen<3> **)quad_constr, quad_constr_indlist);

  rl7.reduce(nquad, (doublen<7> *)quad_mass_in,
	     nquad_mass, (doublen<7> **)quad_mass, quad_mass_indlist);

}
