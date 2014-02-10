#ifndef CONST_REDUCE_LISTS_H
#define CONST_REDUCE_LISTS_H

void reduce_lists(int npair, double *pair_constr_in, double *pair_mass_in,
		  int ntrip, double *trip_constr_in, double *trip_mass_in,
		  int nquad, double *quad_constr_in, double *quad_mass_in,
		  int *npair_constr, double **pair_constr, int **pair_constr_indlist,
		  int *npair_mass, double **pair_mass, int **pair_mass_indlist,
		  int *ntrip_constr, double **trip_constr, int **trip_constr_indlist,
		  int *ntrip_mass, double **trip_mass, int **trip_mass_indlist,
		  int *nquad_constr, double **quad_constr, int **quad_constr_indlist,
		  int *nquad_mass, double **quad_mass, int **quad_mass_indlist);

#endif //#ifndef CONST_REDUCE_LISTS_H
