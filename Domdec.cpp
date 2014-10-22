#include <iostream>
#include <cassert>
#include <algorithm>
#include "Domdec.h"
#include "mpi_utils.h"

//
// Class creator
//
Domdec::Domdec(int ncoord_glo, double boxx, double boxy, double boxz, double rnl,
	       int nx, int ny, int nz, int mynode, MPI_Comm comm) : 
  ncoord_glo(ncoord_glo), boxx(boxx), boxy(boxy), boxz(boxz),
  rnl(rnl), nx(nx), ny(ny), nz(nz), numnode(nx*ny*nz),
  mynode(mynode), comm(comm) {

  // Setup (homeix, homeiy, homeiz)
  int m = mynode;
  homeiz = m/(nx*ny);
  m -= homeiz*(nx*ny);
  homeiy = m/nx;
  m -= homeiy*nx;
  homeix = m;

  fx.resize(nx);
  bx.resize(nx+1);
  fx.assign(nx, 1.0/(double)nx);

  fy.resize(nx);
  by.resize(nx);
  for (int ix=0;ix < nx;ix++) {
    fy.at(ix).resize(ny);
    by.at(ix).resize(ny+1);
    fy.at(ix).assign(ny, 1.0/(double)ny);
  }
  
  fz.resize(nx);
  bz.resize(nx);
  for (int ix=0;ix < nx;ix++) {
    fz.at(ix).resize(ny);
    bz.at(ix).resize(ny);
    for (int iy=0;iy < ny;iy++) {
      fz.at(ix).at(iy).resize(nz);
      bz.at(ix).at(iy).resize(nz+1);
      fz.at(ix).at(iy).assign(nz, 1.0/(double)nz);
    }
  }
  
  update_bxyz();
  
}

// Returns the node index for box (ix, iy, iz)
// NOTE: deals correctly with periodic boundary conditions
int Domdec::get_nodeind_pbc(const int ix, const int iy, const int iz) {
  // ixt = 0...nx-1
  //int ixt = (ix + (abs(ix)/nx)*nx) % nx;
  //int iyt = (iy + (abs(iy)/ny)*ny) % ny;
  //int izt = (iz + (abs(iz)/nz)*nz) % nz;
  int ixt = ix;
  while (ixt < 0) ixt += nx;
  while (ixt >= nx) ixt -= nx;
  int iyt = iy;
  while (iyt < 0) iyt += ny;
  while (iyt >= ny) iyt -= ny;
  int izt = iz;
  while (izt < 0) izt += nz;
  while (izt >= nz) izt -= nz;
  
  return ixt + iyt*nx + izt*nx*ny;
}

//
// Builds global loc2glo mapping:
// loc2glo_glo = mapping (size ncoord_glo)
// nrecv       = number of coordinates we receive from each node     (size numnode)
// precv       = exclusive cumulative sum of nrecv, used as postiion (size numnode)
//
void Domdec::buildGlobal_loc2glo(int* loc2glo, int* loc2glo_glo, int* nrecv, int* precv) {
  int nsend = get_ncoord();
  
  MPICheck(MPI_Allgather(&nsend, 1, MPI_INT, nrecv, 1, MPI_INT, comm));

  precv[0] = 0;
  for (int i=1;i < numnode;i++) precv[i] = precv[i-1] + nrecv[i-1];

  assert(precv[numnode-1] + nrecv[numnode-1] == ncoord_glo);

  MPICheck(MPI_Allgatherv(loc2glo, nsend, MPI_INT, loc2glo_glo, nrecv, precv, MPI_INT, comm));

}

//
// Combines data among all nodes using the global loc2glo mapping
// xrecvbuf = temporary receive buffer (size ncoord_glo)
// x        = send buffer (size ncoord)
// xglo     = final global buffer (size ncoord_glo)
//
void Domdec::combineData(int* loc2glo_glo, int* nrecv, int* precv,
			 double *xrecvbuf, double *x, double *xglo) {
  int nsend = get_ncoord();

  MPICheck(MPI_Allgatherv(x, nsend, MPI_DOUBLE, xrecvbuf, nrecv, precv, MPI_DOUBLE, comm));

  for (int i=0;i < ncoord_glo;i++) {
    int j = loc2glo_glo[i];
    xglo[j] = xrecvbuf[i];
  }

}

//
// Check the total number of atom groups
//
bool Domdec::checkNumGroups(std::vector<AtomGroupBase*>& atomGroupVector) {

  numGroups.resize(atomGroupVector.size());
  for (int i=0;i < atomGroupVector.size();i++) {
    numGroups.at(i) = atomGroupVector.at(i)->get_numTable();
  }

  numGroupsTot.resize(atomGroupVector.size());

  MPICheck(MPI_Allreduce(numGroups.data(), numGroupsTot.data(), atomGroupVector.size(),
			 MPI_INT, MPI_SUM, comm));

  bool ok = true;
  for (int i=0;i < atomGroupVector.size();i++) {
    if (!checkGroup(*atomGroupVector.at(i), numGroupsTot.at(i))) ok = false;
  }

  return ok;
}

//
// Check group and output missing/double group members
//
bool Domdec::checkGroup(AtomGroupBase& atomGroup, const int numTot) {
  bool ok = true;

  if (numTot != atomGroup.get_numGroupList()) {
    ok = false;
    if (mynode == 0) {
      std::cout << "Domdec::checkGroup, number of groups does not match for "
		<< atomGroup.get_name() << ":"<< std::endl;
      std::cout << "Counted=" << numTot
		<< " Correct=" << atomGroup.get_numGroupList() << std::endl;
    }
    if (numTot < atomGroup.get_numGroupList()) {
      // Get host version of Group table
      std::vector<int> tableVec;
      atomGroup.getGroupTableVec(tableVec);

      std::vector<int> tableVecTot;
      if (mynode == 0) tableVecTot.resize(numTot);

      // Concatenate all group tables into root node
      MPI_Concat(tableVec.data(), tableVec.size(), tableVecTot.data(), 0, comm);

      if (mynode == 0) {
	// Print out the missing groups
	std::sort(tableVecTot.begin(), tableVecTot.end());
	std::cout << "Missing Groups:" << std::endl;
	int j=0;
	for (int i=0;i < tableVecTot.size();i++,j++) {
	  if (j != tableVecTot.at(i)) {
	    std::cout << j << " [";
	    atomGroup.printGroup(j);
	    std::cout << "]" << std::endl;
	    j++;
	  }
	}
      }

    }
  }

  return ok;
}

//
// Combines heuristic flags among all nodes
//
bool Domdec::checkHeuristic(const bool heuristic) {
  int heuristic_in = (int)heuristic;
  int heuristic_out;
  MPICheck(MPI_Allreduce(&heuristic_in, &heuristic_out, 1, MPI_INT, MPI_LOR, comm));
  return (bool)heuristic_out;
}

//
// Copies fractional boundary buffer a contigiuous buffer of size 78
// 
//
void Domdec::copy_lohi_buf(double *buf) {

  int p=0;
  for (int x=-1;x <= 1;x++) {
    int xt = (homeix+x+nx) % nx - homeix;
    buf[p++] = (x == 0 || xt != 0) ? get_lo_bx(x) : 2.0;
  }
  for (int x=-1;x <= 1;x++) {
    int xt = (homeix+x+nx) % nx - homeix;
    buf[p++] = (x == 0 || xt != 0) ? get_hi_bx(x) : 2.0;
  }

  for (int x=-1;x <= 1;x++) {
    int xt = (homeix+x+nx) % nx - homeix;
    for (int y=-1;y <= 1;y++) {
      int yt = (homeiy+y+ny) % ny - homeiy;
      buf[p++] = ((x == 0 || xt != 0) && (y == 0 || yt != 0)) ? get_lo_by(x,y) : 2.0;
    }
  }

  for (int x=-1;x <= 1;x++) {
    int xt = (homeix+x+nx) % nx - homeix;
    for (int y=-1;y <= 1;y++) {
      int yt = (homeiy+y+ny) % ny - homeiy;
      buf[p++] = ((x == 0 || xt != 0) && (y == 0 || yt != 0)) ? get_hi_by(x,y) : 2.0;
    }
  }

  for (int x=-1;x <= 1;x++) {
    int xt = (homeix+x+nx) % nx - homeix;
    for (int y=-1;y <= 1;y++) {
      int yt = (homeiy+y+ny) % ny - homeiy;
      for (int z=-1;z <= 1;z++) {
	int zt = (homeiz+z+nz) % nz - homeiz;
	buf[p++] = ((x == 0 || xt != 0) && 
		    (y == 0 || yt != 0) && 
		    (z == 0 || zt != 0)) ? get_lo_bz(x,y,z) : 2.0;
	//if (buf[p-1] != 2.0) fprintf(stderr,"xyz=%d %d %d\n",x,y,z);
      }
    }
  }

  for (int x=-1;x <= 1;x++) {
    int xt = (homeix+x+nx) % nx - homeix;
    for (int y=-1;y <= 1;y++) {
      int yt = (homeiy+y+ny) % ny - homeiy;
      for (int z=-1;z <= 1;z++) {
	int zt = (homeiz+z+nz) % nz - homeiz;
	buf[p++] = ((x == 0 || xt != 0) && 
		    (y == 0 || yt != 0) && 
		    (z == 0 || zt != 0)) ? get_hi_bz(x,y,z) : 2.0;
      }
    }
  }

  assert(p == 78);

  /*
  if (mynode == 0) {

    p=0;
    for (int x=-1;x <= 1;x++) {
      fprintf(stderr,"%lf ",buf[p++]);
    }
    fprintf(stderr,"\n");
    for (int x=-1;x <= 1;x++) {
      fprintf(stderr,"%lf ",buf[p++]);
    }
    fprintf(stderr,"\n");
    
    for (int x=-1;x <= 1;x++) {
      for (int y=-1;y <= 1;y++) {
	fprintf(stderr,"%lf ",buf[p++]);
      }
    }
    fprintf(stderr,"\n");
    
    for (int x=-1;x <= 1;x++) {
      for (int y=-1;y <= 1;y++) {
	fprintf(stderr,"%lf ",buf[p++]);
      }
    }
    fprintf(stderr,"\n");
    
    for (int x=-1;x <= 1;x++) {
      for (int y=-1;y <= 1;y++) {
	for (int z=-1;z <= 1;z++) {
	  fprintf(stderr,"%lf ",buf[p++]);
	}
      }
    }
    fprintf(stderr,"\n");
    
    for (int x=-1;x <= 1;x++) {
      for (int y=-1;y <= 1;y++) {
	for (int z=-1;z <= 1;z++) {
	  fprintf(stderr,"%lf ",buf[p++]);
	}
      }
    }
    fprintf(stderr,"\n");
    
  }
  */
}
