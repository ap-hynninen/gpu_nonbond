#include <cassert>
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "CudaDomdecGroups.h"

//
// Builds groupTable
// Quick and dirty version using atomicAdd()
//
// groupDataStart[i]                 = start
// groupData[groupDataStart[i]]     = size & type
// groupData[groupDataStart[i]+1]   = index
// groupData[groupDataStart[i]+2..] = (global) coordinates indices
//
// groupTableInd[0...numGroupType-1] = number of groups for each type
// groupTable[type][i]                = groups for type "type"
//
// groupTable[type][] is a list of pointers to the bonds, angles, etc. tables
// groupTablePos[type] is the position in the list groupTable[type][]
//
// coordLoc[0...ncoord_glo-1] = coordinate location 
//
template<bool use_const>
__global__ void buildGroupTable_kernel(const int ncoord,
				       const int* __restrict__ loc2glo,
				       const int* __restrict__ groupDataStart,
				       const int* __restrict__ groupData,
				       const char* __restrict__ coordLoc,  // Random access!
				       int* __restrict__ groupTablePos,
				       int** __restrict__ groupTable,
				       const int typeConstStart=0,
				       int* __restrict__ constTablePos=NULL,
				       int* __restrict__ constTable=NULL) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;

  if (tid < ncoord) {
    int i = loc2glo[tid];
    int j   = groupDataStart[i];
    int end = groupDataStart[i+1];
    while (j < end) {
      // Group type & size:
      // size = high 16 bits, type = low 16 bits
      int size_type  = groupData[j++];
      int size = (size_type >> 16);
      int type = (size_type & 0xffff);
      // Group index
      int bi  = groupData[j++];
      int loc_or = 0;
      // 63 = binary 111111
      int loc_and = 63;
      for (int k=0;k < size;k++) {
	int icoord = groupData[j++];
	int loc = (int)coordLoc[icoord];
	// loc = |zz|yy|xx|, where xx:
	// 00 = this node atomlist does not contain this atom
	// 11 = this node has this atom in the home box
	// 10 = node to the left has this atom. SHOULD NOT HAPPEN
	// 01 = node to the right has this atom
	loc_or |= loc;
	loc_and &= loc;
      }
      // loc_or == 63:  this node will take this group (assuming it has all the coordinates)
      // loc_and == 0:  this node is missing one or more coordinates
      // loc_and == 63: this node has all coordinates in its home box
      if (loc_or == 63 && loc_and != 0) {
	int p = atomicAdd(&groupTablePos[type], 1);
	groupTable[type][p] = bi;
	if (use_const) {
	  if (type >= typeConstStart && loc_and != 63) {
	    // Add to the constraint table. We will go through this in a separate kernel 
	    // to find all the atom indices
	    int pp = atomicAdd(constTablePos, 1);
	    constTable[pp] = j-(size+2);
	  }
	}
      }
    }
  }

}

/*
//
// Add constraint atoms to node table. We have maximum 7 neighbors
//
// Result:
// --------
// nodeTablePos[0...6]
// ineigh = 0...6
// nodeTable[ineigh][0...nodeTablePos[ineigh]-1] = list of global coordinate indices that this node requires
//                                                 from neighbor ineigh
//
__global__ void buildNodeTable_kernel(const int* __restrict__ constTablePos,
				      const int* __restrict__ constTable,
				      const int* __restrict__ groupData,
				      const char* __restrict__ coordLoc,  // Random access!
				      int* __restrict__ nodeTablePos,
				      int** __restrict__ nodeTable) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;

  if (tid < *constTablePos) {
    int j = constTable[tid];
    int size  = (groupData[j++] >> 16);
    // Skip group index
    j++;
    for (int k=0;k < size;k++) {
      int icoord = groupData[j++];
      int loc = (int)coordLoc[icoord];
      // loc = |zz|yy|xx|, where xx:
      // 00 = this node atomlist does not contain this atom
      // 11 = this node has this atom in the home box
      // 10 = node to the left has this atom. SHOULD NOT HAPPEN
      // 01 = node to the right has this atom
      int ix = ((loc & 3) == 1);
      loc >>= 2;
      int iy = ((loc & 3) == 1);
      loc >>= 2;
      int iz = ((loc & 3) == 1);
      // ix = 0 if atom is in the home box
      // ix = 1 if atom is in the box to the "right" (i.e. +1 direction)
      int ineigh = ix + iy*2 + iz*4 - 1;
      // ineigh = 0...6
      if (ineigh >= 0) {
	int p = atomicAdd(&nodeTablePos[ineigh], 1);
	nodeTable[ineigh][p] = icoord;
      }
    }
  }

}
*/

//
// Build coordinate index table. Single thread block does the work
//
template<int maxsize>
__global__ void buildCoordInd_kernel(const int n,
				     const int* __restrict__ constTable,
				     const int* __restrict__ groupData,
				     const char* __restrict__ coordLoc,  // Random access!
				     int* __restrict__ neighPos,
				     int* __restrict__ coordTmp,
				     int* __restrict__ coordInd) {
  // Shared memory
  // Requirement: blockDim.x*max(size)*sizeof(int)
  extern __shared__ int sh_buf[];

  if (threadIdx.x < 8) {
    neighPos[threadIdx.x] = 0;
  }
  __threadfence_block();

  int numCoordTot = 0;
  for (int ibase=0;ibase < n;ibase+=blockDim.x) {
    int i = ibase + threadIdx.x;
    int numCoord = 0;
    int coordTable[maxsize];
    if (i < n) {
      int j = constTable[i];
      int size  = (groupData[j++] >> 16);
      // Skip group index
      j++;
      for (int k=0;k < size;k++) {
	int icoord = groupData[j++];
	int loc = (int)coordLoc[icoord];
	// loc = |zz|yy|xx|, where xx:
	// 00 = this node atomlist does not contain this atom
	// 11 = this node has this atom in the home box
	// 10 = node to the left has this atom. SHOULD NOT HAPPEN
	// 01 = node to the right has this atom
	int ix = ((loc & 3) == 1);
	loc >>= 2;
	int iy = ((loc & 3) == 1);
	loc >>= 2;
	int iz = ((loc & 3) == 1);
	// ix = 0 if atom is in the home box
	// ix = 1 if atom is in the box to the "right" (i.e. +1 direction)
	int ineigh = ix + iy*2 + iz*4 - 1;
	// ineigh = 0...6 (takes 3 bits)
	if (ineigh >= 0) {
	  // NOTE: This limits the total number of coordinates to about 512 million
	  coordTable[numCoord++] = (icoord << 3) | ineigh;
	  atomicAdd(&neighPos[ineigh], 1);
	}
      }
    }

    // Calculate position using inclusive cumulative sum
    sh_buf[threadIdx.x] = numCoord;
    __syncthreads();
    for (int d=1;d < n;d*=2) {
      int t = threadIdx.x - d;
      int val = (t >= 0 && i < n) ? sh_buf[t] : 0;
      __syncthreads();
      if (i < n) sh_buf[threadIdx.x] += val;
      __syncthreads();
    }

    // Store coordTable into global memory coordTmp
    int pos = numCoordTot + sh_buf[threadIdx.x] - numCoord; // "-numCoord" to change into exclusive cumulative sum
    numCoordTot += sh_buf[n-1];
    __syncthreads();
    if (i < n) {
      for (int k=0;k < numCoord;k++) {
        coordTmp[pos+k] = coordTable[k];
      }
    }
    
  }

  // Make sure global memory writes are done
  __threadfence_block();
  // neighPos[0...6] gives the number of coordinates for each neighbor
  // coordTmp[0...numCoordTot-1] contains all the coordinate indices

  // Make a copy of neighPos into sh_buf
  if (threadIdx.x < 7) {
    sh_buf[threadIdx.x] = neighPos[threadIdx.x];
  }
  __syncthreads();

  // Change neighPos[] into position using exclusive cumulative sum
  if (threadIdx.x == 0) {
    int val = neighPos[0];
    neighPos[0] = 0;
    for (int i=1;i < 8;i++) {
      int tmp = neighPos[i];
      neighPos[i] = val + neighPos[i-1];
      val = tmp;
    }
  }
  __syncthreads();
  __threadfence_block();
  // neighPos[0...6] = position where each neighbor starts
  // neighPos[7]     = total number of coordinates (=numCoordTot)

  // Write final result into coordInd
  for (int i=threadIdx.x;i < numCoordTot;i+=blockDim.x) {
    int icoord = coordTmp[i];
    int ineigh = icoord & 7;
    icoord >>= 3;
    int p = atomicAdd(&neighPos[ineigh], 1);
    coordInd[p] = icoord;
  }

  __syncthreads();
  __threadfence_block();
  // Restore neighPos[] back to position where each neighbor starts
  if (threadIdx.x < 7) {
    neighPos[threadIdx.x] -= sh_buf[threadIdx.x];
  }

}

//############################################################################################
//############################################################################################
//############################################################################################

//
// Class creator
//
CudaDomdecGroups::CudaDomdecGroups(const CudaDomdec& domdec) : domdec(domdec) {
  tbl_upto_date = false;

  groupTable = NULL;
  groupDataStart = NULL;
  groupData = NULL;
  groupTablePos = NULL;

  h_groupTablePos = NULL;

  constTablePos = NULL;
  constTable = NULL;
  h_constTablePos = NULL;

  //nodeTablePos = NULL;
  //nodeTable = NULL;
  //h_nodeTable = NULL;
  coordTmpLen = 0;
  coordTmp = NULL;

  coordIndLen = 0;
  coordInd = NULL;

  neighPos = NULL;
  h_neighPos = NULL;
}

//
// Class destructor
//
CudaDomdecGroups::~CudaDomdecGroups() {
  if (groupTable != NULL) deallocate<int*>(&groupTable);
  if (groupDataStart != NULL) deallocate<int>(&groupDataStart);
  if (groupData != NULL) deallocate<int>(&groupData);
  if (groupTablePos != NULL) deallocate<int>(&groupTablePos);
  if (h_groupTablePos != NULL) deallocate_host<int>(&h_groupTablePos);

  if (constTablePos != NULL) deallocate<int>(&constTablePos);
  if (constTable != NULL) deallocate<int>(&constTable);
  if (h_constTablePos != NULL) deallocate_host<int>(&h_constTablePos);

  if (coordTmp != NULL) deallocate<int>(&coordTmp);
  if (coordInd != NULL) deallocate<int>(&coordInd);
  if (neighPos != NULL) deallocate<int>(&neighPos);
  if (h_neighPos != NULL) deallocate_host<int>(&h_neighPos);

  /*
  if (nodeTablePos != NULL) deallocate<int>(&nodeTablePos);
  if (h_nodeTable != NULL) {
    for (int ineigh=0;ineigh < 7;ineigh++) {
      if (h_nodeTable[ineigh] != NULL) deallocate<int>(&h_nodeTable[ineigh]);
    }
    if (nodeTable != NULL) deallocate<int*>(&nodeTable);
    delete [] h_nodeTable;
  }
  */
}

//
// Begin group setup
//
void CudaDomdecGroups::beginGroups() {
  regGroups.clear();
  regGroups.resize(domdec.get_ncoord_glo());
  atomGroups.clear();
  atomGroupVector.clear();
}

//
// Finish group setup
//
void CudaDomdecGroups::finishGroups() {
  h_groupTable.reserve(atomGroups.size());
  atomGroupVector.reserve(atomGroups.size());
  hasConstGroups = false;
  int i=0;
  int constTableLen = 0;
  int nodeTableLen = 0;
  for (std::map<int, AtomGroupBase*>::iterator it=atomGroups.begin();it != atomGroups.end();it++,i++) {
    // Set table size to maximum possible = number of entries in the list
    // This makes a big list but avoid possible overflow
    // TODO: In order to reduce memory footprint, we could count the maximum number of
    // table entries and use it
    AtomGroupBase* p = it->second;
    p->resizeTable(p->get_numGroupList());
    p->set_numTable(p->get_numGroupList());
    h_groupTable.push_back(p->get_table());
    atomGroupVector.push_back(p);
    if (domdec.get_numnode() > 1) {
      int id = it->first;
      if (id >= CONST_START) {
	// Record the start of constraint groups
	if (!hasConstGroups) typeConstStart = i;
	// Worst case = all constraints
	constTableLen += p->get_numGroupList();
	// Worst case = all constraint atoms requested
	nodeTableLen += p->get_numGroupList()*p->get_size();
	hasConstGroups = true;
      } else if (hasConstGroups) {
	std::cout << "CudaDomdecGroups::finishGroups, constraint groups must be added last" << std::endl;
	exit(1);
      }
    }
  }
  allocate<int*>(&groupTable, atomGroups.size());
  copy_HtoD_sync<int*>(h_groupTable.data(), groupTable, atomGroups.size());
  allocate<int>(&groupTablePos, atomGroups.size());
  allocate_host<int>(&h_groupTablePos, atomGroups.size());
  // --------------------------------------------
  // Allocate constraint groups data structures
  // --------------------------------------------
  if (hasConstGroups) {
    allocate<int>(&constTable, constTableLen);
    allocate<int>(&constTablePos, 1);
    allocate<int>(&neighPos, 8);
    allocate_host<int>(&h_neighPos, 8);
    allocate_host<int>(&h_constTablePos, 1);
  }
  /*
  //--------------------------------
  // Build molecules
  //--------------------------------
  bond_t *bond = this->getGroupList<bond_t>(BOND);
  int numBonds = this->getNumGroupList(BOND);
  bond_t *h_bond = new bond_t[numBonds];
  this->buildMolecules(domdec.get_ncoord_glo(), numBonds, h_bond);
  delete [] h_bond;
  */
  //--------------------------------
  // Build groupData etc.
  //--------------------------------
  // Count how much space we need
  int groupDataLen = 0;
  for (int i=0;i < domdec.get_ncoord_glo();i++) {
    groupDataLen += regGroups.at(i).size();
  }
  std::vector<int> h_groupDataStart(domdec.get_ncoord_glo()+1);
  std::vector<int> h_groupData;
  h_groupData.reserve(groupDataLen);
  // Loop through atoms and copy into contiguous data structure in h_groupData
  for (int i=0;i < domdec.get_ncoord_glo();i++) {
    h_groupDataStart.at(i) = h_groupData.size();
    h_groupData.insert(h_groupData.end(), regGroups.at(i).begin(), regGroups.at(i).end());
  }
  h_groupDataStart.at(domdec.get_ncoord_glo()) = h_groupData.size();
  assert(h_groupData.size() == groupDataLen);
  // Allocate device memory, copy, and deallocate host memory
  allocate<int>(&groupData, groupDataLen);
  allocate<int>(&groupDataStart, domdec.get_ncoord_glo()+1);
  copy_HtoD_sync<int>(h_groupDataStart.data(), groupDataStart, domdec.get_ncoord_glo()+1);
  copy_HtoD_sync<int>(h_groupData.data(), groupData, groupDataLen);
  // clear regGroups
  regGroups.clear();
  tbl_upto_date = false;
}

//
// Build tables
//
void CudaDomdecGroups::buildGroupTables(cudaStream_t stream) {

  if (domdec.get_numnode() > 1 || !tbl_upto_date) {
    clear_gpu_array<int>(groupTablePos, atomGroups.size(), stream);

    int nthread = 512;
    int nblock = (domdec.get_ncoord_tot() - 1)/nthread + 1;
    if (domdec.get_numnode() > 1 && hasConstGroups) {
      clear_gpu_array<int>(constTablePos, 1, stream);
      //clear_gpu_array<int>(nodeTablePos, 7, stream);
      buildGroupTable_kernel<true> <<< nblock, nthread, 0, stream >>>
	(domdec.get_ncoord_tot(), domdec.get_loc2glo_ptr(),
	 groupDataStart, groupData, domdec.get_coordLoc(), groupTablePos, groupTable,
	 typeConstStart, constTablePos, constTable);
      cudaCheck(cudaGetLastError());
      copy_DtoH<int>(groupTablePos, h_groupTablePos, atomGroups.size(), stream);
      copy_DtoH<int>(constTablePos, h_constTablePos, 1, stream);

      // Wait while we receive h_constTablePos
      cudaCheck(cudaStreamSynchronize(stream));
      // Maximum number of coordinates in constraint
      const int maxsize = 4;
      int maxNumCoord = (*h_constTablePos)*maxsize;
      reallocate<int>(&coordTmp, &coordTmpLen, maxNumCoord, 1.2f);
      reallocate<int>(&coordInd, &coordIndLen, maxNumCoord, 1.2f);

      nthread = min(min(get_max_nthread(), ((maxNumCoord-1)/warpsize+1)*warpsize ),
		    (int)((get_max_shmem_size()/(maxsize*sizeof(int))-1)/warpsize + 1)*warpsize);
      int shmem_size = nthread*maxsize*sizeof(int);
      /*
      fprintf(stderr,"nthread = %d (max %d) shmem_size = %d h_constTablePos=%d\n",
	      nthread,get_max_nthread(),shmem_size, *h_constTablePos);
      */
      buildCoordInd_kernel<maxsize> <<< 1, nthread, shmem_size, stream >>>
	(*h_constTablePos, constTable, groupData, domdec.get_coordLoc(), neighPos,
	 coordTmp, coordInd);
      cudaCheck(cudaGetLastError());

      copy_DtoH<int>(neighPos, h_neighPos, 8, stream);

      /*
      cudaCheck(cudaDeviceSynchronize());
      fprintf(stderr,"%d: %d |",domdec.get_mynode());
      for (int i=0;i < 8;i++) {
	fprintf(stderr," %d",h_neighPos[i]);
      }
      fprintf(stderr,"\n");
      */

    } else {
      buildGroupTable_kernel<false> <<< nblock, nthread, 0, stream >>>
	(domdec.get_ncoord_tot(), domdec.get_loc2glo_ptr(),
	 groupDataStart, groupData, domdec.get_coordLoc(), groupTablePos, groupTable);
      cudaCheck(cudaGetLastError());
      copy_DtoH<int>(groupTablePos, h_groupTablePos, atomGroups.size(), stream);
    }

  }
}

//
// Synchronizes and sets group table sizes in host memory
//
void CudaDomdecGroups::syncGroupTables(cudaStream_t stream) {
  if (domdec.get_numnode() > 1 || !tbl_upto_date) {
    // Wait for work done in buildGroupTables to finish
    cudaCheck(cudaStreamSynchronize(stream));

    for (std::map<int, AtomGroupBase*>::iterator it=atomGroups.begin();it != atomGroups.end();it++) {
      AtomGroupBase* p = it->second;
      int i = p->get_type();
      p->set_numTable(h_groupTablePos[i]);
    }

    tbl_upto_date = true;
  }
}

