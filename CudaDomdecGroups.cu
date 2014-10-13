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
__global__ void buildGroupTable_kernel(const int ncoord,
				       const int* __restrict__ loc2glo,
				       const int* __restrict__ groupDataStart,
				       const int* __restrict__ groupData,
				       const char* __restrict__ coordLoc,  // Random access!
				       int* __restrict__ groupTablePos,
				       int** __restrict__ groupTable) {
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
      int loc_and = 7;
      for (int k=0;k < size;k++) {
	int icoord = groupData[j++];
	int loc = (int)coordLoc[icoord];
	loc_or |= loc;
	loc_and &= loc;
      }
      // loc_and == 0: this node does not have all the coordinates
      // 7 = binary 111
      if (loc_or == 7 && loc_and != 0) {
	int p = atomicAdd(&groupTablePos[type], 1);
	groupTable[type][p] = bi;
      }
    }
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
  for (std::map<int, AtomGroupBase*>::iterator it=atomGroups.begin();it != atomGroups.end();it++) {
    // Set table size to maximum possible = number of entries in the list
    // This makes a big list but avoid possible overflow
    // TODO: In order to reduce memory footprint, we could count the maximum number of
    // table entries and use it
    AtomGroupBase* p = it->second;
    p->resizeTable(p->get_numGroupList());
    p->set_numTable(p->get_numGroupList());
    h_groupTable.push_back(p->get_table());
    atomGroupVector.push_back(p);
  }
  allocate<int*>(&groupTable, atomGroups.size());
  copy_HtoD_sync<int*>(h_groupTable.data(), groupTable, atomGroups.size());
  allocate<int>(&groupTablePos, atomGroups.size());
  allocate_host<int>(&h_groupTablePos, atomGroups.size());
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
    buildGroupTable_kernel<<< nblock, nthread, 0, stream >>>
      (domdec.get_ncoord_tot(), domdec.get_loc2glo_ptr(),
       groupDataStart, groupData, domdec.get_coordLoc(), groupTablePos, groupTable);
    cudaCheck(cudaGetLastError());

    copy_DtoH<int>(groupTablePos, h_groupTablePos, atomGroups.size(), stream);
  }
}

//
// Synchronizes and sets group table sizes in host memory
//
void CudaDomdecGroups::syncGroupTables(cudaStream_t stream) {
  if (domdec.get_numnode() > 1 || !tbl_upto_date) {
    cudaCheck(cudaStreamSynchronize(stream));

    for (std::map<int, AtomGroupBase*>::iterator it=atomGroups.begin();it != atomGroups.end();it++) {
      AtomGroupBase* p = it->second;
      int i = p->get_type();
      p->set_numTable(h_groupTablePos[i]);
    }

    tbl_upto_date = true;
  }
}
