
# Detect OS
OS := $(shell uname -s)

YES := $(shell which make | wc -l 2> /dev/null)

# Detect Intel compiler
CUDA_COMPILER := $(shell which nvcc | wc -l 2> /dev/null)
INTEL_COMPILER := $(shell which icc | wc -l 2> /dev/null)
MPI_FOUND := $(shell which mpicc | wc -l 2> /dev/null)

ifeq ($(MPI_FOUND), $(YES))
DEFS := USE_MPI
else
DEFS := DONT_USE_MPI
endif

CCMPI = mpicc
CLMPI = mpicc

ifeq ($(INTEL_COMPILER), $(YES))
CC = icc
CL = icc
else
CC = g++
CL = g++
endif

ifeq ($(INTEL_COMPILER), $(YES))
OPENMP_OPT = -openmp
else
OPENMP_OPT = -fopenmp
endif

OBJS_RECIP = Grid.o Bspline.o XYZQ.o Matrix3d.o Force.o reduce.o cuda_utils.o gpu_recip.o

OBJS_DIRECT = XYZQ.o Force.o reduce.o cuda_utils.o CudaPMEDirectForce.o CudaPMEDirectForceBlock.o NeighborList.o VirialPressure.o BondedForce.o gpu_direct.o

OBJS_BONDED = XYZQ.o Force.o reduce.o cuda_utils.o VirialPressure.o BondedForce.o gpu_bonded.o

OBJS_CONST = cuda_utils.o gpu_const.o HoloConst.o

OBJS_DYNA = cuda_utils.o gpu_dyna.o Force.o reduce.o CudaLeapfrogIntegrator.o CudaPMEForcefield.o NeighborList.o CudaPMEDirectForce.o BondedForce.o Grid.o Matrix3d.o XYZQ.o CudaDomdec.o CudaDomdecBonded.o HoloConst.o

OBJS_TRANSPOSE = cpu_transpose.o mpi_utils.o CpuMultiNodeMatrix3d.o CpuMatrix3d.o

ifeq ($(CUDA_COMPILER), $(YES))
OBJS = $(OBJS_RECIP)
OBJS += $(OBJS_DIRECT)
OBJS += $(OBJS_BONDED)
OBJS += $(OBJS_CONST)
OBJS += $(OBJS_DYNA)
endif
ifeq ($(MPI_FOUND), $(YES))
OBJS += $(OBJS_TRANSPOSE)
endif

ifeq ($(CUDA_COMPILER), $(YES))
CUDAROOT := $(subst /bin/,,$(dir $(shell which nvcc)))
endif

ifeq ($(OS),Linux)
LFLAGS = -std=c++0x
ifeq ($(CUDA_COMPILER), $(YES))
LFLAGS += -L $(CUDAROOT)/lib64 -lcudart -lnvToolsExt -lcufft
endif
else
LFLAGS = -lstdc++.6
ifeq ($(CUDA_COMPILER), $(YES))
LFLAGS += -L /usr/local/cuda/lib -I /usr/local/cuda/include -lcudart -lcufft -lcuda -lnvToolsExt
endif
endif

ifeq ($(INTEL_COMPILER), $(YES))
CFLAGS = -O3 -std=c++0x
else
CFLAGS = -O3 -std=c++0x #-std=c++11
endif

ifeq ($(CUDA_COMPILER), $(YES))
GENCODE_SM20  := -gencode arch=compute_20,code=sm_20
GENCODE_SM30  := -gencode arch=compute_30,code=sm_30
GENCODE_SM35  := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)
endif

ifeq ($(CUDA_COMPILER), $(YES))
exec_targets := gpu_direct gpu_bonded gpu_recip gpu_const gpu_dyna
endif
ifeq ($(MPI_FOUND), $(YES))
exec_targets += cpu_transpose
endif

all: $(exec_targets)

gpu_recip : $(OBJS_RECIP)
	$(CL) $(LFLAGS) -o gpu_recip $(OBJS_RECIP)

gpu_direct : $(OBJS_DIRECT)
	$(CL) $(LFLAGS) -o gpu_direct $(OBJS_DIRECT)

gpu_bonded : $(OBJS_BONDED)
	$(CL) $(LFLAGS) -o gpu_bonded $(OBJS_BONDED)

gpu_const : $(OBJS_CONST)
	$(CL) $(LFLAGS) -o gpu_const $(OBJS_CONST)

gpu_dyna : $(OBJS_DYNA)
	$(CL) $(LFLAGS) -o gpu_dyna $(OBJS_DYNA)

cpu_transpose : $(OBJS_TRANSPOSE)
	$(CCMPI) $(LFLAGS) $(OPENMP_OPT) -o cpu_transpose $(OBJS_TRANSPOSE)

clean: 
	rm -f *.o
	rm -f *.d
	rm -f *~
	rm -f gpu_recip
	rm -f gpu_direct
	rm -f gpu_bonded
	rm -f gpu_const
	rm -f gpu_dyna
	rm -f cpu_transpose

# Pull in dependencies that already exist
-include $(OBJS:.o=.d)

%.o : %.cu
	nvcc -c -O3 $(GENCODE_FLAGS) -lineinfo -fmad=true -use_fast_math -D$(DEFS) $<
	$(CC) -MM $(CFLAGS) $(OPENMP_OPT) -D$(DEFS) $*.cu > $*.d

CpuMultiNodeMatrix3d.o : CpuMultiNodeMatrix3d.cpp
	$(CCMPI) -c $(CFLAGS) $(OPENMP_OPT) -D$(DEFS) $<
	$(CCMPI) -MM $(CFLAGS) $(OPENMP_OPT) -D$(DEFS) $*.cpp > $*.d

MultiNodeMatrix3d.o : MultiNodeMatrix3d.cpp
	$(CCMPI) -c $(CFLAGS) $(OPENMP_OPT) -D$(DEFS) $<
	$(CCMPI) -MM $(CFLAGS) $(OPENMP_OPT) -D$(DEFS) $*.cpp > $*.d

cpu_transpose.o : cpu_transpose.cpp
	$(CCMPI) -c $(CFLAGS) $(OPENMP_OPT) -D$(DEFS) $<
	$(CCMPI) -MM $(CFLAGS) $(OPENMP_OPT) -D$(DEFS) $*.cpp > $*.d

mpi_utils.o : mpi_utils.cpp
	$(CCMPI) -c $(CFLAGS) -D$(DEFS) $<
	$(CCMPI) -MM $(CFLAGS) -D$(DEFS) $*.cpp > $*.d

%.o : %.cpp
	$(CC) -c $(CFLAGS) $(OPENMP_OPT) -D$(DEFS) $<
	$(CC) -MM $(CFLAGS) $(OPENMP_OPT) -D$(DEFS) $*.cpp > $*.d
