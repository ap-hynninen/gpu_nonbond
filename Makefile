
# Detect OS
OS := $(shell uname -s)

YES := $(shell which make | wc -l 2> /dev/null)

# Set optimization level
OPTLEV = -O3

# Detect CUDA, Intel compiler, and MPI
CUDA_COMPILER := $(shell which nvcc | wc -l 2> /dev/null)
INTEL_COMPILER := $(shell which icc | wc -l 2> /dev/null)
MPI_FOUND := $(shell which mpicc | wc -l 2> /dev/null)

ifeq ($(MPI_FOUND), $(YES))

DEFS = -D USE_MPI

ifeq ($(INTEL_COMPILER), $(YES))
CC = mpicc
CL = mpicc
DEFS += -D MPICH_IGNORE_CXX_SEEK
else
CC = mpic++
CL = mpic++
endif

else

DEFS = -D DONT_USE_MPI

echo $(YES)
echo $(INTEL_COMPILER)

ifeq ($(INTEL_COMPILER), $(YES))
CC = icc
CL = icc
DEFS += -D MPICH_IGNORE_CXX_SEEK
else
CC = g++
CL = g++
endif

endif

OBJS_RECIP = Grid.o Bspline.o XYZQ.o Matrix3d.o Force.o reduce.o cuda_utils.o gpu_recip.o

OBJS_DIRECT = XYZQ.o Force.o reduce.o cuda_utils.o CudaPMEDirectForce.o CudaPMEDirectForceBlock.o NeighborList.o VirialPressure.o BondedForce.o gpu_direct.o

OBJS_BONDED = XYZQ.o Force.o reduce.o cuda_utils.o VirialPressure.o BondedForce.o gpu_bonded.o

OBJS_CONST = cuda_utils.o gpu_const.o HoloConst.o

OBJS_DYNA = cuda_utils.o gpu_dyna.o Force.o reduce.o CudaLeapfrogIntegrator.o CudaPMEForcefield.o NeighborList.o CudaPMEDirectForce.o BondedForce.o Grid.o Matrix3d.o XYZQ.o CudaDomdec.o CudaDomdecBonded.o HoloConst.o CudaDomdecHomezone.o CudaMPI.o mpi_utils.o CudaDomdecD2DComm.o DomdecD2DComm.o DomdecRecipComm.o CudaDomdecRecipComm.o CudaDomdecRecipLooper.o

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
CUDAROOT = $(subst /bin/,,$(dir $(shell which nvcc)))
endif

ifeq ($(MPI_FOUND), $(YES))
MPIROOT = $(subst /bin/,,$(dir $(shell which mpicc)))
endif

ifeq ($(INTEL_COMPILER), $(YES))
OPENMP_OPT = -openmp
else
OPENMP_OPT = -fopenmp
endif

ifeq ($(CUDA_COMPILER), $(YES))
GENCODE_SM20  := -gencode arch=compute_20,code=sm_20
GENCODE_SM30  := -gencode arch=compute_30,code=sm_30
GENCODE_SM35  := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)
endif

# CUDA_CFLAGS = flags for compiling CUDA API calls using c compiler
# NVCC_CFLAGS = flags for nvcc compiler
# CUDA_LFLAGS = flags for linking with CUDA

CUDA_CFLAGS = -I${CUDAROOT}/include $(OPTLEV) -std=c++0x
NVCC_CFLAGS = $(OPTLEV) -lineinfo -fmad=true -use_fast_math $(GENCODE_FLAGS)
MPI_CFLAGS = -I${MPIROOT}/include

ifeq ($(OS),Linux)
CUDA_LFLAGS = -L$(CUDAROOT)/lib64
else
CUDA_LFLAGS = -L$(CUDAROOT)/lib
endif
CUDA_LFLAGS += -lcudart -lcufft -lnvToolsExt

ifeq ($(CUDA_COMPILER), $(YES))
BINARIES = gpu_direct gpu_bonded gpu_recip gpu_const gpu_dyna
endif
ifeq ($(MPI_FOUND), $(YES))
BINARIES += cpu_transpose
endif

all: $(BINARIES)

gpu_recip : $(OBJS_RECIP)
	$(CL) $(CUDA_LFLAGS) -o gpu_recip $(OBJS_RECIP)

gpu_direct : $(OBJS_DIRECT)
	$(CL) $(CUDA_LFLAGS) -o gpu_direct $(OBJS_DIRECT)

gpu_bonded : $(OBJS_BONDED)
	$(CL) $(CUDA_LFLAGS) -o gpu_bonded $(OBJS_BONDED)

gpu_const : $(OBJS_CONST)
	$(CL) $(CUDA_LFLAGS) -o gpu_const $(OBJS_CONST)

gpu_dyna : $(OBJS_DYNA)
	$(CL) $(OPTLEV) $(CUDA_LFLAGS) -o gpu_dyna $(OBJS_DYNA)

cpu_transpose : $(OBJS_TRANSPOSE)
	$(CL) $(CUDA_LFLAGS) -o cpu_transpose $(OBJS_TRANSPOSE)

clean: 
	rm -f *.o
	rm -f *.d
	rm -f *~
	rm -f $(BINARIES)

# Pull in dependencies that already exist
-include $(OBJS:.o=.d)

%.o : %.cu
	nvcc -c $(MPI_CFLAGS) $(NVCC_CFLAGS) $(DEFS) $<
	nvcc -M $(MPI_CFLAGS) $(NVCC_CFLAGS) $(DEFS) $*.cu > $*.d

%.o : %.cpp
	$(CC) -c $(CUDA_CFLAGS) $(DEFS) $<
	$(CC) -MM $(CUDA_CFLAGS) $(DEFS) $*.cpp > $*.d
