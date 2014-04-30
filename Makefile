
# Detect OS
OS := $(shell uname -s)

DEFS = DUMMY #USE_MPI

ifeq ($(DEFS),USE_MPI)
CC = mpiicc
CL = mpiicc
else
ifeq ($(OS),Linux)
CC = icc
CL = icc
else
CC = gcc
CL = gcc
endif
endif

OBJS_RECIP = Grid.o Bspline.o XYZQ.o Matrix3d.o MultiNodeMatrix3d.o Force.o cuda_utils.o gpu_recip.o mpi_utils.o

OBJS_DIRECT = XYZQ.o Force.o cuda_utils.o mpi_utils.o DirectForce.o NeighborList.o VirialPressure.o BondedForce.o gpu_direct.o

OBJS_BONDED = XYZQ.o Force.o cuda_utils.o VirialPressure.o BondedForce.o gpu_bonded.o

OBJS_CONST = cuda_utils.o gpu_const.o HoloConst.o #const_reduce_lists.o

CUDAROOT := $(subst /bin/,,$(dir $(shell which nvcc)))

ifeq ($(OS),Linux)
LFLAGS = -std=c++0x -L $(CUDAROOT)/lib64 -lcudart -lnvToolsExt -lcufft
else
LFLAGS = -L /usr/local/cuda/lib -I /usr/local/cuda/include -lcudart -lcufft -lcuda -lstdc++.6 -lnvToolsExt
endif

all: gpu_direct gpu_bonded gpu_recip gpu_const

gpu_recip : $(OBJS_RECIP)
	$(CL) $(LFLAGS) -o gpu_recip $(OBJS_RECIP)

gpu_direct : $(OBJS_DIRECT)
	$(CL) $(LFLAGS) -o gpu_direct $(OBJS_DIRECT)

gpu_bonded : $(OBJS_BONDED)
	$(CL) $(LFLAGS) -o gpu_bonded $(OBJS_BONDED)

gpu_const : $(OBJS_CONST)
	$(CL) $(LFLAGS) -o gpu_const $(OBJS_CONST)

clean: 
	rm -f *.o
	rm -f *~
	rm -f gpu_recip
	rm -f gpu_direct
	rm -f gpu_bonded
	rm -f gpu_const

%.o : %.cu
	nvcc -c -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -lineinfo -fmad=true -use_fast_math -D$(DEFS) $<

%.o : %.cpp
	$(CC) -c -O3 -std=c++11 -D$(DEFS) $<

