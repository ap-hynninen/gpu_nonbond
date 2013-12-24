
# Detect OS
OS := $(shell uname -s)

DEFS = DUMMY #USE_MPI

ifeq ($(DEFS),USE_MPI)
CC = mpiicc
CL = mpiicc
else
CC = icc
CL = icc
endif

#SRCS = Grid.cu Bspline.cu XYZQ.cu Matrix3d.cu MultiNodeMatrix3d.cpp Force.cu cuda_utils.cu gpu_recip.cu mpi_utils.cpp DirectForce.cu NeighborList.cu gpu_direct.cu

OBJS_RECIP = Grid.o Bspline.o XYZQ.o Matrix3d.o MultiNodeMatrix3d.o Force.o cuda_utils.o gpu_recip.o mpi_utils.o

OBJS_DIRECT = XYZQ.o Force.o cuda_utils.o mpi_utils.o DirectForce.o NeighborList.o gpu_direct.o

CUDAROOT := $(subst /bin/,,$(dir $(shell which nvcc)))

ifeq ($(OS),Linux)
LFLAGS = -L $(CUDAROOT)/lib64 -lcudart -lnvToolsExt -lcufft
else
LFLAGS = -L /usr/local/cuda/lib -I /usr/local/cuda/include -lcudart -lcuda -lstdc++.6 -lnvToolsExt
endif

all: gpu_recip gpu_direct

gpu_recip : $(OBJS_RECIP)
	$(CL) $(LFLAGS) -o gpu_recip $(OBJS_RECIP)

gpu_direct : $(OBJS_DIRECT)
	$(CL) $(LFLAGS) -o gpu_direct $(OBJS_DIRECT)

clean: 
	rm -f *.o
	rm -f *~
	rm -f gpu_recip
	rm -f gpu_direct

%.o : %.cu
	nvcc -c -O3 -arch=sm_30 -fmad=true -use_fast_math -lineinfo -D$(DEFS) $<

%.o : %.cpp
	$(CC) -c -O3 -D$(DEFS) $<

