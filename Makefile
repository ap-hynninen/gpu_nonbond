
# Detect OS
OS := $(shell uname -s)

DEFS = USE_MPI

CC = mpiicc
CL = mpiicc

SRCS = Grid.cu Bspline.cu XYZQ.cu Matrix3d.cu MultiNodeMatrix3d.cpp Force.cu cuda_utils.cu gpu_recip.cu mpi_utils.cpp
OBJS = Grid.o Bspline.o XYZQ.o Matrix3d.o MultiNodeMatrix3d.o Force.o cuda_utils.o gpu_recip.o mpi_utils.o

ifeq ($(OS),Linux)
LFLAGS = -L /usr/local/cuda-6.0/lib64 -lcudart -lnvToolsExt -lcufft
else
LFLAGS = -L /usr/local/cuda/lib -I /usr/local/cuda/include -lcudart -lcuda -lstdc++.6 -lnvToolsExt
endif

gpu_recip : $(OBJS)
	$(CL) $(LFLAGS) -o gpu_recip $(OBJS)

clean: 
	rm -f *.o
	rm -f *~
	rm -f gpu_recip

%.o : %.cu
	nvcc -c -O3 -arch=sm_20 -fmad=true -use_fast_math -lineinfo -D$(DEFS) $<

%.o : %.cpp
	$(CC) -c -O3 -D$(DEFS) $<
