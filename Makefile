
# Detect OS
OS := $(shell uname -s)

LFLAGS = 

SRCS = Grid.cu Bspline.cu XYZQ.cu Matrix3d.cu Force.cu gpu_recip.cu
OBJS = Grid.o Bspline.o XYZQ.o Matrix3d.o Force.o gpu_recip.o

ifeq ($(OS),Linux)
LFLAGS += -lcudart -lnvToolsExt -lcufft
else
LFLAGS += -L /usr/local/cuda/lib -I /usr/local/cuda/include -lcudart -lcuda -lstdc++.6 -lnvToolsExt
endif

gpu_recip : $(OBJS)
	nvcc $(LFLAGS) -o gpu_recip $(OBJS)

clean: 
	rm -f *.o
	rm -f *~
	rm -f gpu_recip

%.o : %.cu
	nvcc -c -O3 -arch=sm_20 -fmad=true -use_fast_math -lineinfo $<

%.o : %.c++
	icc -c -O3 $<
