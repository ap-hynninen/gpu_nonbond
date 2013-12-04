
# Detect OS
OS := $(shell uname -s)

LFLAGS = 

SRCS = Grid.cu Bspline.cu XYZQ.cu Matrix3d.cu gpu_recip.cu
OBJS = Grid.o Bspline.o XYZQ.o Matrix3d.o gpu_recip.o

ifeq ($(OS),Linux)
LFLAGS += -L /usr/local/cuda-6.0/lib64 -I /usr/local/cuda-6.0/include -lcudart -lnvToolsExt -lcufft
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
