
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

SRC = BondedForce.cu NeighborList.cu Bspline.cu VirialPressure.cu CudaDomdec.cu	XYZQ.cu CudaLeapfrogIntegrator.cu cuda_utils.cu CudaPMEForcefield.cu DirectForce.cu gpu_bonded.cu gpu_const.cu Force.cu	gpu_direct.cu Grid.cu gpu_dyna.cu HoloConst.cu gpu_recip.cu Matrix3d.cu MultiNodeMatrix3d.cpp mpi_utils.cpp CudaDomdecBonded.cu

OBJS_RECIP = Grid.o Bspline.o XYZQ.o Matrix3d.o MultiNodeMatrix3d.o Force.o cuda_utils.o gpu_recip.o mpi_utils.o

OBJS_DIRECT = XYZQ.o Force.o cuda_utils.o mpi_utils.o DirectForce.o NeighborList.o VirialPressure.o BondedForce.o gpu_direct.o

OBJS_BONDED = XYZQ.o Force.o cuda_utils.o VirialPressure.o BondedForce.o gpu_bonded.o

OBJS_CONST = cuda_utils.o gpu_const.o HoloConst.o

OBJS_DYNA = cuda_utils.o gpu_dyna.o Force.o CudaLeapfrogIntegrator.o CudaPMEForcefield.o NeighborList.o DirectForce.o BondedForce.o Grid.o Matrix3d.o XYZQ.o CudaDomdec.o CudaDomdecBonded.o HoloConst.o

CUDAROOT := $(subst /bin/,,$(dir $(shell which nvcc)))

ifeq ($(OS),Linux)
LFLAGS = -std=c++0x -L $(CUDAROOT)/lib64 -lcudart -lnvToolsExt -lcufft
else
LFLAGS = -L /usr/local/cuda/lib -I /usr/local/cuda/include -lcudart -lcufft -lcuda -lstdc++.6 -lnvToolsExt
endif

#CUDAARCH = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35
CUDAARCH = -arch=compute_30

all: gpu_direct gpu_bonded gpu_recip gpu_const gpu_dyna

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

clean: 
	rm -f *.o
	rm -f *~
	rm -f gpu_recip
	rm -f gpu_direct
	rm -f gpu_bonded
	rm -f gpu_const
	rm -f gpu_dyna

depend:
	makedepend $(SRC)

%.o : %.cu
	nvcc -c -O3 $(CUDAARCH) -lineinfo -fmad=true -use_fast_math -D$(DEFS) $<

%.o : %.cpp
	$(CC) -c -O3 -std=c++11 -D$(DEFS) $<

# DO NOT DELETE

BondedForce.o: cuda_utils.h gpu_utils.h /usr/include/stdio.h
BondedForce.o: /usr/include/sys/cdefs.h /usr/include/sys/_symbol_aliasing.h
BondedForce.o: /usr/include/sys/_posix_availability.h
BondedForce.o: /usr/include/Availability.h
BondedForce.o: /usr/include/AvailabilityInternal.h /usr/include/_types.h
BondedForce.o: /usr/include/sys/_types.h /usr/include/machine/_types.h
BondedForce.o: /usr/include/i386/_types.h /usr/include/sys/_types/_va_list.h
BondedForce.o: /usr/include/sys/_types/_size_t.h
BondedForce.o: /usr/include/sys/_types/_null.h
BondedForce.o: /usr/include/sys/_types/_off_t.h
BondedForce.o: /usr/include/sys/_types/_ssize_t.h
BondedForce.o: /usr/include/secure/_stdio.h /usr/include/secure/_common.h
BondedForce.o: BondedForce.h Bonded_struct.h
NeighborList.o: gpu_utils.h /usr/include/stdio.h /usr/include/sys/cdefs.h
NeighborList.o: /usr/include/sys/_symbol_aliasing.h
NeighborList.o: /usr/include/sys/_posix_availability.h
NeighborList.o: /usr/include/Availability.h
NeighborList.o: /usr/include/AvailabilityInternal.h /usr/include/_types.h
NeighborList.o: /usr/include/sys/_types.h /usr/include/machine/_types.h
NeighborList.o: /usr/include/i386/_types.h /usr/include/sys/_types/_va_list.h
NeighborList.o: /usr/include/sys/_types/_size_t.h
NeighborList.o: /usr/include/sys/_types/_null.h
NeighborList.o: /usr/include/sys/_types/_off_t.h
NeighborList.o: /usr/include/sys/_types/_ssize_t.h
NeighborList.o: /usr/include/secure/_stdio.h /usr/include/secure/_common.h
NeighborList.o: cuda_utils.h NeighborList.h
Bspline.o: /usr/include/math.h /usr/include/sys/cdefs.h
Bspline.o: /usr/include/sys/_symbol_aliasing.h
Bspline.o: /usr/include/sys/_posix_availability.h /usr/include/Availability.h
Bspline.o: /usr/include/AvailabilityInternal.h gpu_utils.h
Bspline.o: /usr/include/stdio.h /usr/include/_types.h
Bspline.o: /usr/include/sys/_types.h /usr/include/machine/_types.h
Bspline.o: /usr/include/i386/_types.h /usr/include/sys/_types/_va_list.h
Bspline.o: /usr/include/sys/_types/_size_t.h /usr/include/sys/_types/_null.h
Bspline.o: /usr/include/sys/_types/_off_t.h
Bspline.o: /usr/include/sys/_types/_ssize_t.h /usr/include/secure/_stdio.h
Bspline.o: /usr/include/secure/_common.h cuda_utils.h Bspline.h
VirialPressure.o: /usr/include/math.h /usr/include/sys/cdefs.h
VirialPressure.o: /usr/include/sys/_symbol_aliasing.h
VirialPressure.o: /usr/include/sys/_posix_availability.h
VirialPressure.o: /usr/include/Availability.h
VirialPressure.o: /usr/include/AvailabilityInternal.h gpu_utils.h
VirialPressure.o: /usr/include/stdio.h /usr/include/_types.h
VirialPressure.o: /usr/include/sys/_types.h /usr/include/machine/_types.h
VirialPressure.o: /usr/include/i386/_types.h
VirialPressure.o: /usr/include/sys/_types/_va_list.h
VirialPressure.o: /usr/include/sys/_types/_size_t.h
VirialPressure.o: /usr/include/sys/_types/_null.h
VirialPressure.o: /usr/include/sys/_types/_off_t.h
VirialPressure.o: /usr/include/sys/_types/_ssize_t.h
VirialPressure.o: /usr/include/secure/_stdio.h /usr/include/secure/_common.h
VirialPressure.o: cuda_utils.h VirialPressure.h cudaXYZ.h XYZ.h Force.h
VirialPressure.o: hostXYZ.h
CudaDomdec.o: gpu_utils.h /usr/include/stdio.h /usr/include/sys/cdefs.h
CudaDomdec.o: /usr/include/sys/_symbol_aliasing.h
CudaDomdec.o: /usr/include/sys/_posix_availability.h
CudaDomdec.o: /usr/include/Availability.h /usr/include/AvailabilityInternal.h
CudaDomdec.o: /usr/include/_types.h /usr/include/sys/_types.h
CudaDomdec.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
CudaDomdec.o: /usr/include/sys/_types/_va_list.h
CudaDomdec.o: /usr/include/sys/_types/_size_t.h
CudaDomdec.o: /usr/include/sys/_types/_null.h
CudaDomdec.o: /usr/include/sys/_types/_off_t.h
CudaDomdec.o: /usr/include/sys/_types/_ssize_t.h /usr/include/secure/_stdio.h
CudaDomdec.o: /usr/include/secure/_common.h CudaDomdec.h Decomp.h cudaXYZ.h
CudaDomdec.o: cuda_utils.h XYZ.h Force.h hostXYZ.h
XYZQ.o: cuda_utils.h gpu_utils.h /usr/include/stdio.h
XYZQ.o: /usr/include/sys/cdefs.h /usr/include/sys/_symbol_aliasing.h
XYZQ.o: /usr/include/sys/_posix_availability.h /usr/include/Availability.h
XYZQ.o: /usr/include/AvailabilityInternal.h /usr/include/_types.h
XYZQ.o: /usr/include/sys/_types.h /usr/include/machine/_types.h
XYZQ.o: /usr/include/i386/_types.h /usr/include/sys/_types/_va_list.h
XYZQ.o: /usr/include/sys/_types/_size_t.h /usr/include/sys/_types/_null.h
XYZQ.o: /usr/include/sys/_types/_off_t.h /usr/include/sys/_types/_ssize_t.h
XYZQ.o: /usr/include/secure/_stdio.h /usr/include/secure/_common.h XYZQ.h
XYZQ.o: cudaXYZ.h XYZ.h
CudaLeapfrogIntegrator.o: CudaLeapfrogIntegrator.h LeapfrogIntegrator.h
CudaLeapfrogIntegrator.o: Forcefield.h cudaXYZ.h cuda_utils.h XYZ.h Force.h
CudaLeapfrogIntegrator.o: hostXYZ.h CudaPMEForcefield.h CudaForcefield.h
CudaLeapfrogIntegrator.o: XYZQ.h NeighborList.h DirectForce.h BondedForce.h
CudaLeapfrogIntegrator.o: Bonded_struct.h Grid.h Bspline.h Matrix3d.h
CudaLeapfrogIntegrator.o: CudaDomdec.h Decomp.h CudaDomdecBonded.h
CudaLeapfrogIntegrator.o: HoloConst.h gpu_utils.h /usr/include/stdio.h
CudaLeapfrogIntegrator.o: /usr/include/sys/cdefs.h
CudaLeapfrogIntegrator.o: /usr/include/sys/_symbol_aliasing.h
CudaLeapfrogIntegrator.o: /usr/include/sys/_posix_availability.h
CudaLeapfrogIntegrator.o: /usr/include/Availability.h
CudaLeapfrogIntegrator.o: /usr/include/AvailabilityInternal.h
CudaLeapfrogIntegrator.o: /usr/include/_types.h /usr/include/sys/_types.h
CudaLeapfrogIntegrator.o: /usr/include/machine/_types.h
CudaLeapfrogIntegrator.o: /usr/include/i386/_types.h
CudaLeapfrogIntegrator.o: /usr/include/sys/_types/_va_list.h
CudaLeapfrogIntegrator.o: /usr/include/sys/_types/_size_t.h
CudaLeapfrogIntegrator.o: /usr/include/sys/_types/_null.h
CudaLeapfrogIntegrator.o: /usr/include/sys/_types/_off_t.h
CudaLeapfrogIntegrator.o: /usr/include/sys/_types/_ssize_t.h
CudaLeapfrogIntegrator.o: /usr/include/secure/_stdio.h
CudaLeapfrogIntegrator.o: /usr/include/secure/_common.h
cuda_utils.o: gpu_utils.h /usr/include/stdio.h /usr/include/sys/cdefs.h
cuda_utils.o: /usr/include/sys/_symbol_aliasing.h
cuda_utils.o: /usr/include/sys/_posix_availability.h
cuda_utils.o: /usr/include/Availability.h /usr/include/AvailabilityInternal.h
cuda_utils.o: /usr/include/_types.h /usr/include/sys/_types.h
cuda_utils.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
cuda_utils.o: /usr/include/sys/_types/_va_list.h
cuda_utils.o: /usr/include/sys/_types/_size_t.h
cuda_utils.o: /usr/include/sys/_types/_null.h
cuda_utils.o: /usr/include/sys/_types/_off_t.h
cuda_utils.o: /usr/include/sys/_types/_ssize_t.h /usr/include/secure/_stdio.h
cuda_utils.o: /usr/include/secure/_common.h cuda_utils.h
CudaPMEForcefield.o: CudaPMEForcefield.h CudaForcefield.h Forcefield.h
CudaPMEForcefield.o: cudaXYZ.h cuda_utils.h XYZ.h Force.h hostXYZ.h XYZQ.h
CudaPMEForcefield.o: NeighborList.h DirectForce.h BondedForce.h
CudaPMEForcefield.o: Bonded_struct.h Grid.h Bspline.h Matrix3d.h CudaDomdec.h
CudaPMEForcefield.o: Decomp.h CudaDomdecBonded.h gpu_utils.h
CudaPMEForcefield.o: /usr/include/stdio.h /usr/include/sys/cdefs.h
CudaPMEForcefield.o: /usr/include/sys/_symbol_aliasing.h
CudaPMEForcefield.o: /usr/include/sys/_posix_availability.h
CudaPMEForcefield.o: /usr/include/Availability.h
CudaPMEForcefield.o: /usr/include/AvailabilityInternal.h
CudaPMEForcefield.o: /usr/include/_types.h /usr/include/sys/_types.h
CudaPMEForcefield.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
CudaPMEForcefield.o: /usr/include/sys/_types/_va_list.h
CudaPMEForcefield.o: /usr/include/sys/_types/_size_t.h
CudaPMEForcefield.o: /usr/include/sys/_types/_null.h
CudaPMEForcefield.o: /usr/include/sys/_types/_off_t.h
CudaPMEForcefield.o: /usr/include/sys/_types/_ssize_t.h
CudaPMEForcefield.o: /usr/include/secure/_stdio.h
CudaPMEForcefield.o: /usr/include/secure/_common.h
DirectForce.o: /usr/include/math.h /usr/include/sys/cdefs.h
DirectForce.o: /usr/include/sys/_symbol_aliasing.h
DirectForce.o: /usr/include/sys/_posix_availability.h
DirectForce.o: /usr/include/Availability.h
DirectForce.o: /usr/include/AvailabilityInternal.h gpu_utils.h
DirectForce.o: /usr/include/stdio.h /usr/include/_types.h
DirectForce.o: /usr/include/sys/_types.h /usr/include/machine/_types.h
DirectForce.o: /usr/include/i386/_types.h /usr/include/sys/_types/_va_list.h
DirectForce.o: /usr/include/sys/_types/_size_t.h
DirectForce.o: /usr/include/sys/_types/_null.h
DirectForce.o: /usr/include/sys/_types/_off_t.h
DirectForce.o: /usr/include/sys/_types/_ssize_t.h
DirectForce.o: /usr/include/secure/_stdio.h /usr/include/secure/_common.h
DirectForce.o: cuda_utils.h NeighborList.h DirectForce.h
gpu_bonded.o: cuda_utils.h gpu_utils.h /usr/include/stdio.h
gpu_bonded.o: /usr/include/sys/cdefs.h /usr/include/sys/_symbol_aliasing.h
gpu_bonded.o: /usr/include/sys/_posix_availability.h
gpu_bonded.o: /usr/include/Availability.h /usr/include/AvailabilityInternal.h
gpu_bonded.o: /usr/include/_types.h /usr/include/sys/_types.h
gpu_bonded.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
gpu_bonded.o: /usr/include/sys/_types/_va_list.h
gpu_bonded.o: /usr/include/sys/_types/_size_t.h
gpu_bonded.o: /usr/include/sys/_types/_null.h
gpu_bonded.o: /usr/include/sys/_types/_off_t.h
gpu_bonded.o: /usr/include/sys/_types/_ssize_t.h /usr/include/secure/_stdio.h
gpu_bonded.o: /usr/include/secure/_common.h XYZQ.h cudaXYZ.h XYZ.h Force.h
gpu_bonded.o: hostXYZ.h BondedForce.h Bonded_struct.h VirialPressure.h
gpu_const.o: cuda_utils.h gpu_utils.h /usr/include/stdio.h
gpu_const.o: /usr/include/sys/cdefs.h /usr/include/sys/_symbol_aliasing.h
gpu_const.o: /usr/include/sys/_posix_availability.h
gpu_const.o: /usr/include/Availability.h /usr/include/AvailabilityInternal.h
gpu_const.o: /usr/include/_types.h /usr/include/sys/_types.h
gpu_const.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
gpu_const.o: /usr/include/sys/_types/_va_list.h
gpu_const.o: /usr/include/sys/_types/_size_t.h
gpu_const.o: /usr/include/sys/_types/_null.h /usr/include/sys/_types/_off_t.h
gpu_const.o: /usr/include/sys/_types/_ssize_t.h /usr/include/secure/_stdio.h
gpu_const.o: /usr/include/secure/_common.h HoloConst.h cudaXYZ.h XYZ.h
gpu_const.o: hostXYZ.h
Force.o: gpu_utils.h /usr/include/stdio.h /usr/include/sys/cdefs.h
Force.o: /usr/include/sys/_symbol_aliasing.h
Force.o: /usr/include/sys/_posix_availability.h /usr/include/Availability.h
Force.o: /usr/include/AvailabilityInternal.h /usr/include/_types.h
Force.o: /usr/include/sys/_types.h /usr/include/machine/_types.h
Force.o: /usr/include/i386/_types.h /usr/include/sys/_types/_va_list.h
Force.o: /usr/include/sys/_types/_size_t.h /usr/include/sys/_types/_null.h
Force.o: /usr/include/sys/_types/_off_t.h /usr/include/sys/_types/_ssize_t.h
Force.o: /usr/include/secure/_stdio.h /usr/include/secure/_common.h reduce.h
Force.o: cuda_utils.h Force.h cudaXYZ.h XYZ.h hostXYZ.h
gpu_direct.o: cuda_utils.h XYZQ.h cudaXYZ.h XYZ.h Force.h hostXYZ.h
gpu_direct.o: NeighborList.h DirectForce.h VirialPressure.h
Grid.o: /usr/include/math.h /usr/include/sys/cdefs.h
Grid.o: /usr/include/sys/_symbol_aliasing.h
Grid.o: /usr/include/sys/_posix_availability.h /usr/include/Availability.h
Grid.o: /usr/include/AvailabilityInternal.h gpu_utils.h /usr/include/stdio.h
Grid.o: /usr/include/_types.h /usr/include/sys/_types.h
Grid.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
Grid.o: /usr/include/sys/_types/_va_list.h /usr/include/sys/_types/_size_t.h
Grid.o: /usr/include/sys/_types/_null.h /usr/include/sys/_types/_off_t.h
Grid.o: /usr/include/sys/_types/_ssize_t.h /usr/include/secure/_stdio.h
Grid.o: /usr/include/secure/_common.h cuda_utils.h reduce.h Matrix3d.h
Grid.o: MultiNodeMatrix3d.h Grid.h Bspline.h
gpu_dyna.o: cuda_utils.h CudaLeapfrogIntegrator.h LeapfrogIntegrator.h
gpu_dyna.o: Forcefield.h cudaXYZ.h XYZ.h Force.h hostXYZ.h
gpu_dyna.o: CudaPMEForcefield.h CudaForcefield.h XYZQ.h NeighborList.h
gpu_dyna.o: DirectForce.h BondedForce.h Bonded_struct.h Grid.h Bspline.h
gpu_dyna.o: Matrix3d.h CudaDomdec.h Decomp.h CudaDomdecBonded.h HoloConst.h
HoloConst.o: /usr/include/math.h /usr/include/sys/cdefs.h
HoloConst.o: /usr/include/sys/_symbol_aliasing.h
HoloConst.o: /usr/include/sys/_posix_availability.h
HoloConst.o: /usr/include/Availability.h /usr/include/AvailabilityInternal.h
HoloConst.o: gpu_utils.h /usr/include/stdio.h /usr/include/_types.h
HoloConst.o: /usr/include/sys/_types.h /usr/include/machine/_types.h
HoloConst.o: /usr/include/i386/_types.h /usr/include/sys/_types/_va_list.h
HoloConst.o: /usr/include/sys/_types/_size_t.h
HoloConst.o: /usr/include/sys/_types/_null.h /usr/include/sys/_types/_off_t.h
HoloConst.o: /usr/include/sys/_types/_ssize_t.h /usr/include/secure/_stdio.h
HoloConst.o: /usr/include/secure/_common.h cuda_utils.h HoloConst.h cudaXYZ.h
HoloConst.o: XYZ.h
gpu_recip.o: cuda_utils.h XYZQ.h cudaXYZ.h XYZ.h Bspline.h Grid.h Matrix3d.h
gpu_recip.o: Force.h hostXYZ.h MultiNodeMatrix3d.h
Matrix3d.o: gpu_utils.h /usr/include/stdio.h /usr/include/sys/cdefs.h
Matrix3d.o: /usr/include/sys/_symbol_aliasing.h
Matrix3d.o: /usr/include/sys/_posix_availability.h
Matrix3d.o: /usr/include/Availability.h /usr/include/AvailabilityInternal.h
Matrix3d.o: /usr/include/_types.h /usr/include/sys/_types.h
Matrix3d.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
Matrix3d.o: /usr/include/sys/_types/_va_list.h
Matrix3d.o: /usr/include/sys/_types/_size_t.h /usr/include/sys/_types/_null.h
Matrix3d.o: /usr/include/sys/_types/_off_t.h
Matrix3d.o: /usr/include/sys/_types/_ssize_t.h /usr/include/secure/_stdio.h
Matrix3d.o: /usr/include/secure/_common.h cuda_utils.h Matrix3d.h
MultiNodeMatrix3d.o: /usr/include/stdlib.h /usr/include/Availability.h
MultiNodeMatrix3d.o: /usr/include/AvailabilityInternal.h
MultiNodeMatrix3d.o: /usr/include/_types.h /usr/include/sys/_types.h
MultiNodeMatrix3d.o: /usr/include/sys/cdefs.h
MultiNodeMatrix3d.o: /usr/include/sys/_symbol_aliasing.h
MultiNodeMatrix3d.o: /usr/include/sys/_posix_availability.h
MultiNodeMatrix3d.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
MultiNodeMatrix3d.o: /usr/include/sys/wait.h /usr/include/sys/_types/_pid_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_id_t.h
MultiNodeMatrix3d.o: /usr/include/sys/signal.h
MultiNodeMatrix3d.o: /usr/include/sys/appleapiopts.h
MultiNodeMatrix3d.o: /usr/include/machine/signal.h /usr/include/i386/signal.h
MultiNodeMatrix3d.o: /usr/include/machine/_mcontext.h
MultiNodeMatrix3d.o: /usr/include/i386/_mcontext.h
MultiNodeMatrix3d.o: /usr/include/mach/i386/_structs.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_sigaltstack.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_ucontext.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_pthread_attr_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_sigset_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_size_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_uid_t.h
MultiNodeMatrix3d.o: /usr/include/sys/resource.h /usr/include/stdint.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_int8_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_int16_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_int32_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_int64_t.h
MultiNodeMatrix3d.o: /usr/include/_types/_uint8_t.h
MultiNodeMatrix3d.o: /usr/include/_types/_uint16_t.h
MultiNodeMatrix3d.o: /usr/include/_types/_uint32_t.h
MultiNodeMatrix3d.o: /usr/include/_types/_uint64_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_intptr_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_uintptr_t.h
MultiNodeMatrix3d.o: /usr/include/_types/_intmax_t.h
MultiNodeMatrix3d.o: /usr/include/_types/_uintmax_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_timeval.h
MultiNodeMatrix3d.o: /usr/include/machine/endian.h /usr/include/i386/endian.h
MultiNodeMatrix3d.o: /usr/include/sys/_endian.h
MultiNodeMatrix3d.o: /usr/include/libkern/_OSByteOrder.h
MultiNodeMatrix3d.o: /usr/include/libkern/i386/_OSByteOrder.h
MultiNodeMatrix3d.o: /usr/include/alloca.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_ct_rune_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_rune_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_wchar_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_null.h
MultiNodeMatrix3d.o: /usr/include/machine/types.h /usr/include/i386/types.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/___offsetof.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_dev_t.h
MultiNodeMatrix3d.o: /usr/include/sys/_types/_mode_t.h mpi_utils.h
MultiNodeMatrix3d.o: cuda_utils.h MultiNodeMatrix3d.h Matrix3d.h
CudaDomdecBonded.o: cuda_utils.h gpu_utils.h /usr/include/stdio.h
CudaDomdecBonded.o: /usr/include/sys/cdefs.h
CudaDomdecBonded.o: /usr/include/sys/_symbol_aliasing.h
CudaDomdecBonded.o: /usr/include/sys/_posix_availability.h
CudaDomdecBonded.o: /usr/include/Availability.h
CudaDomdecBonded.o: /usr/include/AvailabilityInternal.h /usr/include/_types.h
CudaDomdecBonded.o: /usr/include/sys/_types.h /usr/include/machine/_types.h
CudaDomdecBonded.o: /usr/include/i386/_types.h
CudaDomdecBonded.o: /usr/include/sys/_types/_va_list.h
CudaDomdecBonded.o: /usr/include/sys/_types/_size_t.h
CudaDomdecBonded.o: /usr/include/sys/_types/_null.h
CudaDomdecBonded.o: /usr/include/sys/_types/_off_t.h
CudaDomdecBonded.o: /usr/include/sys/_types/_ssize_t.h
CudaDomdecBonded.o: /usr/include/secure/_stdio.h
CudaDomdecBonded.o: /usr/include/secure/_common.h CudaDomdecBonded.h
CudaDomdecBonded.o: Bonded_struct.h CudaDomdec.h Decomp.h cudaXYZ.h XYZ.h
CudaDomdecBonded.o: Force.h hostXYZ.h
