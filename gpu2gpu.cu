#include <iostream>
#include <cuda.h>
#include "gpu_utils.h"
#include "cuda_utils.h"


<template T>
void gpu2gpu::sendrecv(T *sendbuf, int sendcount, int dst, int sendtag,
		       T *recvbuf, int recvcount, int src, int recvtag) {

  cudaCheck();

}
