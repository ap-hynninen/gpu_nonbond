#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <cufft.h>
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "fbfft/FBFFT.h"
#include "fbfft/FBFFTCommon.cuh"

#define fbfftCheck(stmt) do {           \
    facebook::cuda::fbfft::FBFFTParameters::ErrorCode err = stmt;           \
    if (err != facebook::cuda::fbfft::FBFFTParameters::Success) {           \
      printf("Error running %s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
      if (err == facebook::cuda::fbfft::FBFFTParameters::UnsupportedSize) \
        printf("Error code: UnsupportedSize\n"); \
      if (err == facebook::cuda::fbfft::FBFFTParameters::UnsupportedDimension) \
        printf("Error code: UnsupportedDimension\n"); \
      exit(1);                \
    }                 \
  } while(0)

//
// Loads vector from file
//
template <typename T>
void load_vec(const int nind, const char *filename, const int n, T *ind) {
  std::ifstream file(filename);
  if (file.is_open()) {

    for (int i=0;i < n;i++) {
      for (int k=0;k < nind;k++) {
	if (!(file >> ind[i*nind+k])) {
	  std::cerr<<"Error reading file "<<filename<<std::endl;
	  exit(1);
	}
      }
    }

  } else {
    std::cerr<<"Error opening file "<<filename<<std::endl;
    exit(1);
  }

}

void test_1dfbfft();
void test_2dfbfft();

int main(int argc, char *argv[]) {

  int numnode = 1;
  int mynode = 0;

  std::vector<int> devices;
  start_gpu(numnode, mynode, devices);

	test_1dfbfft();
	test_2dfbfft();

	stop_gpu();

	return 0;
}

void check_err1D(int nfft, int nbatch, float* h_dataOut, float* h_dataOutRef) {
  double max_err = 0.0;
  for (int j=0;j < nbatch;++j) {
  	int start = j*(nfft/2+1)*2;
  	int startRef = j*nfft*2;
	  for (int i=0;i < nfft/2+1;++i) {
	  	double err1 = fabs(h_dataOut[start+2*i]-h_dataOutRef[startRef+2*i]);
	  	double err2 = fabs(h_dataOut[start+2*i+1]-h_dataOutRef[startRef+2*i+1]);
	  	max_err = max(max_err, err1);
	  	max_err = max(max_err, err2);
	  	if (max_err > 1.0e-5) {
	  		printf("Maximum error exceeded batch=%d i=%d\n",j,i);
	  		printf("Res: %f %f\n",h_dataOut[start+2*i],h_dataOut[start+2*i+1]);
	  		printf("Ref: %f %f\n",h_dataOutRef[startRef+2*i],h_dataOutRef[startRef+2*i+1]);
	  		break;
	  	}
	  }
	}
  printf("max_err=%e\n",max_err);
}

void check_err2DfbfftSMALL(int nfftx, int nffty, int nbatch, float* h_dataOut, float* h_dataOutRef) {
  double max_err = 0.0;
  int stride = nfftx*2;
  int strideRef = nfftx*2;
  for (int k=0;k < nbatch;++k) {
  	int start = k*stride*nffty;
  	int startRef = k*strideRef*nffty;
  	for (int j=0;j < nffty/2+1;++j) {
		  for (int i=0;i < nfftx;++i) {
		  	int pos = i*2 + j*stride;
		  	int posRef = i*2 + j*strideRef;
		  	double err1 = fabs(h_dataOut[start+pos]-h_dataOutRef[startRef+posRef]);
		  	double err2 = fabs(h_dataOut[start+pos+1]-h_dataOutRef[startRef+posRef+1]);
		  	max_err = max(max_err, err1);
		  	max_err = max(max_err, err2);
		  	if (max_err > 1.0e-4) {
		  		printf("Maximum error exceeded batch=%d i=%d j=%d | %d %d | %d %d\n",k,i,j,start,pos,startRef,posRef);
		  		printf("Res: %f %f\n",h_dataOut[start+pos],h_dataOut[start+pos+1]);
		  		printf("Ref: %f %f\n",h_dataOutRef[startRef+posRef],h_dataOutRef[startRef+posRef+1]);
		  		return;
		  	}
		  }
	  }
	}
  printf("max_err=%e\n",max_err);
}

void check_err2DfbfftBIG(int nfftx, int nffty, int nbatch, float* h_dataOut, float* h_dataOutRef) {
  double max_err = 0.0;
  int stride = nfftx*2;
  int strideRef = nfftx*2;
  for (int k=0;k < nbatch;++k) {
  	int start = k*stride*nffty;
  	int startRef = k*strideRef*nffty;
  	for (int j=0;j < nffty/2+1;++j) {
		  for (int i=0;i < nfftx;++i) {
		  	int pos = i*2 + j*stride;
		  	int posRef = i*2 + j*strideRef;
		  	double err1 = fabs(h_dataOut[start+pos]-h_dataOutRef[startRef+posRef]);
		  	double err2 = fabs(h_dataOut[start+pos+1]-h_dataOutRef[startRef+posRef+1]);
		  	max_err = max(max_err, err1);
		  	max_err = max(max_err, err2);
		  	if (max_err > 1.0e-4) {
		  		printf("Maximum error exceeded batch=%d i=%d j=%d | %d %d | %d %d\n",k,i,j,start,pos,startRef,posRef);
		  		printf("Res: %f %f\n",h_dataOut[start+pos],h_dataOut[start+pos+1]);
		  		printf("Ref: %f %f\n",h_dataOutRef[startRef+posRef],h_dataOutRef[startRef+posRef+1]);
		  		return;
		  	}
		  }
	  }
	}
  printf("max_err=%e\n",max_err);
}

void check_err2Dcufft(int nfftx, int nffty, int nbatch, float* h_dataOut, float* h_dataOutRef) {
  double max_err = 0.0;
  int stride = (nfftx/2+1)*2;
  int strideRef = nfftx*2;
  for (int k=0;k < nbatch;++k) {
  	int start = k*stride*nffty;
  	int startRef = k*strideRef*nffty;
  	for (int j=0;j < nffty;++j) {
		  for (int i=0;i < nfftx/2+1;++i) {
		  	int pos = i*2 + j*stride;
		  	int posRef = i*2 + j*strideRef;
		  	double err1 = fabs(h_dataOut[start+pos]-h_dataOutRef[startRef+posRef]);
		  	double err2 = fabs(h_dataOut[start+pos+1]-h_dataOutRef[startRef+posRef+1]);
		  	max_err = max(max_err, err1);
		  	max_err = max(max_err, err2);
		  	if (max_err > 1.0e-4) {
		  		printf("Maximum error exceeded batch=%d i=%d j=%d | %d %d | %d %d\n",k,i,j,start,pos,startRef,posRef);
		  		printf("Res: %f %f\n",h_dataOut[start+pos],h_dataOut[start+pos+1]);
		  		printf("Ref: %f %f\n",h_dataOutRef[startRef+posRef],h_dataOutRef[startRef+posRef+1]);
		  		return;
		  	}
		  }
	  }
	}
  printf("max_err=%e\n",max_err);
}

void transpose_xy(int nfftx, int nffty, int nbatch, float* h_dataOut) {
	float *tmp = new float[nfftx*nffty*2];
  for (int k=0;k < nbatch;++k) {
  	int start = k*nfftx*nffty*2;
  	for (int j=0;j < nffty;++j) {
		  for (int i=0;i < nfftx;++i) {
		  	int ii = i + j*nfftx;
		  	int jj = j + i*nfftx;
		  	tmp[2*jj]   = h_dataOut[start+2*ii];
		  	tmp[2*jj+1] = h_dataOut[start+2*ii+1];
			}
		}
		memcpy(h_dataOut+start, tmp, nfftx*nffty*2*sizeof(float));
	}
	delete [] tmp;
}

//
// Test Facebook 1D FFT
//
void test_1dfbfft() {
	int nfft = 64;
	int nbatch = 2;

	float *h_dataIn = new float[nfft*nbatch];
	float *h_dataOut = new float[(nfft/2+1)*2*nbatch];
	float *h_dataOutRef = new float[nfft*2*nbatch];

	float *d_dataIn = NULL;
	//allocate<float>(&d_dataIn, nfft*nbatch);
	allocate<float>(&d_dataIn, (nfft/2+1)*2*nbatch);
	float *d_dataOut = NULL;
	allocate<float>(&d_dataOut, (nfft/2+1)*2*nbatch);

	if (nbatch == 1) {
		load_vec<float>(1, "test_data/dataFFTin64.txt", nfft*nbatch, h_dataIn);
		load_vec<float>(2, "test_data/dataFFTout64.txt", nfft*nbatch, h_dataOutRef);
	} else if (nbatch == 2) {
		load_vec<float>(1, "test_data/dataFFTin64x2.txt", nfft*nbatch, h_dataIn);
		load_vec<float>(2, "test_data/dataFFTout64x2.txt", nfft*nbatch, h_dataOutRef);
	} else {
		std::cerr << "Only nbatch=1 or 2 are supported" << std::endl;
		exit(1);
	}

	cufftHandle x_r2c_plan;
  cufftCheck(cufftPlanMany(&x_r2c_plan, 1, &nfft,
  	NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, nbatch));
  cufftCheck(cufftSetCompatibilityMode(x_r2c_plan, CUFFT_COMPATIBILITY_NATIVE));

  copy_HtoD_sync<float>(h_dataIn, d_dataIn, nfft*nbatch);
  cufftCheck(cufftExecR2C(x_r2c_plan,
  	(cufftReal *)d_dataIn, (cufftComplex *)d_dataOut));
  cudaCheck(cudaDeviceSynchronize());
  copy_DtoH_sync<float>(d_dataOut, h_dataOut, (nfft/2+1)*2*nbatch);

  check_err1D(nfft, nbatch, h_dataOut, h_dataOutRef);

  cufftCheck(cufftDestroy(x_r2c_plan));

  clear_gpu_array_sync<float>(d_dataOut, (nfft/2+1)*2*nbatch);

  {
  	using namespace facebook::cuda::fbfft;
  	int dataInSize[2] = {nbatch, nfft};
  	int dataOutSize[3] = {nbatch, nfft/2+1, 2};
  	DeviceTensor<float, 2> dataInTensor(d_dataIn, dataInSize);
  	//DeviceTensor<float, 3> dataOutTensor(d_dataOut, dataOutSize);
  	DeviceTensor<float, 3> dataOutTensor(d_dataIn, dataOutSize);
  	fbfftCheck(fbfft1D<1>(dataInTensor, dataOutTensor));
  	cudaCheck(cudaDeviceSynchronize());
	  //copy_DtoH_sync<float>(d_dataOut, h_dataOut, (nfft/2+1)*2*nbatch);
	  copy_DtoH_sync<float>(d_dataIn, h_dataOut, (nfft/2+1)*2*nbatch);
  	check_err1D(nfft, nbatch, h_dataOut, h_dataOutRef);
  	//for (int i=0;i < nfft/2+1;i++) {
	  //	printf("%f %f\n",h_dataOut[2*i],h_dataOut[2*i+1]);	
  	//}
	}

  deallocate<float>(&d_dataIn);
  deallocate<float>(&d_dataOut);
  delete [] h_dataIn;
  delete [] h_dataOut;
	delete [] h_dataOutRef;
}

//
// Test Facebook 2D FFT
//
void test_2dfbfft() {
	int nfftx = 64;
	int nffty = 64;
	int nbatch = 1;

	float *h_dataIn = new float[nfftx*nffty*nbatch];
	float *h_dataOut = new float[nfftx*(nffty/2+1)*2*nbatch];
	float *h_dataOutRef = new float[nfftx*nffty*2*nbatch];
	float *h_dataOutRefTransp = new float[nfftx*nffty*2*nbatch];

	float *d_dataIn = NULL;
	allocate<float>(&d_dataIn, nfftx*nffty*nbatch);
	float *d_dataOut = NULL;
	allocate<float>(&d_dataOut, (nfftx/2+1)*2*nffty*nbatch);

	if (nbatch == 1) {
		if (nfftx == 64 && nffty == 64) {
			load_vec<float>(1, "test_data/dataFFTin64x64.txt", nfftx*nffty*nbatch, h_dataIn);
			load_vec<float>(2, "test_data/dataFFTout64x64.txt", nfftx*nffty*nbatch, h_dataOutRef);
		} else if (nfftx == 4 && nffty == 4) {
			load_vec<float>(1, "test_data/dataFFTin4x4.txt", nfftx*nffty*nbatch, h_dataIn);
			load_vec<float>(2, "test_data/dataFFTout4x4.txt", nfftx*nffty*nbatch, h_dataOutRef);
		} else {
			std::cerr << "FFT size not supported" << std::endl;
			exit(1);
		}
	} else {
		std::cerr << "Only nbatch=1 is supported" << std::endl;
		exit(1);
	}

	memcpy(h_dataOutRefTransp, h_dataOutRef, nfftx*nffty*2*nbatch*sizeof(float));
	transpose_xy(nfftx, nffty, nbatch, h_dataOutRefTransp);

  int n[2] = {nffty, nfftx};
	cufftHandle xy_r2c_plan;
  cufftCheck(cufftPlanMany(&xy_r2c_plan, 2, n,
  	NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, nbatch));
  cufftCheck(cufftSetCompatibilityMode(xy_r2c_plan, CUFFT_COMPATIBILITY_NATIVE));

  copy_HtoD_sync<float>(h_dataIn, d_dataIn, nfftx*nffty*nbatch);
  cufftCheck(cufftExecR2C(xy_r2c_plan,
  	(cufftReal *)d_dataIn, (cufftComplex *)d_dataOut));
  cudaCheck(cudaDeviceSynchronize());
  copy_DtoH_sync<float>(d_dataOut, h_dataOut, nfftx*(nffty/2+1)*2*nbatch);

  check_err2Dcufft(nfftx, nffty, nbatch, h_dataOut, h_dataOutRef);

  cufftCheck(cufftDestroy(xy_r2c_plan));

  clear_gpu_array_sync<float>(d_dataOut, (nfftx/2+1)*2*nffty*nbatch);

/*
  printf("-------------------------------\n");
  printf("h_dataOut (cufft)\n");
  printf("-------------------------------\n");
  int pos = 0;
  for (int j=0;j < nffty;++j) {
  	for (int i=0;i < nfftx/2+1;i++,pos+=2) {
			printf("%d %d %f %f\n",i,j,h_dataOut[pos],h_dataOut[pos+1]);
		}
  }
  printf("-------------------------------\n");
  printf("h_dataOutRef\n");
  printf("-------------------------------\n");
  pos = 0;
  for (int j=0;j < nffty;++j) {
  	for (int i=0;i < nfftx;i++,pos+=2) {
			printf("%d %d %f %f\n",i,j,h_dataOutRef[pos],h_dataOutRef[pos+1]);
		}
  }
  printf("-------------------------------\n");
  printf("h_dataOutRefTransp\n");
  printf("-------------------------------\n");
  pos = 0;
  for (int j=0;j < nffty/2+1;++j) {
  	for (int i=0;i < nfftx;i++,pos+=2) {
			printf("%d %d %f %f\n",i,j,h_dataOutRefTransp[pos],h_dataOutRefTransp[pos+1]);
		}
  }
*/

  /*
  int pos = 0;
  for (int j=0;j < nffty;++j) {
  	for (int i=0;i < nfftx/2+1;i++,pos+=2) {
			if (j == 1) printf("%f %f\n",h_dataOut[pos],h_dataOut[pos+1]);
		}
  }
*/

  {
  	using namespace facebook::cuda::fbfft;
  	int dataInSize[3] = {nbatch, nffty, nfftx};
  	int dataOutSize[4] = {nbatch, 0, 0, 2};
		if (nfftx == 64 && nffty == 64) {
  		dataOutSize[1] = nfftx;
  		dataOutSize[2] = nffty/2+1;
  	} else {
  		dataOutSize[1] = nfftx/2+1;
  		dataOutSize[2] = nffty;
  	}
  	DeviceTensor<float, 3> dataInTensor(d_dataIn, dataInSize);
  	DeviceTensor<float, 4> dataOutTensor(d_dataOut, dataOutSize);
  	fbfftCheck(fbfft2D<1>(dataInTensor, dataOutTensor));

  	//int dataInSize[3] = {nbatch, nffty, nfftx};
  	//int dataOutSize[3] = {nbatch, nffty, nfftx/2+1};
  	//DeviceTensor<Complex, 3> dataInTensor((Complex *)d_dataIn, dataInSize);
  	//DeviceTensor<Complex, 3> dataOutTensor((Complex *)d_dataOut, dataOutSize);
  	//fbfftCheck(fbfft2D<1>(dataInTensor, dataOutTensor));

  	cudaCheck(cudaDeviceSynchronize());
	  copy_DtoH_sync<float>(d_dataOut, h_dataOut, (nfftx/2+1)*2*nffty*nbatch);
		if (nfftx == 64 && nffty == 64) {
  		check_err2DfbfftBIG(nfftx, nffty, nbatch, h_dataOut, h_dataOutRefTransp);
  	} else {
  		check_err2DfbfftSMALL(nfftx, nffty, nbatch, h_dataOut, h_dataOutRefTransp);
  	}

/*
		if (nfftx == 64 && nffty == 64) {
			float *h_dataInCmplx = new float[nfftx*nffty*2*nbatch];
			float *d_dataInCmplx = NULL;
			allocate<float>(&d_dataInCmplx, nfftx*nffty*2*nbatch);
 	 		int dataInCmplxSize[3] = {nbatch, nffty, nfftx/2+1};
  		int dataOutCmplxSize[3] = {nbatch, nffty, nfftx/2+1};
  		DeviceTensor<Complex, 3> dataInCmplxTensor((Complex *)d_dataInCmplx, dataInCmplxSize);
  		DeviceTensor<Complex, 3> dataOutCmplxTensor((Complex *)d_dataOut, dataOutCmplxSize);
  		fbfftCheck(fbfft2D<1>(dataInCmplxTensor, dataOutCmplxTensor));
  		cudaCheck(cudaDeviceSynchronize());
  		delete [] h_dataInCmplx;
  		deallocate<float>(&d_dataInCmplx);
  	}
*/
  	//for (int i=0;i < nfft/2+1;i++) {
	  //	printf("%f %f\n",h_dataOut[2*i],h_dataOut[2*i+1]);	
  	//}
	}

/*
  printf("-------------------------------\n");
  printf("h_dataOut (fbfft)\n");
  printf("-------------------------------\n");
  pos = 0;
  for (int j=0;j < nffty/2+1;++j) {
  	for (int i=0;i < nfftx;i++,pos+=2) {
			printf("%d %d %f %f\n",i,j,h_dataOut[pos],h_dataOut[pos+1]);
		}
  }
*/

  deallocate<float>(&d_dataIn);
  deallocate<float>(&d_dataOut);
  delete [] h_dataIn;
  delete [] h_dataOut;
	delete [] h_dataOutRef;
	delete [] h_dataOutRefTransp;
}
