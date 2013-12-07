#include <iostream>
#include <math.h>
#include <cuda.h>
#include "gpu_utils.h"
#include "Bspline.h"

template <typename T>
__global__ void fill_bspline_4(const float4 *xyzq, const int ncoord, const float *recip,
			       const int nfftx, const int nffty, const int nfftz,
			       gridp_t *gridp, float3 *theta, float3 *dtheta) {

  // Position to xyzq and atomgrid
  unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;

  while (pos < ncoord) {
    float4 xyzqi = xyzq[pos];
    float x = xyzqi.x;
    float y = xyzqi.y;
    float z = xyzqi.z;
    float q = xyzqi.w;

    float w;
    // NOTE: I don't think we need the +2.0f here..
    w = x*recip[0] + y*recip[1] + z*recip[2] + 2.0f;
    float frx = (float)(nfftx*(w - (floorf(w + 0.5f) - 0.5f)));

    w = x*recip[3] + y*recip[4] + z*recip[5] + 2.0f;
    float fry = (float)(nffty*(w - (floorf(w + 0.5f) - 0.5f)));

    w = x*recip[6] + y*recip[7] + z*recip[8] + 2.0f;
    float frz = (float)(nfftz*(w - (floorf(w + 0.5f) - 0.5f)));

    int frxi = (int)(frx);
    int fryi = (int)(fry);
    int frzi = (int)(frz);

    float wx = frx - (float)frxi;
    float wy = fry - (float)fryi;
    float wz = frz - (float)frzi;

    gridp[pos].x = frxi;
    gridp[pos].y = fryi;
    gridp[pos].z = frzi;
    gridp[pos].q = q;

    float3 theta_tmp[4];
    float3 dtheta_tmp[4];

    theta_tmp[3].x = 0.0f;
    theta_tmp[3].y = 0.0f;
    theta_tmp[3].z = 0.0f;
    theta_tmp[1].x = wx;
    theta_tmp[1].y = wy;
    theta_tmp[1].z = wz;
    theta_tmp[0].x = 1.0f - wx;
    theta_tmp[0].y = 1.0f - wy;
    theta_tmp[0].z = 1.0f - wz;

    // compute standard b-spline recursion
    theta_tmp[2].x = 0.5f*wx*theta_tmp[1].x;
    theta_tmp[2].y = 0.5f*wy*theta_tmp[1].y;
    theta_tmp[2].z = 0.5f*wz*theta_tmp[1].z;
       
    theta_tmp[1].x = 0.5f*((wx+1.0f)*theta_tmp[0].x + (2.0f-wx)*theta_tmp[1].x);
    theta_tmp[1].y = 0.5f*((wy+1.0f)*theta_tmp[0].y + (2.0f-wy)*theta_tmp[1].y);
    theta_tmp[1].z = 0.5f*((wz+1.0f)*theta_tmp[0].z + (2.0f-wz)*theta_tmp[1].z);
       
    theta_tmp[0].x = 0.5f*(1.0f-wx)*theta_tmp[0].x;
    theta_tmp[0].y = 0.5f*(1.0f-wy)*theta_tmp[0].y;
    theta_tmp[0].z = 0.5f*(1.0f-wz)*theta_tmp[0].z;
       
    // perform standard b-spline differentiationa
    dtheta_tmp[0].x = -theta_tmp[0].x;
    dtheta_tmp[0].y = -theta_tmp[0].y;
    dtheta_tmp[0].z = -theta_tmp[0].z;

    dtheta_tmp[1].x = theta_tmp[0].x - theta_tmp[1].x;
    dtheta_tmp[1].y = theta_tmp[0].y - theta_tmp[1].y;
    dtheta_tmp[1].z = theta_tmp[0].z - theta_tmp[1].z;

    dtheta_tmp[2].x = theta_tmp[1].x - theta_tmp[2].x;
    dtheta_tmp[2].y = theta_tmp[1].y - theta_tmp[2].y;
    dtheta_tmp[2].z = theta_tmp[1].z - theta_tmp[2].z;

    dtheta_tmp[3].x = theta_tmp[2].x - theta_tmp[3].x;
    dtheta_tmp[3].y = theta_tmp[2].y - theta_tmp[3].y;
    dtheta_tmp[3].z = theta_tmp[2].z - theta_tmp[3].z;
          
    // one more recursion
    theta_tmp[3].x = (1.0f/3.0f)*wx*theta_tmp[2].x;
    theta_tmp[3].y = (1.0f/3.0f)*wy*theta_tmp[2].y;
    theta_tmp[3].z = (1.0f/3.0f)*wz*theta_tmp[2].z;

    theta_tmp[2].x = (1.0f/3.0f)*((wx+1.0f)*theta_tmp[1].x + (3.0f-wx)*theta_tmp[2].x);
    theta_tmp[2].y = (1.0f/3.0f)*((wy+1.0f)*theta_tmp[1].y + (3.0f-wy)*theta_tmp[2].y);
    theta_tmp[2].z = (1.0f/3.0f)*((wz+1.0f)*theta_tmp[1].z + (3.0f-wz)*theta_tmp[2].z);

    theta_tmp[1].x = (1.0f/3.0f)*((wx+2.0f)*theta_tmp[0].x + (2.0f-wx)*theta_tmp[1].x);
    theta_tmp[1].y = (1.0f/3.0f)*((wy+2.0f)*theta_tmp[0].y + (2.0f-wy)*theta_tmp[1].y);
    theta_tmp[1].z = (1.0f/3.0f)*((wz+2.0f)*theta_tmp[0].z + (2.0f-wz)*theta_tmp[1].z);
       
    theta_tmp[0].x = (1.0f/3.0f)*(1.0f-wx)*theta_tmp[0].x;
    theta_tmp[0].y = (1.0f/3.0f)*(1.0f-wy)*theta_tmp[0].y;
    theta_tmp[0].z = (1.0f/3.0f)*(1.0f-wz)*theta_tmp[0].z;

    // Store theta_tmp and dtheta_tmp into global memory
    int pos4 = pos*4;
    theta[pos4]   = theta_tmp[0];
    theta[pos4+1] = theta_tmp[1];
    theta[pos4+2] = theta_tmp[2];
    theta[pos4+3] = theta_tmp[3];

    dtheta[pos4]   = dtheta_tmp[0];
    dtheta[pos4+1] = dtheta_tmp[1];
    dtheta[pos4+2] = dtheta_tmp[2];
    dtheta[pos4+3] = dtheta_tmp[3];

    pos += blockDim.x*gridDim.x;
  }

}

//
// Bspline class method definitions
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//

template <typename T>
void Bspline<T>::set_ncoord(const int ncoord) {
  reallocate<T>(&theta, &theta_len, 3*ncoord*order, 1.2f);
  reallocate<T>(&dtheta, &dtheta_len, 3*ncoord*order, 1.2f);
  reallocate<gridp_t>(&gridp, &gridp_len, ncoord, 1.2f);  
}

template <typename T>
Bspline<T>::Bspline(const int ncoord, const int order, const int nfftx, const int nffty, const int nfftz) :
  theta(NULL), dtheta(NULL), gridp(NULL), order(order), nfftx(nfftx), nffty(nffty), nfftz(nfftz) {

  set_ncoord(ncoord);

  allocate<T>(&recip, 9);
  allocate<T>(&prefac_x, nfftx);
  allocate<T>(&prefac_y, nffty);
  allocate<T>(&prefac_z, nfftz);

}
  
template <typename T>
Bspline<T>::~Bspline() {
  deallocate<T>(&theta);
  deallocate<T>(&dtheta);
  deallocate<gridp_t>(&gridp);
  deallocate<T>(&recip);
  deallocate<T>(&prefac_x);
  deallocate<T>(&prefac_y);
  deallocate<T>(&prefac_z);
}

template <typename T>
template <typename B>
void Bspline<T>::set_recip(const B *h_recip) {
  T h_recip_T[9];
  for (int i=0;i < 9;i++) h_recip_T[i] = (T)h_recip[i];
  copy_HtoD<T>(h_recip_T, recip, 9);
}

template <typename T>
void Bspline<T>::fill_bspline(const float4 *xyzq, const int ncoord) {

  // Re-allocates (theta, dtheta, gridp) if needed
  set_ncoord(ncoord);

  int nthread = 64;
  int nblock = (ncoord-1)/nthread + 1;

  std::cout << "nblock=" << nblock << std::endl;
  std::cout << nfftx << " "<< nffty << " "<< nfftz << std::endl;

  //bool ortho = (recip[1] == recip[2] == recip[3] == recip[5] == recip[6] == recip[7] == 0.0f);
  
  switch(order) {
  case 4:
    fill_bspline_4<T> <<< nblock, nthread >>>(xyzq, ncoord, recip, 
					      nfftx, nffty, nfftz, gridp, 
					      (float3 *)theta, (float3 *)dtheta);
    break;
  default:
    exit(1);
  }
  
  cudaCheck(cudaGetLastError());
}

template <typename T>
void Bspline<T>::dftmod(double *bsp_mod, const double *bsp_arr, const int nfft) {

  const double rsmall = 1.0e-10;
  double nfftr = (2.0*3.14159265358979323846)/(double)nfft;

  for (int k=1;k <= nfft;k++) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    double arg1 = (k-1)*nfftr;
    for (int j=1;j < nfft;j++) {
      double arg = arg1*(j-1);
      sum1 += bsp_arr[j-1]*cos(arg);
      sum2 += bsp_arr[j-1]*sin(arg);
    }
    bsp_mod[k-1] = sum1*sum1 + sum2*sum2;
  }

  for (int k=1;k <= nfft;k++)
    if (bsp_mod[k-1] < rsmall)
      bsp_mod[k-1] = 0.5*(bsp_mod[k-1-1] + bsp_mod[k+1-1]);

  for (int k=1;k <= nfft;k++)
    bsp_mod[k-1] = 1.0/bsp_mod[k-1];

}

template <typename T>
void Bspline<T>::fill_bspline_host(const double w, double *array, double *darray) {

  //--- do linear case
  array[order-1] = 0.0;
  array[2-1] = w;
  array[1-1] = 1.0 - w;

  //--- compute standard b-spline recursion
  for (int k=3;k <= order-1;k++) {
    double div = 1.0 / (double)(k-1);
    array[k-1] = div*w*array[k-1-1];
    for (int j=1;j <= k-2;j++)
      array[k-j-1] = div*((w+j)*array[k-j-1-1] + (k-j-w)*array[k-j-1]);
    array[1-1] = div*(1.0-w)*array[1-1];
  }

  //--- perform standard b-spline differentiation
  darray[1-1] = -array[1-1];
  for (int j=2;j <= order;j++)
    darray[j-1] = array[j-1-1] - array[j-1];

  //--- one more recursion
  int k = order;
  double div = 1.0 / (double)(k-1);
  array[k-1] = div*w*array[k-1-1];
  for (int j=1;j <= k-2;j++)
    array[k-j-1] = div*((w+j)*array[k-j-1-1] + (k-j-w)*array[k-j-1]);

  array[1-1] = div*(1.0-w)*array[1-1];

}

//
// Calculates (prefac_x, prefac_y, prefac_z)
// NOTE: This calculation is done on the CPU since it is only done infrequently
//
template <typename T>
void Bspline<T>::calc_prefac() {
  
  int max_nfft = max(nfftx, max(nffty, nfftz));
  double *bsp_arr = new double[max_nfft];
  double *bsp_mod = new double[max_nfft];
  double *array = new double[order];
  double *darray = new double[order];
  T *h_prefac_x = new T[nfftx];
  T *h_prefac_y = new T[nffty];
  T *h_prefac_z = new T[nfftz];

  fill_bspline_host(0.0, array, darray);
  for (int i=0;i < max_nfft;i++) bsp_arr[i] = 0.0;
  for (int i=2;i <= order+1;i++) bsp_arr[i-1] = array[i-1-1];

  dftmod(bsp_mod, bsp_arr, nfftx);
  for (int i=0;i < nfftx;i++) h_prefac_x[i] = (T)bsp_mod[i];

  dftmod(bsp_mod, bsp_arr, nffty);
  for (int i=0;i < nffty;i++) h_prefac_y[i] = (T)bsp_mod[i];

  dftmod(bsp_mod, bsp_arr, nfftz);
  for (int i=0;i < nfftz;i++) h_prefac_z[i] = (T)bsp_mod[i];

  //  std::cout<< "h_prefac_x = "<<std::endl;
  //  for (int i=0;i < 10;i++) std::cout<<h_prefac_x[i]<<std::endl;

  copy_HtoD<T>(h_prefac_x, prefac_x, nfftx);
  copy_HtoD<T>(h_prefac_y, prefac_y, nfftx);
  copy_HtoD<T>(h_prefac_z, prefac_z, nfftx);

  delete [] bsp_arr;
  delete [] bsp_mod;
  delete [] array;
  delete [] darray;
  delete [] h_prefac_x;
  delete [] h_prefac_y;
  delete [] h_prefac_z;
}

//
// Explicit instances of Bspline
//
template class Bspline<float>;
template void Bspline<float>::set_recip<double>(const double *h_recip);
