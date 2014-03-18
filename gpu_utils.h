
#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <stdio.h>

static const int warpsize = 32;
//static __constant__ const int warpsize = 32;

static __constant__ const float FORCE_SCALE = (float)(1ll << 40);
static __constant__ const double INV_FORCE_SCALE = (double)1.0/(double)(1ll << 40);

static __constant__ const float FORCE_SCALE_I = (float)(1 << 31);
static __constant__ const double INV_FORCE_SCALE_I = (double)1.0/(double)(1 << 31);

#define cudaCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
	  printf("Error running %s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
	  printf("Error string: %s\n",cudaGetErrorString(err)); \
	  exit(1);							\
        }                                                  \
    } while(0)

#define cufftCheck(stmt) do {						\
    cufftResult err = stmt;						\
    if (err != CUFFT_SUCCESS) {						\
      printf("Error running %s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
      exit(1);								\
    }									\
  } while(0)

//
// Double precision atomicAdd, copied from CUDA_C_Programming_Guide.pdf (ver 5.0)
//
static __device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old; old = atomicCAS(address_as_ull, assumed,
				   __double_as_longlong(val +
							__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

//----------------------------------------------------------------------------------------

static __forceinline__ __device__ float __internal_fmad(float a, float b, float c)
{
#if __CUDA_ARCH__ >= 200
  return __fmaf_rn (a, b, c);
#else // __CUDA_ARCH__ >= 200
  return a * b + c;
#endif // __CUDA_ARCH__ >= 200
}

// Following inline functions are copied from PMEMD CUDA implementation.
// Credit goes to:
/*             Scott Le Grand (NVIDIA)             */
/*               Duncan Poole (NVIDIA)             */
/*                Ross Walker (SDSC)               */
//
// Faster ERFC approximation courtesy of Norbert Juffa. NVIDIA Corporation
static __forceinline__ __device__ float fasterfc(float a) 
{
  /* approximate log(erfc(a)) with rel. error < 7e-9 */
  float t, x = a;
  t =                       (float)-1.6488499458192755E-006;
  t = __internal_fmad(t, x, (float)2.9524665006554534E-005);
  t = __internal_fmad(t, x, (float)-2.3341951153749626E-004);
  t = __internal_fmad(t, x, (float)1.0424943374047289E-003);
  t = __internal_fmad(t, x, (float)-2.5501426008983853E-003);
  t = __internal_fmad(t, x, (float)3.1979939710877236E-004);
  t = __internal_fmad(t, x, (float)2.7605379075746249E-002);
  t = __internal_fmad(t, x, (float)-1.4827402067461906E-001);
  t = __internal_fmad(t, x, (float)-9.1844764013203406E-001);
  t = __internal_fmad(t, x, (float)-1.6279070384382459E+000);
  t = t * x;
  return exp2f(t);
}

__device__ inline unsigned long long int llitoulli(long long int l)
{
    unsigned long long int u;
    asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
    return u;
}

__device__ inline long long int lliroundf(float f)
{
    long long int l;
    asm("cvt.rni.s64.f32 	%0, %1;" : "=l"(l) : "f"(f));
    return l;
}

__device__ inline long long int lliroundd(double d)
{
    long long int l;
    asm("cvt.rni.s64.f64 	%0, %1;" : "=l"(l) : "d"(d));
    return l;
}

// End of copied code.

__device__ inline unsigned int itoui(int l)
{
    unsigned int u;
    asm("mov.b32    %0, %1;" : "=r"(u) : "r"(l));
    return u;
}

__device__ inline int iroundf(float f)
{
    int l;
    asm("cvt.rni.s32.f32 	%0, %1;" : "=r"(l) : "f"(f));
    return l;
}

// ----------------------------------------------------------------------------------------------
template <typename AT, typename CT>
__forceinline__ __device__
void calc_component_force(CT fij,
			  const CT dx, const CT dy, const CT dz,
			  AT &fxij, AT &fyij, AT &fzij) {
  fxij = (AT)(fij*dx);
  fyij = (AT)(fij*dy);
  fzij = (AT)(fij*dz);
}

template <>
__forceinline__ __device__
void calc_component_force<long long int, float>(float fij,
						const float dx, const float dy, const float dz,
						long long int &fxij,
						long long int &fyij,
						long long int &fzij) {
  fij *= FORCE_SCALE;
  fxij = lliroundf(fij*dx);
  fyij = lliroundf(fij*dy);
  fzij = lliroundf(fij*dz);
}

template <>
__forceinline__ __device__
void calc_component_force<long long int, double>(double fij,
						 const double dx, const double dy, const double dz,
						 long long int &fxij,
						 long long int &fyij,
						 long long int &fzij) {
  fij *= FORCE_SCALE;
  fxij = lliroundd(fij*dx);
  fyij = lliroundd(fij*dy);
  fzij = lliroundd(fij*dz);
}

// ----------------------------------------------------------------------------------------------
template <typename AT, typename CT>
__forceinline__ __device__
void calc_component_force(CT fij1,
			  const CT dx1, const CT dy1, const CT dz1,
			  CT fij2,
			  const CT dx2, const CT dy2, const CT dz2,
			  AT &fxij, AT &fyij, AT &fzij) {
  fxij = (AT)(fij1*dx1 + fij2*dx2);
  fyij = (AT)(fij1*dy1 + fij2*dy2);
  fzij = (AT)(fij1*dz1 + fij2*dz2);
}

template <>
__forceinline__ __device__
void calc_component_force<long long int, float>(float fij1,
						const float dx1,
						const float dy1,
						const float dz1,
						float fij2,
						const float dx2,
						const float dy2,
						const float dz2,
						long long int &fxij,
						long long int &fyij,
						long long int &fzij) {
  fij1 *= FORCE_SCALE;
  fij2 *= FORCE_SCALE;
  fxij = lliroundf(fij1*dx1 + fij2*dx2);
  fyij = lliroundf(fij1*dy1 + fij2*dy2);
  fzij = lliroundf(fij1*dz1 + fij2*dz2);
}

template <>
__forceinline__ __device__
void calc_component_force<long long int, double>(double fij1,
						 const double dx1,
						 const double dy1,
						 const double dz1,
						 double fij2,
						 const double dx2,
						 const double dy2,
						 const double dz2,
						 long long int &fxij,
						 long long int &fyij,
						 long long int &fzij) {
  fij1 *= FORCE_SCALE;
  fij2 *= FORCE_SCALE;
  fxij = lliroundd(fij1*dx1 + fij2*dx2);
  fyij = lliroundd(fij1*dy1 + fij2*dy2);
  fzij = lliroundd(fij1*dz1 + fij2*dz2);
}

// ----------------------------------------------------------------------------------------------

template <typename AT>
__forceinline__ __device__
void write_force(const AT fx, const AT fy, const AT fz,
		 const int ind, const int stride,
		 AT* force) {
  // The generic version can not be used
}

// Template specialization for 64bit integer = "long long int"
template <>
__forceinline__ __device__ 
void write_force <long long int> (const long long int fx,
				  const long long int fy,
				  const long long int fz,
				  const int ind, const int stride,
				  long long int* force) {
  atomicAdd((unsigned long long int *)&force[ind           ], llitoulli(fx));
  atomicAdd((unsigned long long int *)&force[ind + stride  ], llitoulli(fy));
  atomicAdd((unsigned long long int *)&force[ind + stride*2], llitoulli(fz));
}
// ----------------------------------------------------------------------------------------------

#endif // GPU_UTILS_H
