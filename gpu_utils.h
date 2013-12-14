
#include <stdio.h>

static __constant__ const float FORCE_SCALE = (float)(1ll << 40);
static __constant__ const double INV_FORCE_SCALE = (double)1.0/(double)(1ll << 40);

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
// End of copied code.

