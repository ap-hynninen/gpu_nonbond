#include <iostream>
#include <cassert>
#include <cuda.h>
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "BondedForce.h"

// Energy and virial in device memory
static __device__ BondedEnergyVirial_t d_energy_virial;

//
// Calculates box shift
// 
// On CPU this index is calculated as:
//
// ! shift index = 1...26*3+1
// calc_ishift_{P*} = (is(1)+1 + (is(2)+1)*3 + (is(3)+1)*9 + 1)*3 - 2
//
// where is(1:3) = {-1, 0, 1}
//
__forceinline__ __device__
float3 calc_box_shift(int ish,
		      const float boxx,
		      const float boxy,
		      const float boxz) {
  float3 sh;
  ish = (ish+2)/3 - 1;
  sh.z = (ish/9 - 1)*boxz;
  ish -= (ish/9)*9;
  sh.y = (ish/3 - 1)*boxy;
  ish -= (ish/3)*3;
  sh.x = (ish - 1)*boxx;
  return sh;
}

//
// Reduces energy values
//
__forceinline__ __device__
void reduce_energy(const double epot, const unsigned int tid, volatile double *sh_epot,
		   double *global_epot) {
  sh_epot[tid] = epot;
  __syncthreads();
  for (int i=1;i < blockDim.x/2;i *= 2) {
    int t = tid + i;
    double epot_val  = (t < blockDim.x) ? sh_epot[t] : 0.0;
    __syncthreads();
    sh_epot[tid] += epot_val;
    __syncthreads();
  }
  if (tid == 0) {
    double epot_val = sh_epot[0];
    atomicAdd(global_epot, epot_val);
  }
}

//
// Templated sqrt() -function
//
template <typename T>
__forceinline__ __device__
double sqrt_template(const T x) {
  if (sizeof(T) == 4) {
    return sqrtf(x);
  } else {
    return sqrt(x);
  }
}

//
// bondcoef.x = cbb
// bondcoef.y = cbc
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_bond_force_kernel(const int nbondlist, const bondlist_t* bondlist,
				       const float2 *bondcoef, const float4 *xyzq,
				       const int stride,
				       const float boxx, const float boxy, const float boxz,
				       AT *force) {
  // Amount of shared memory required:
  // sh_epot: blockDim.x*sizeof(double)
  extern __shared__ double sh_epot[];

  int pos = threadIdx.x + blockIdx.x*blockDim.x;

  double epot;
  if (calc_energy) {
    epot = 0.0;
  }

  while (pos < nbondlist) {
    int ii = bondlist[pos].i - 1;
    int jj = bondlist[pos].j - 1;
    int ic = bondlist[pos].itype - 1;
    int ish = bondlist[pos].ishift;

    // Calculate shift for i-atom
    float3 sh_xyz = calc_box_shift(ish, boxx, boxy, boxz);

    float4 xyzqi = xyzq[ii];
    float4 xyzqj = xyzq[jj];

    //    CT dx = xyzqi.x + sh_xyz.x - xyzqj.x;
    //    CT dy = xyzqi.y + sh_xyz.y - xyzqj.y;
    //    CT dz = xyzqi.z + sh_xyz.z - xyzqj.z;

    CT dx = (CT)xyzqi.x + (CT)sh_xyz.x - (CT)xyzqj.x;
    CT dy = (CT)xyzqi.y + (CT)sh_xyz.y - (CT)xyzqj.y;
    CT dz = (CT)xyzqi.z + (CT)sh_xyz.z - (CT)xyzqj.z;

    CT r = sqrt_template<CT>(dx*dx + dy*dy + dz*dz);

    float2 bondcoef_val = bondcoef[ic];
    CT db = r - (CT)bondcoef_val.x;
    CT fij = db*(CT)bondcoef_val.y;

    if (calc_energy) {
      epot += (double)(fij*db);
    }
    fij *= ((CT)2)/r;

    AT fxij, fyij, fzij;
    calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);

    // Store forces
    write_force<AT>(fxij, fyij, fzij, ii, stride, force);
    write_force<AT>(-fxij, -fyij, -fzij, jj, stride, force);

    // Store shifted forces
    if (calc_virial) {
      //       sforce(is)   = sforce(is)   + fx
      //       sforce(is+1) = sforce(is+1) + fy
      //       sforce(is+2) = sforce(is+2) + fz
    }
    pos += blockDim.x*gridDim.x;
  }

  if (calc_energy) {
    // Reduce energy
    reduce_energy(epot, threadIdx.x, sh_epot, &d_energy_virial.energy_bond);
  }

}

//
// anglecoef.x = ctb
// anglecoef.y = ctc
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_angle_force_kernel(const int nanglelist, const anglelist_t *anglelist,
					const float2 *anglecoef, const float4 *xyzq,
					const int stride,
					const float boxx, const float boxy, const float boxz,
					AT *force) {
  int pos = threadIdx.x + blockIdx.x*blockDim.x;

  double epot;
  if (calc_energy) epot = 0.0;

  while (pos < nanglelist) {
    int ii = anglelist[pos].i - 1;
    int jj = anglelist[pos].j - 1;
    int kk = anglelist[pos].k - 1;
    int ic = anglelist[pos].itype;
    int ish = anglelist[pos].ishift1;
    int ksh = anglelist[pos].ishift2;

    // Calculate shift for i-atom
    float3 ish_xyz = calc_box_shift(ish, boxx, boxy, boxz);

    // Calculate shift for k-atom
    float3 ksh_xyz = calc_box_shift(ksh, boxx, boxy, boxz);

    CT dxij = xyzq[ii].x + ish_xyz.x - xyzq[jj].x;
    CT dyij = xyzq[ii].y + ish_xyz.y - xyzq[jj].y;
    CT dzij = xyzq[ii].z + ish_xyz.z - xyzq[jj].z;

    CT dxkj = xyzq[kk].x + ksh_xyz.x - xyzq[jj].x;
    CT dykj = xyzq[kk].y + ksh_xyz.y - xyzq[jj].y;
    CT dzkj = xyzq[kk].z + ksh_xyz.z - xyzq[jj].z;

    CT rij = sqrtf(dxij*dxij + dyij*dyij + dzij*dzij);
    CT rkj = sqrtf(dxkj*dxkj + dykj*dykj + dzkj*dzkj);

    CT rij_inv = ((CT)1)/rij;
    CT rkj_inv = ((CT)1)/rkj;

    CT dxijr = dxij*rij_inv;
    CT dyijr = dyij*rij_inv;
    CT dzijr = dzij*rij_inv;
    CT dxkjr = dxkj*rkj_inv;
    CT dykjr = dykj*rkj_inv;
    CT dzkjr = dzkj*rkj_inv;
    CT cst = dxijr*dxkjr + dyijr*dykjr + dzijr*dzkjr;

    // anglecoef.x = ctb
    // anglecoef.y = ctc
    float2 anglecoef_val = anglecoef[ic];

    // Restrict values of cst to the interval [-0.999 ... 0.999]
    // NOTE: we are ignoring the fancy stuff that is done on the CPU version
    cst = min((CT)0.999, max(-(CT)0.999, cst));

    CT at = acosf(cst);
    CT da = at - (CT)anglecoef_val.x;
    CT df = ((CT)anglecoef_val.y)*da;
    if (calc_energy) {
      epot += epot + (double)(df*da);
    }
    CT st2r = ((CT)1.0)/(((CT)1.0) - cst*cst);
    CT str = sqrtf(st2r);
    df = -((CT)2.0)*df*str;

    /*
      CPU code:
    if (fabsf(cst) >= (CT)0.999999) {
      if (fabsf(cst) > (CT)1.0) cst = (cst >= (CT)0.0) ? (CT)1.0 : -(CT)1.0; //sign(1.0f,cst);
      CT at = acosf(cst);
      CT da = at - (CT)anglecoef_val.x;
      df = ((CT)anglecoef_val.y)*da;
      if (calc_energy) {
	epot += epot + (double)(df*da);
      }
      CT st2r = ((CT)1.0)/(((CT)1.0) - cst*cst + rpreci);
      CT str = sqrtf(st2r);
      if (anglecoef_val.x < 0.001f) {
	df = -((CT)2.0)*((CT)anglecoef_val.y)*(((CT)1.0) + da*da*((CT)0.166666666666667));
      } else if ((CT)3.14159265358979323846 - (CT)anglecoef_val.x < (CT)0.001) {
	df = ((CT)2.0)*((CT)anglecoef_val.y)*(((CT)1.0) + da*da*((CT)0.166666666666667));
      } else {
	df = -((CT)2.0)*df*str;
      }
    } else {
      CT at = acosf(cst);
      CT da = at - (CT)anglecoef_val.x;
      df = ((CT)anglecoef_val.y)*da;
      if (calc_energy) {
	epot += epot + (double)(df*da);
      }
      CT st2r = ((CT)1.0)/(((CT)1.0) - cst*cst);
      CT str = sqrtf(st2r);
      df = -((CT)2.0)*df*str;
    }
    */

    CT dtxi = rij_inv*(dxkjr - cst*dxijr);
    CT dtxj = rkj_inv*(dxijr - cst*dxkjr);
    CT dtyi = rij_inv*(dykjr - cst*dyijr);
    CT dtyj = rkj_inv*(dyijr - cst*dykjr);
    CT dtzi = rij_inv*(dzkjr - cst*dzijr);
    CT dtzj = rkj_inv*(dzijr - cst*dzkjr);

    AT AT_dtxi, AT_dtyi, AT_dtzi;
    AT AT_dtxj, AT_dtyj, AT_dtzj;
    calc_component_force<AT, CT>(df, dtxi, dtyi, dtzi, AT_dtxi, AT_dtyi, AT_dtzi);
    calc_component_force<AT, CT>(df, dtxj, dtyj, dtzj, AT_dtxj, AT_dtyj, AT_dtzj);

    write_force<AT>(AT_dtxi, AT_dtyi, AT_dtzi, ii, stride, force);
    write_force<AT>(AT_dtxj, AT_dtyj, AT_dtzj, kk, stride, force);
    write_force<AT>(-AT_dtxi-AT_dtxj, -AT_dtyi-AT_dtyj, -AT_dtzi-AT_dtzj, jj, stride, force);

    /*
#ifdef PREC_SPFP
    df *= FORCE_SCALE;
    FORCE_T dtxi = lliroundf(df*dtxi);
    FORCE_T dtxj = lliroundf(df*dtxj);
    atomicAdd((unsigned long long int *)&force[ii], llitoulli(dtxi));
    atomicAdd((unsigned long long int *)&force[kk], llitoulli(dtxj));
    atomicAdd((unsigned long long int *)&force[jj], llitoulli(-dtxi-dtxj));
#endif

    if (calc_virial) {
      //       sforce(is) = sforce(is) + dtxi
      //       sforce(ks) = sforce(ks) + dtxj
    }

#ifdef PREC_SPFP
    dtyi = lliroundf(df*dtyi);
    dtyj = lliroundf(df*dtyj);
    atomicAdd((unsigned long long int *)&force[ii+stride], llitoulli(dtyi));
    atomicAdd((unsigned long long int *)&force[kk+stride], llitoulli(dtyj));
    atomicAdd((unsigned long long int *)&force[jj+stride], llitoulli(-dtyi-dtyj));
#endif


    if (calc_virial) {
      //       sforce(is+1) = sforce(is+1) + dtxi
      //       sforce(ks+1) = sforce(ks+1) + dtxj
    }

#ifdef PREC_SPFP
    const int stride2 = stride*2;
    dtzi = lliroundf(df*dtzi);
    dtzj = lliroundf(df*dtzj);
    atomicAdd((unsigned long long int *)&force[ii+stride2], llitoulli(dtzi));
    atomicAdd((unsigned long long int *)&force[kk+stride2], llitoulli(dtzj));
    atomicAdd((unsigned long long int *)&force[jj+stride2], llitoulli(-dtzi-dtzj));
#endif

    if (calc_virial) {
      //       sforce(is+2) = sforce(is+2) + dtxi
      //       sforce(ks+2) = sforce(ks+2) + dtxj
    }
    */

    pos += blockDim.x*gridDim.x;
  }

}

#ifdef NOTREADY


//
// Dihedral potential
//
// dihecoef.x = cpc
// dihecoef.y = cpcos
// dihecoef.z = cpsin
//
// Out:
// res.x = e
// res.y = df
//
static __forceinline__ __device__ CT2 dihe_pot(const int *cpd, const float3 *dihecoef,
						  const int ic_in,
						  const float st, const float ct)
{

  float e = 0.0f;
  float df = 0.0f;
  ic = ic_in

30  continue
    iper = cpd(ic)
    if (iper > 0) then
       lrep = .false.
    else
       lrep = .true.
       iper = -iper
    endif

	 e1 = 1.0f;
  df1 = 0.0f;
  ddf1 = 0.0f;

	 // Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
    do nper=1,iper
       ddf1 = e1*ct - df1*st
       df1 = e1*st + df1*ct
       e1 = ddf1
    enddo
    cpcos_val = cpcos(ic)
    cpsin_val = cpsin(ic)
    e1 = e1*cpcos_val + df1*cpsin_val
    df1 = df1*cpcos_val - ddf1*cpsin_val
    df1 = -iper*df1
    e1 = one + e1

    if(iper == 0) e1 = one

    arg = cpc(ic)
    e = e + arg*e1
    df = df + arg*df1

    if(lrep) then
       ic = ic+1
       goto 30
    endif

	       return res;
	       }

/*
//
// Improper dihedral potential
//
// imdihecoef.x = cic
// imdihecoef.y = cicos
// imdihecoef.z = cisin
//
// Out:
// res.x = e   (energy)
// res.y = df  (force)
//
static __forceinline__ __device__ float2 imdihe_pot(const int *cid, const float3 *imdihecoef,
						    const int ic_in,
						    const float st, const float ct)
{

    integer ic, iper, nper
    logical lrep
    real(chm_real4) e1, df1, ddf1, arg
    real(chm_real4) ca, sa, ap
    real(chm_real4) cicos_val, cisin_val

    ic = ic_in

##IF OPLS
    if (cid(ic) /= 0) then

       e = zero
       df = zero
35     continue
       iper = cid(ic)
       if (iper >= 0) then
          lrep = .false.
       else
          lrep = .true.
          iper = -iper
       endif
       !
       e1 = one
       df1 = zero
       !calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
       do nper=1,iper
          ddf1 = e1*ct - df1*st
          df1 = e1*st + df1*ct
          e1 = ddf1
       enddo
##IF dp
       cicos_val = cicos(ic)
       cisin_val = cisin(ic)
##ELSE
       cicos_val = real(cicos(ic),kind=chm_real4)
       cisin_val = real(cisin(ic),kind=chm_real4)
##ENDIF
       e1 = e1*cicos_val + df1*cisin_val
       df1 = df1*cicos_val - ddf1*cisin_val
       df1 = -iper*df1
       e1 = one + e1
       !
       arg = cic(ic)                       !##dp
       arg = real(cic(ic),kind=chm_real4)  !##sp
       e = e + arg*e1
       df = df + arg*df1
       !
       if(lrep) then
          ic = ic + 1
          goto 35
       endif
       !
       
       ! use harmonic potential
       !
    else
       !
##ENDIF
       !
       !calcul of cos(phi-phi0),sin(phi-phi0) and (phi-phi0).
##IF dp
       cicos_val = cicos(ic)
       cisin_val = cisin(ic)
##ELSE
       cicos_val = real(cicos(ic),kind=chm_real4)
       cisin_val = real(cisin(ic),kind=chm_real4)
##ENDIF
       ca = ct*cicos_val + st*cisin_val
       sa = st*cicos_val - ct*cisin_val
       if (ca > ptone ) then
          ap = asin(sa)
       else
          ap = sign(acos(max(ca,minone)),sa)
          ! warning is now triggered at deltaphi=84.26...deg (used to be 90).
          nbent = nbent + 1
       endif
       !
       df = cic(ic)*ap                       !##dp
       df = real(cic(ic),kind=chm_real4)*ap  !##sp
       e = df*ap
       df = two*df
    endif   !##OPLS

	    return res;
	    }
*/

//
// dihecoef.x = ctb
// dihecoef.y = ctc
//
template <typename AT, typename CT, bool calc_energy, bool calc_virial>
__global__ void calc_dihe_force_kernel(const int ndihelist, const dihelist_t *dihelist,
				       const float2 *dihecoef, const float4 *xyzq,
				       const int stride,
				       const float boxx, const float boxy, const float boxz,
				       FORCE_T *force) {
  int pos = threadIdx.x + blockIdx.x*blockDim.x;

  double epot;
  if (calc_energy) epot = 0.0;

  while (pos < ndihelist) {
    int ii = dihelist[pos].i - 1;
    int jj = dihelist[pos].j - 1;
    int kk = dihelist[pos].k - 1;
    int ll = dihelist[pos].l - 1;
    int ic = dihelist[pos].itype;
    int ish = dihelist[pos].ishift1;
    int jsh = dihelist[pos].ishift2;
    int lsh = dihelist[pos].ishift3;

    // Calculate shift for i-atom
    float3 si = calc_box_shift(ish, boxx, boxy, boxz);

    // Calculate shift for j-atom
    float3 sj = calc_box_shift(jsh, boxx, boxy, boxz);

    // Calculate shift for l-atom
    float3 sl = calc_box_shift(ksh, boxx, boxy, boxz);

    CT fx = (xyzq[ii].x + si.x) - (xyzq[jj].x + sj.x);
    CT fy = (xyzq[ii].y + si.y) - (xyzq[jj].y + sj.y);
    CT fz = (xyzq[ii].z + si.z) - (xyzq[jj].z + sj.z);

    CT gx = xyzq[jj].x + sj.x - xyzq[kk].x;
    CT gy = xyzq[jj].y + sj.y - xyzq[kk].y;
    CT gz = xyzq[jj].z + sj.z - xyzq[kk].z;

    CT hx = xyzq[ll].x + sl.x - xyzq[kk].x;
    CT hy = xyzq[ll].y + sl.y - xyzq[kk].y;
    CT hz = xyzq[ll].z + sl.z - xyzq[kk].z;

    // A=F^G, B=H^G.
    CT ax = fy*gz - fz*gy;
    CT ay = fz*gx - fx*gz;
    CT az = fx*gy - fy*gx;
    CT bx = hy*gz - hz*gy;
    CT by = hz*gx - hx*gz;
    CT bz = hx*gy - hy*gx;

    CT ra2 = ax*ax + ay*ay + az*az;
    CT rb2 = bx*bx + by*by + bz*bz;
    CT rg = sqrtf(gx*gx + gy*gy + gz*gz);

    //    if((ra2 <= rxmin2) .or. (rb2 <= rxmin2) .or. (rg <= rxmin)) then
    //          nlinear = nlinear + 1
    //       endif

    CT rgr = 1.0f / rg;
    CT ra2r = 1.0f / ra2;
    CT rb2r = 1.0f / rb2;
    CT rabr = sqrtf(ra2r*rb2r);

    // ct=cos(phi)
    CT ct = (ax*bx + ay*by + az*bz)*rabr;
    //
    // ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
    // which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
    CT st = rg*rabr*(ax*hx + ay*hy + az*hz);
    //
    //     Energy and derivative contributions.

    dihe_pot(cpd, cpc, cpcos, cpsin, ic, st, ct, e, df);

    if (calc_energy) {
      epot += epot + e;
    }

    //
    //     Compute derivatives wrt catesian coordinates.
    //
    // GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
    //  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)

    CT fg = fx*gx + fy*gy + fz*gz;
    CT hg = hx*gx + hy*gy + hz*gz;
    CT ra2r = df*ra2r;
    CT rb2r = df*rb2r;
    CT fga = fg*ra2r*rgr;
    CT hgb = hg*rb2r*rgr;
    CT gaa = ra2r*rg;
    CT gbb = rb2r*rg;
    // DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.

    // Store forces
#ifdef PREC_SPFP
    const int stride2 = stride*2;
    gaa *= FORCE_SCALE;
    FORCE_T dfx = lliroundf(-gaa*ax);
    FORCE_T dfy = lliroundf(-gaa*ay);
    FORCE_T dfz = lliroundf(-gaa*az);
    atomicAdd((unsigned long long int *)&force[ii        ], llitoulli(dfx));
    atomicAdd((unsigned long long int *)&force[ii+stride ], llitoulli(dfy));
    atomicAdd((unsigned long long int *)&force[ii+stride2], llitoulli(dfz));
    if (calc_virial) {
    //       sforce(is)   = sforce(is)   + dfx
    //       sforce(is+1) = sforce(is+1) + dfy
    //       sforce(is+2) = sforce(is+2) + dfz
    }

    FORCE_T dgx = lliroundf(fga*ax - hgb*bx);
    FORCE_T dgy = lliroundf(fga*ay - hgb*by);
    FORCE_T dgz = lliroundf(fga*az - hgb*bz);
    atomicAdd((unsigned long long int *)&force[jj        ], llitoulli(-dfx + dgx));
    atomicAdd((unsigned long long int *)&force[jj+stride ], llitoulli(-dfy + dgy));
    atomicAdd((unsigned long long int *)&force[jj+stride2], llitoulli(-dfz + dgz));
    if (calc_virial) {
	 //       sforce(js)   = sforce(js)   - dfx + dgx
	 //       sforce(js+1) = sforce(js+1) - dfy + dgy
	 //       sforce(js+2) = sforce(js+2) - dfz + dgz
    }
#endif

#ifdef PREC_SPFP
    FORCE_T dhx = lliroundf(gbb*bx);
    FORCE_T dhy = lliroundf(gbb*by);
    FORCE_T dhz = lliroundf(gbb*bz);
    atomicAdd((unsigned long long int *)&force[kk        ], llitoulli(-dhx - dgx));
    atomicAdd((unsigned long long int *)&force[kk+stride ], llitoulli(-dhy - dgy));
    atomicAdd((unsigned long long int *)&force[kk+stride2], llitoulli(-dhz - dgz));
    atomicAdd((unsigned long long int *)&force[ll        ], llitoulli(dhx));
    atomicAdd((unsigned long long int *)&force[ll+stride ], llitoulli(dhy));
    atomicAdd((unsigned long long int *)&force[ll+stride2], llitoulli(dhz));
    if (calc_virial) {
    //       sforce(ls)   = sforce(ls)   + dhx
    //       sforce(ls+1) = sforce(ls+1) + dhy
    //       sforce(ls+2) = sforce(ls+2) + dhz
    }
#endif

    pos += blockDim.x*gridDim.x;
  }

}

#endif // NOTREADY

//#############################################################################################

//
// Class creator
//
template <typename AT, typename CT>
BondedForce<AT, CT>::BondedForce() {
  nbondlist = 0;
  nbondcoef = 0;
  bondlist_len = 0;
  bondlist = NULL;
  bondcoef_len = 0;
  bondcoef = NULL;

  nureyblist = 0;
  nureybcoef = 0;
  ureyblist_len = 0;
  ureyblist = NULL;
  ureybcoef_len = 0;
  ureybcoef = NULL;

  nanglelist = 0;
  nanglecoef = 0;
  anglelist_len = 0;
  anglelist = NULL;
  anglecoef_len = 0;
  anglecoef = NULL;

  ndihelist = 0;
  ndihecoef = 0;
  dihelist_len = 0;
  dihelist = NULL;
  dihecoef_len = 0;
  dihecoef = NULL;

  nimdihelist = 0;
  nimdihecoef = 0;
  imdihelist_len = 0;
  imdihelist = NULL;
  imdihecoef_len = 0;
  imdihecoef = NULL;

  ncmaplist = 0;
  ncmapcoef = 0;
  cmaplist_len = 0;
  cmaplist = NULL;
  cmapcoef_len = 0;
  cmapcoef = NULL;

  allocate_host<BondedEnergyVirial_t>(&h_energy_virial, 1);
}

//
// Class destructor
//
template <typename AT, typename CT>
BondedForce<AT, CT>::~BondedForce() {
  if (bondlist != NULL) deallocate<bondlist_t>(&bondlist);
  if (bondcoef != NULL) deallocate<float2>(&bondcoef);

  if (ureyblist != NULL) deallocate<bondlist_t>(&ureyblist);
  if (ureybcoef != NULL) deallocate<float2>(&ureybcoef);

  if (anglelist != NULL) deallocate<anglelist_t>(&anglelist);
  if (anglecoef != NULL) deallocate<float2>(&anglecoef);

  if (dihelist != NULL) deallocate<dihelist_t>(&dihelist);
  if (dihecoef != NULL) deallocate<float2>(&dihecoef);

  if (imdihelist != NULL) deallocate<dihelist_t>(&imdihelist);
  if (imdihecoef != NULL) deallocate<float2>(&imdihecoef);

  if (cmaplist != NULL) deallocate<cmaplist_t>(&cmaplist);
  if (cmapcoef != NULL) deallocate<float2>(&cmapcoef);

  if (h_energy_virial != NULL) deallocate_host<BondedEnergyVirial_t>(&h_energy_virial);
}

//
// Setup coefficients (copies them from CPU to GPU)
// NOTE: This only has to be once in the beginning of the simulation
//
template <typename AT, typename CT>
void BondedForce<AT, CT>::setup_coef(int nbondcoef, float2 *h_bondcoef,
				     int nureybcoef, float2 *h_ureybcoef,
				     int nanglecoef, float2 *h_anglecoef,
				     int ndihecoef, float2 *h_dihecoef,
				     int nimdihecoef, float2 *h_imdihecoef,
				     int ncmapcoef, float2 *h_cmapcoef) {

  this->nbondcoef = nbondcoef;
  if (nbondcoef > 0) {
    reallocate<float2>(&bondcoef, &bondcoef_len, nbondcoef, 1.2f);
    copy_HtoD<float2>(h_bondcoef, bondcoef, nbondcoef);
  }

  this->nureybcoef = nureybcoef;
  if (nureybcoef > 0) {
    reallocate<float2>(&ureybcoef, &ureybcoef_len, nureybcoef, 1.2f);
    copy_HtoD<float2>(h_ureybcoef, ureybcoef, nureybcoef);
  }

  this->nanglecoef = nanglecoef;
  if (nanglecoef > 0) {
    reallocate<float2>(&anglecoef, &anglecoef_len, nanglecoef, 1.2f);
    copy_HtoD<float2>(h_anglecoef, anglecoef, nanglecoef);
  }

  this->ndihecoef = ndihecoef;
  if (ndihecoef > 0) {
    reallocate<float2>(&dihecoef, &dihecoef_len, ndihecoef, 1.2f);
    copy_HtoD<float2>(h_dihecoef, dihecoef, ndihecoef);
  }

  this->nimdihecoef = nimdihecoef;
  if (nimdihecoef > 0) {
    reallocate<float2>(&imdihecoef, &imdihecoef_len, nimdihecoef, 1.2f);
    copy_HtoD<float2>(h_imdihecoef, imdihecoef, nimdihecoef);
  }

  this->ncmapcoef = ncmapcoef;
  if (ncmapcoef > 0) {
    reallocate<float2>(&cmapcoef, &cmapcoef_len, ncmapcoef, 1.2f);
    copy_HtoD<float2>(h_cmapcoef, cmapcoef, ncmapcoef);
  }

}

//
// Setup bondlists (copies them from CPU to GPU)
// NOTE: This has to be done after neighborlist update
//
template <typename AT, typename CT>
void BondedForce<AT, CT>::setup_list(int nbondlist, bondlist_t *h_bondlist, 
				     int nureyblist, bondlist_t *h_ureyblist,
				     int nanglelist, anglelist_t *h_anglelist,
				     int ndihelist, dihelist_t *h_dihelist,
				     int nimdihelist, dihelist_t *h_imdihelist,
				     int ncmaplist, cmaplist_t *h_cmaplist) {
  assert(nureyblist == nanglelist);

  this->nbondlist = nbondlist;
  if (nbondlist > 0) {
    reallocate<bondlist_t>(&bondlist, &bondlist_len, nbondlist, 1.2f);
    copy_HtoD<bondlist_t>(h_bondlist, bondlist, nbondlist);
  }

  this->nureyblist = nureyblist;
  if (nureyblist > 0) {
    reallocate<bondlist_t>(&ureyblist, &ureyblist_len, nureyblist, 1.2f);
    copy_HtoD<bondlist_t>(h_ureyblist, ureyblist, nureyblist);
  }

  this->nanglelist = nanglelist;
  if (nanglelist > 0) {
    reallocate<anglelist_t>(&anglelist, &anglelist_len, nanglelist, 1.2f);
    copy_HtoD<anglelist_t>(h_anglelist, anglelist, nanglelist);
  }

  this->ndihelist = ndihelist;
  if (ndihelist > 0) {
    reallocate<dihelist_t>(&dihelist, &dihelist_len, ndihelist, 1.2f);
    copy_HtoD<dihelist_t>(h_dihelist, dihelist, ndihelist);
  }

  this->nimdihelist = nimdihelist;
  if (nimdihelist > 0) {
    reallocate<dihelist_t>(&imdihelist, &imdihelist_len, nimdihelist, 1.2f);
    copy_HtoD<dihelist_t>(h_imdihelist, imdihelist, nimdihelist);
  }

  this->ncmaplist = ncmaplist;
  if (ncmaplist > 0) {
    reallocate<cmaplist_t>(&cmaplist, &cmaplist_len, ncmaplist, 1.2f);
    copy_HtoD<cmaplist_t>(h_cmaplist, cmaplist, ncmaplist);
  }

}

//
// Calculates forces
//
template <typename AT, typename CT>
void BondedForce<AT, CT>::calc_force(const float4 *xyzq,
				     const float boxx, const float boxy, const float boxz,
				     const bool calc_energy,
				     const bool calc_virial,
				     const int stride, AT *force, cudaStream_t stream) {

  int nthread, nblock, shmem_size;

  nthread = 512;
  nblock = (nbondlist -1)/nthread + 1;
  shmem_size = 0;
  if (calc_energy) {
    shmem_size += nthread*sizeof(double);
  }

  if (calc_energy) {
    std::cerr << "BondedForce<AT, CT>::calc_force, calc_energy not implemented yet" << std::endl;
  } else {
    if (calc_virial) {
      std::cerr << "BondedForce<AT, CT>::calc_force, calc_virial not implemented yet" << std::endl;
    } else {
      nthread = 512;
      nblock = (nbondlist -1)/nthread + 1;
      shmem_size = 0;
      if (calc_energy) shmem_size += nthread*sizeof(double);
      calc_bond_force_kernel<AT, CT, false, false >
	<<< nblock, nthread, shmem_size, stream >>>
	(nbondlist, bondlist, bondcoef, xyzq, stride, boxx, boxy, boxz, force);
      cudaCheck(cudaGetLastError());
      
      /*
      nthread = 512;
      nblock = (nbondlist -1)/nthread + 1;
      shmem_size = 0;
      if (calc_energy) shmem_size += nthread*sizeof(double);
      calc_angle_force_kernel<AT, CT, false, false >
	<<< nblock, nthread, shmem_size, stream >>>
	(nanglelist, anglelist, anglecoef, xyzq, stride, boxx, boxy, boxz, force);
      cudaCheck(cudaGetLastError());
      */

    }
  }

}

//
// Sets Energies and virials to zero
//
template <typename AT, typename CT>
void BondedForce<AT, CT>::clear_energy_virial() {
  h_energy_virial->energy_bond = 0.0;
  h_energy_virial->energy_ureyb = 0.0;
  h_energy_virial->energy_angle = 0.0;
  h_energy_virial->energy_dihe = 0.0;
  h_energy_virial->energy_imdihe = 0.0;
  h_energy_virial->energy_cmap = 0.0;
  for (int i=0;i < 27;i++) {
    h_energy_virial->sforcex[i] = 0.0;
    h_energy_virial->sforcey[i] = 0.0;
    h_energy_virial->sforcez[i] = 0.0;
  }
  cudaCheck(cudaMemcpyToSymbol(d_energy_virial, h_energy_virial, sizeof(BondedEnergyVirial_t)));
}

//
// Read Energies and virials
// prev_calc_energy = true, if energy was calculated when the force kernel was last called
// prev_calc_virial = true, if virial was calculated when the force kernel was last called
//
template <typename AT, typename CT>
void BondedForce<AT, CT>::get_energy_virial(bool prev_calc_energy, bool prev_calc_virial,
					    double *energy_bond, double *energy_ureyb,
					    double *energy_angle,
					    double *energy_dihe, double *energy_imdihe,
					    double *energy_cmap,
					    double *sforcex, double *sforcey, double *sforcez) {
  if (prev_calc_energy && prev_calc_virial) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, sizeof(BondedEnergyVirial_t)));
  } else if (prev_calc_energy) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, 6*sizeof(double)));
  } else if (prev_calc_virial) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, 27*3*sizeof(double),
				   6*sizeof(double)));
  }
  *energy_bond = h_energy_virial->energy_bond;
  *energy_ureyb = h_energy_virial->energy_ureyb;
  *energy_angle = h_energy_virial->energy_angle;
  *energy_dihe = h_energy_virial->energy_dihe;
  *energy_imdihe = h_energy_virial->energy_imdihe;
  *energy_cmap = h_energy_virial->energy_cmap;
  for (int i=0;i < 27;i++) {
    sforcex[i] = h_energy_virial->sforcex[i];
    sforcey[i] = h_energy_virial->sforcey[i];
    sforcez[i] = h_energy_virial->sforcez[i];
  }
}

//
// Explicit instances of BondedForce
//
template class BondedForce<long long int, float>;
template class BondedForce<long long int, double>;

