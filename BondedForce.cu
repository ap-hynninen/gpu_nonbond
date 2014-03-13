#include <iostream>
#include <cuda.h>
#include "cuda_utils.h"
#include "gpu_utils.h"
#include "BondedForce.h"

// Energy and virial in device memory
static __device__ BondedEnergyVirial_t d_energy_virial;


__forceinline__ __device__
float3 calc_box_shift(int ish,
		      const float boxx,
		      const float boxy,
		      const float boxz) {
  float3 sh;
  sh.z = (ish/9 - 1)*boxz;
  ish -= (ish/9)*9;
  sh.y = (ish/3 - 1)*boxy;
  ish -= (ish/3)*3;
  sh.x = (ish - 1)*boxx;
  return sh;
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
    int ii = bondlist[pos].i;
    int jj = bondlist[pos].j;
    int ic = bondlist[pos].itype;
    int ish = bondlist[pos].ishift;

    // Calculate shift for i-atom
    float3 sh_xyz = calc_box_shift(ish, boxx, boxy, boxz);

    // NOTE: Take this into account in building bondcoef
    //    if (ic == 0) cycle

    float4 xyzqi = xyzq[ii];
    float4 xyzqj = xyzq[jj];

    float dx = xyzqi.x + sh_xyz.x - xyzqj.x;
    float dy = xyzqi.y + sh_xyz.y - xyzqj.y;
    float dz = xyzqi.z + sh_xyz.z - xyzqj.z;

    float r = sqrtf(dx*dx + dy*dy + dz*dz);

    float2 bondcoef_val = bondcoef[ic];
    float db = r - bondcoef_val.x;
    float fij = db*bondcoef_val.y;

    if (calc_energy) {
      epot += (double)(fij*db);
    }
    fij *= 2.0f/r;

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
    sh_epot[threadIdx.x] = epot;
    __syncthreads();
    for (int i=1;i < blockDim.x/2;i *= 2) {
      int t = threadIdx.x + i;
      double epot_val  = (t < blockDim.x) ? sh_epot[t] : 0.0;
      __syncthreads();
      sh_epot[threadIdx.x] += epot_val;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      double epot_val = sh_epot[0];
      atomicAdd(&d_energy_virial.energy_bond, epot_val);
    }

  }

}

#ifdef NOTREADY

//
// anglecoef.x = ctb
// anglecoef.y = ctc
//
__global__ void eangle_kernel(const int nanglelist, const anglelist_t *anglelist,
			      const float2 *anglecoef, const float4 *xyzq,
			      const int stride,
			      const float boxx, const float boxy, const float boxz,
			      FORCE_T *force) {
  int pos = threadIdx.x + blockIdx.x*blockDim.x;

  while (pos < nanglelist) {
    int ii = anglelist[pos].i;
    int jj = anglelist[pos].j;
    int kk = anglelist[pos].k;
    int ic = anglelist[pos].itype;
    int ish = anglelist[pos].ishift1;
    int ksh = anglelist[pos].ishift2;

    // Calculate shift for i-atom
    float3 ish_xyz = calc_box_shift(ish, boxx, boxy, boxz);

    // Calculate shift for k-atom
    float3 ksh_xyz = calc_box_shift(ksh, boxx, boxy, boxz);

    // NOTE: Take this into account when building anglecoef
    // if (ic == 0) cycle

    float dxij = xyzq[ii].x + ish_xyz.x - xyzq[jj].x;
    float dyij = xyzq[ii].y + ish_xyz.y - xyzq[jj].y;
    float dzij = xyzq[ii].z + ish_xyz.z - xyzq[jj].z;

    float dxkj = xyzq[kk].x + ksh_xyz.x - xyzq[jj].x;
    float dykj = xyzq[kk].y + ksh_xyz.y - xyzq[jj].y;
    float dzkj = xyzq[kk].z + ksh_xyz.z - xyzq[jj].z;

    float rij = sqrtf(dxij*dxij + dyij*dyij + dzij*dzij);
    float rkj = sqrtf(dxkj*dxkj + dykj*dykj + dzkj*dzkj);

    float rij_inv = 1.0f/rij;
    float rkj_inv = 1.0f/rkj;

    float dxijr = dxij*rij_inv;
    float dyijr = dyij*rij_inv;
    float dzijr = dzij*rij_inv;
    float dxkjr = dxkj*rkj_inv;
    float dykjr = dykj*rkj_inv;
    float dzkjr = dzkj*rkj_inv;
    float cst = dxijr*dxkjr + dyijr*dykjr + dzijr*dzkjr;

    float2 anglecoef_val = anglecoef[ic];

    if (fabsf(cst) >= 0.99f) {
      if (abs(cst) > 1.0f) cst = sign(1.0f,cst);
      at = acos(cst);
      da = at - ctb_val;
      df = ctc_val*da;

#ifdef CALC_ENERGY
      //epot = epot + real(df*da,kind=chm_real);
#endif
      st2r = 1.0f/(1.0f - cst*cst + rpreci);
      str = sqrtf(st2r);
      if (ctb_val < 0.001f) {
	df = -2.0f*ctc_val*(one + da*da*sixth);
      } else if (pi_val-ctb_val < 0.001) {
	df = 2.0f*ctc_val*(one + da*da*sixth);
      } else {
	df = mintwo*df*str;
      }
    } else {
      at = acos(cst);
      da = at - ctb_val;
      df = ctc_val*da;
#ifdef CALC_ENERGY
      epot = epot + real(df*da,kind=chm_real);
#endif
      st2r = one/(one - cst*cst);
      str = sqrtf(st2r);
      df = mintwo*df*str;
    }

    float dtxi = rij_inv*(dxkjr - cst*dxijr);
    float dtxj = rkj_inv*(dxijr - cst*dxkjr);
    float dtyi = rij_inv*(dykjr - cst*dyijr);
    float dtyj = rkj_inv*(dyijr - cst*dykjr);
    float dtzi = rij_inv*(dzkjr - cst*dzijr);
    float dtzj = rkj_inv*(dzijr - cst*dzkjr);

#ifdef PREC_SPFP
    df *= FORCE_SCALE;
    FORCE_T dti = lliroundf(df*dtxi);
    FORCE_T dtj = lliroundf(df*dtxj);
    atomicAdd((unsigned long long int *)&force[ii], llitoulli(dti));
    atomicAdd((unsigned long long int *)&force[kk], llitoulli(dtj));
    atomicAdd((unsigned long long int *)&force[jj], llitoulli(-dti-dtj));
#endif

#ifdef CALC_VIRIAL
	 //       sforce(is) = sforce(is) + dti
	 //       sforce(ks) = sforce(ks) + dtj
#endif

#ifdef PREC_SPFP
    dti = lliroundf(df*dtyi);
    dtj = lliroundf(df*dtyj);
    atomicAdd((unsigned long long int *)&force[ii+stride], llitoulli(dti));
    atomicAdd((unsigned long long int *)&force[kk+stride], llitoulli(dtj));
    atomicAdd((unsigned long long int *)&force[jj+stride], llitoulli(-dti-dtj));
#endif


#ifdef CALC_VIRIAL
	 //       sforce(is+1) = sforce(is+1) + dti
	 //       sforce(ks+1) = sforce(ks+1) + dtj
#endif

#ifdef PREC_SPFP
    const int stride2 = stride*2;
    dti = lliroundf(df*dtzi);
    dtj = lliroundf(df*dtzj);
    atomicAdd((unsigned long long int *)&force[ii+stride2], llitoulli(dti));
    atomicAdd((unsigned long long int *)&force[kk+stride2], llitoulli(dtj));
    atomicAdd((unsigned long long int *)&force[jj+stride2], llitoulli(-dti-dtj));
#endif

#ifdef CALC_VIRIAL
	 //       sforce(is+2) = sforce(is+2) + dti
	 //       sforce(ks+2) = sforce(ks+2) + dtj
#endif

    pos += blockDim.x*gridDim.x;
  }

}

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
static __forceinline__ __device__ float2 dihe_pot(const int *cpd, const float3 *dihecoef,
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
__global__ void edihe_kernel(const int ndihelist, const dihelist_t *dihelist,
			      const float2 *dihecoef, const float4 *xyzq,
			      const int stride,
			      const float boxx, const float boxy, const float boxz,
			      FORCE_T *force) {
  int pos = threadIdx.x + blockIdx.x*blockDim.x;

  while (pos < ndihelist) {
    int ii = dihelist[pos].i;
    int jj = dihelist[pos].j;
    int kk = dihelist[pos].k;
    int ll = dihelist[pos].l;
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

    float fx = (xyzq[ii].x + si.x) - (xyzq[jj].x + sj.x);
    float fy = (xyzq[ii].y + si.y) - (xyzq[jj].y + sj.y);
    float fz = (xyzq[ii].z + si.z) - (xyzq[jj].z + sj.z);

    float gx = xyzq[jj].x + sj.x - xyzq[kk].x;
    float gy = xyzq[jj].y + sj.y - xyzq[kk].y;
    float gz = xyzq[jj].z + sj.z - xyzq[kk].z;

    float hx = xyzq[ll].x + sl.x - xyzq[kk].x;
    float hy = xyzq[ll].y + sl.y - xyzq[kk].y;
    float hz = xyzq[ll].z + sl.z - xyzq[kk].z;

    // A=F^G, B=H^G.
    float ax = fy*gz - fz*gy;
    float ay = fz*gx - fx*gz;
    float az = fx*gy - fy*gx;
    float bx = hy*gz - hz*gy;
    float by = hz*gx - hx*gz;
    float bz = hx*gy - hy*gx;

    float ra2 = ax*ax + ay*ay + az*az;
    float rb2 = bx*bx + by*by + bz*bz;
    float rg = sqrtf(gx*gx + gy*gy + gz*gz);

    //    if((ra2 <= rxmin2) .or. (rb2 <= rxmin2) .or. (rg <= rxmin)) then
    //          nlinear = nlinear + 1
    //       endif

    float rgr = 1.0f / rg;
    float ra2r = 1.0f / ra2;
    float rb2r = 1.0f / rb2;
    float rabr = sqrtf(ra2r*rb2r);

    // ct=cos(phi)
    float ct = (ax*bx + ay*by + az*bz)*rabr;
    //
    // ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
    // which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
    float st = rg*rabr*(ax*hx + ay*hy + az*hz);
    //
    //     Energy and derivative contributions.

    dihe_pot(cpd, cpc, cpcos, cpsin, ic, st, ct, e, df);

#ifdef CALC_ENERGY
    //       epot = epot + real(e,kind=chm_real4)  !##sp
#endif

    //
    //     Compute derivatives wrt catesian coordinates.
    //
    // GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
    //  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)

    float fg = fx*gx + fy*gy + fz*gz;
    float hg = hx*gx + hy*gy + hz*gz;
    float ra2r = df*ra2r;
    float rb2r = df*rb2r;
    float fga = fg*ra2r*rgr;
    float hgb = hg*rb2r*rgr;
    float gaa = ra2r*rg;
    float gbb = rb2r*rg;
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
#ifdef CALC_VIRIAL
    //       sforce(is)   = sforce(is)   + dfx
    //       sforce(is+1) = sforce(is+1) + dfy
    //       sforce(is+2) = sforce(is+2) + dfz
#endif

    FORCE_T dgx = lliroundf(fga*ax - hgb*bx);
    FORCE_T dgy = lliroundf(fga*ay - hgb*by);
    FORCE_T dgz = lliroundf(fga*az - hgb*bz);
    atomicAdd((unsigned long long int *)&force[jj        ], llitoulli(-dfx + dgx));
    atomicAdd((unsigned long long int *)&force[jj+stride ], llitoulli(-dfy + dgy));
    atomicAdd((unsigned long long int *)&force[jj+stride2], llitoulli(-dfz + dgz));
#ifdef CALC_VIRIAL
	 //       sforce(js)   = sforce(js)   - dfx + dgx
	 //       sforce(js+1) = sforce(js+1) - dfy + dgy
	 //       sforce(js+2) = sforce(js+2) - dfz + dgz
#endif
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
#ifdef CALC_VIRIAL
    //       sforce(ls)   = sforce(ls)   + dhx
    //       sforce(ls+1) = sforce(ls+1) + dhy
    //       sforce(ls+2) = sforce(ls+2) + dhz
#endif
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
  allocate_host<BondedEnergyVirial_t>(&h_energy_virial, 1);
}

//
// Class destructor
//
template <typename AT, typename CT>
BondedForce<AT, CT>::~BondedForce() {
  if (h_energy_virial != NULL) deallocate_host<BondedEnergyVirial_t>(&h_energy_virial);
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

  int nthread = 512;
  int nblock = (nbondlist -1)/nthread + 1;
  int shmem_size = 0;
  if (calc_energy) {
    shmem_size += nthread*sizeof(double);
  }

  if (calc_energy) {
    std::cerr << "BondedForce<AT, CT>::calc_force, calc_energy not implemented yet" << std::endl;
  } else {
    if (calc_virial) {
      std::cerr << "BondedForce<AT, CT>::calc_force, calc_virial not implemented yet" << std::endl;
    } else {
      calc_bond_force_kernel<AT, CT, false, false >
	<<< nblock, nthread, shmem_size, stream >>>
	(nbondlist, bondlist, bondcoef, xyzq, stride, boxx, boxy, boxz, force);
    }
  }

}

//
// Sets Energies and virials to zero
//
template <typename AT, typename CT>
void BondedForce<AT, CT>::clear_energy_virial() {
  h_energy_virial->energy_bond = 0.0;
  h_energy_virial->energy_angle = 0.0;
  h_energy_virial->energy_dihe = 0.0;
  h_energy_virial->energy_imdihe = 0.0;
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
					    double *energy_bond, double *energy_angle,
					    double *energy_dihe, double *energy_imdihe,
					    double *sforcex, double *sforcey, double *sforcez) {
  if (prev_calc_energy && prev_calc_virial) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, sizeof(BondedEnergyVirial_t)));
  } else if (prev_calc_energy) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, 4*sizeof(double)));
  } else if (prev_calc_virial) {
    cudaCheck(cudaMemcpyFromSymbol(h_energy_virial, d_energy_virial, 27*3*sizeof(double),
				   4*sizeof(double)));
  }
  *energy_bond = h_energy_virial->energy_bond;
  *energy_angle = h_energy_virial->energy_angle;
  *energy_dihe = h_energy_virial->energy_dihe;
  *energy_imdihe = h_energy_virial->energy_imdihe;
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

