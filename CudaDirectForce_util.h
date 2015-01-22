//
// CUDA device functions for direct force calculation
//

#define USE_TEXTURES true

#define CREATE_KERNEL(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, CALC_ENERGY, CALC_VIRIAL, TEX_VDWPARAM, ...) \
  {									\
    KERNEL_NAME <AT, CT, tilesize, VDW_MODEL, ELEC_MODEL, CALC_ENERGY, CALC_VIRIAL, TEX_VDWPARAM> \
      <<< nblock, nthread, shmem_size, stream >>>			\
      (__VA_ARGS__);							\
  }

#define CREATE_KERNEL14(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, CALC_ENERGY, CALC_VIRIAL, TEX_VDWPARAM, ...) \
  {									\
    KERNEL_NAME <AT, CT, VDW_MODEL, ELEC_MODEL, CALC_ENERGY, CALC_VIRIAL, TEX_VDWPARAM> \
      <<< nblock, nthread, shmem_size, stream >>>			\
      (__VA_ARGS__);							\
  }

#define EXPAND_ENERGY_VIRIAL(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL, ELEC_MODEL, ...) \
  {									\
    if (calc_energy) {							\
      if (calc_virial) {						\
	KERNEL_CREATOR(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, true, true, USE_TEXTURES, __VA_ARGS__); \
      } else {								\
	KERNEL_CREATOR(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, true, false, USE_TEXTURES, __VA_ARGS__); \
      }									\
    } else {								\
      if (calc_virial) {						\
	KERNEL_CREATOR(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, false, true, USE_TEXTURES, __VA_ARGS__); \
      } else {								\
	KERNEL_CREATOR(KERNEL_NAME, VDW_MODEL, ELEC_MODEL, false, false, USE_TEXTURES, __VA_ARGS__); \
      }									\
    }									\
  }

#define EXPAND_ELEC(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL, ...)	\
  {									\
    if (elec_model_loc == EWALD) {					\
      EXPAND_ENERGY_VIRIAL(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL, EWALD, __VA_ARGS__); \
    } else if (elec_model_loc == EWALD_LOOKUP) {			\
      EXPAND_ENERGY_VIRIAL(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL, EWALD_LOOKUP, __VA_ARGS__); \
    } else if (elec_model_loc == GSHFT) {				\
      EXPAND_ENERGY_VIRIAL(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL, GSHFT, __VA_ARGS__); \
    } else if (elec_model_loc == NONE) {				\
      EXPAND_ENERGY_VIRIAL(KERNEL_CREATOR, KERNEL_NAME, VDW_MODEL, NONE, __VA_ARGS__); \
    } else {								\
      std::cout<<__func__<<" Invalid EWALD model "<<elec_model_loc<<std::endl; \
      exit(1);								\
    }									\
  }

#define CREATE_KERNELS(KERNEL_CREATOR, KERNEL_NAME, ...)		\
  {									\
    if (vdw_model_loc == VDW_VSH) {					\
      EXPAND_ELEC(KERNEL_CREATOR, KERNEL_NAME, VDW_VSH, __VA_ARGS__);	\
    } else if (vdw_model_loc == VDW_VSW) {				\
      EXPAND_ELEC(KERNEL_CREATOR, KERNEL_NAME, VDW_VSW, __VA_ARGS__);	\
    } else if (vdw_model_loc == VDW_VFSW) {				\
      EXPAND_ELEC(KERNEL_CREATOR, KERNEL_NAME, VDW_VFSW, __VA_ARGS__);	\
    } else if (vdw_model_loc == VDW_CUT) {				\
      EXPAND_ELEC(KERNEL_CREATOR, KERNEL_NAME, VDW_CUT, __VA_ARGS__);	\
    } else if (vdw_model_loc == VDW_VGSH) {				\
      EXPAND_ELEC(KERNEL_CREATOR, KERNEL_NAME, VDW_VGSH, __VA_ARGS__);	\
    } else {								\
      std::cout<<__func__<<" Invalid VDW model "<<vdw_model_loc<<std::endl; \
      exit(1);								\
    }									\
  }

static __constant__ const float ccelec = 332.0716;

//
// Calculates VdW pair force & energy
// NOTE: force (fij_vdw) is r*dU/dr
//
template <int vdw_model, bool calc_energy>
__forceinline__ __device__
float pair_vdw_force(const float r2, const float r, const float rinv, const float rinv2,
		     const float c6, const float c12,float &pot_vdw) {

  float fij_vdw;

  if (vdw_model == VDW_VSH) {
    float r6 = r2*r2*r2;
    float rinv6 = rinv2*rinv2*rinv2;
    float rinv12 = rinv6*rinv6;
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      pot_vdw = c12*one_twelve*(rinv12 + 2.0f*r6*d_setup.roffinv18 - 3.0f*d_setup.roffinv12)-
	c6*one_six*(rinv6 + r6*d_setup.roffinv12 - 2.0f*d_setup.roffinv6);
    }
	  
    fij_vdw = c6*(rinv6 - r6*d_setup.roffinv12) - c12*(rinv12 + r6*d_setup.roffinv18);
  } else if (vdw_model == VDW_VSW) {
    float roff2_r2_sq = d_setup.roff2 - r2;
    roff2_r2_sq *= roff2_r2_sq;
    float sw = (r2 <= d_setup.ron2) ? 1.0f : 
      roff2_r2_sq*(d_setup.roff2 + 2.0f*r2 - 3.0f*d_setup.ron2)*d_setup.inv_roff2_ron2;
    // dsw_6 = dsw/6.0
    float dsw_6 = (r2 <= d_setup.ron2) ? 0.0f : 
      (d_setup.roff2-r2)*(d_setup.ron2-r2)*d_setup.inv_roff2_ron2;
    float rinv4 = rinv2*rinv2;
    float rinv6 = rinv4*rinv2;
    fij_vdw = rinv4*( c12*rinv6*(dsw_6 - sw*rinv2) - c6*(2.0f*dsw_6 - sw*rinv2) );
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      pot_vdw = sw*rinv6*(one_twelve*c12*rinv6 - one_six*c6);
    }
  } else if (vdw_model == VDW_CUT) {
    float rinv6 = rinv2*rinv2*rinv2;
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      float rinv12 = rinv6*rinv6;
      pot_vdw = c12*one_twelve*rinv12 - c6*one_six*rinv6;
      fij_vdw = c6*rinv6 - c12*rinv12;
    } else {
      fij_vdw = c6*rinv6 - c12*rinv6*rinv6;
    }
  } else if (vdw_model == VDW_VFSW) {
    float rinv3 = rinv*rinv2;
    float rinv6 = rinv3*rinv3;
    float A6 = (r2 > d_setup.ron2) ? d_setup.k6 : 1.0f;
    float B6 = (r2 > d_setup.ron2) ? d_setup.roffinv3  : 0.0f;
    float A12 = (r2 > d_setup.ron2) ? d_setup.k12 : 1.0f;
    float B12 = (r2 > d_setup.ron2) ? d_setup.roffinv6 : 0.0f;
    fij_vdw = c6*A6*(rinv3 - B6)*rinv3 - c12*A12*(rinv6 - B12)*rinv6;
    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      float C6  = (r2 > d_setup.ron2) ? 0.0f : d_setup.dv6;
      float C12 = (r2 > d_setup.ron2) ? 0.0f : d_setup.dv12;

      float rinv3_B6_sq = rinv3 - B6;
      rinv3_B6_sq *= rinv3_B6_sq;

      float rinv6_B12_sq = rinv6 - B12;
      rinv6_B12_sq *= rinv6_B12_sq;

      pot_vdw = one_twelve*c12*(A12*rinv6_B12_sq + C12) - one_six*c6*(A6*rinv3_B6_sq + C6);
    }
  } else if (vdw_model == VDW_VGSH) {
    float rinv3 = rinv*rinv2;
    float rinv6 = rinv3*rinv3;
    float rinv12 = rinv6*rinv6;
    float r_ron = (r2 > d_setup.ron2) ? (r-d_setup.ron) : 0.0f;
    float r_ron2_r = r_ron*r_ron*r;

    fij_vdw = c6*(rinv6 + (d_setup.ga6 + d_setup.gb6*r_ron)*r_ron2_r ) -
      c12*(rinv12 + (d_setup.ga12 + d_setup.gb12*r_ron)*r_ron2_r );

    if (calc_energy) {
      const float one_twelve = 0.0833333333333333f;
      const float one_six = 0.166666666666667f;
      const float one_third = (float)(1.0/3.0);
      float r_ron3 = r_ron*r_ron*r_ron;
      pot_vdw = c6*(-one_six*rinv6 + (one_third*d_setup.ga6 + 0.25f*d_setup.gb6*r_ron)*r_ron3 
		    + d_setup.gc6) +
	c12*(one_twelve*rinv12 - (one_third*d_setup.ga12 + 0.25f*d_setup.gb12*r_ron)*r_ron3 
	     - d_setup.gc12);
    }
    /*
    if (r > ctonnb) then
             d = 6.0f/r**7 + GA6*(r-ctonnb)**2 + GB6*(r-ctonnb)**3
             d = -(12.0f/r**13 + GA12*(r-ctonnb)**2 + GB12*(r-ctonnb)**3)

             e = -r**(-6) + (GA6*(r-ctonnb)**3)/3.0 + (GB6*(r-ctonnb)**4)/4.0 + GC6
             e = r**(-12) - (GA12*(r-ctonnb)**3)/3.0 - (GB12*(r-ctonnb)**4)/4.0 - GC12

          else
             d = 6.0f/r**7
             d = -12.0f/r**13

             e = - r**(-6) + GC6
             e = r**(-12) - GC12
          endif
    */
  } else if (vdw_model == NONE) {
    fij_vdw = 0.0f;
    if (calc_energy) {
      pot_vdw = 0.0f;
    }
  }

  return fij_vdw;
}

//static texture<float, 1, cudaReadModeElementType> ewald_force_texref;

//
// Returns simple linear interpolation
// NOTE: Could the interpolation be done implicitly using the texture unit?
//
__forceinline__ __device__ float lookup_force(const float r, const float hinv) {
  float r_hinv = r*hinv;
  int ind = (int)r_hinv;
  float f1 = r_hinv - (float)ind;
  float f2 = 1.0f - f1;
#if __CUDA_ARCH__ < 350
  return f1*d_setup.ewald_force[ind] + f2*d_setup.ewald_force[ind+1];
#else
  return f1*__ldg(&d_setup.ewald_force[ind]) + f2*__ldg(&d_setup.ewald_force[ind+1]);
#endif
  //return f1*tex1Dfetch(ewald_force_texref, ind) + f2*tex1Dfetch(ewald_force_texref, ind+1);
}

//
// Calculates electrostatic force & energy
//
template <int elec_model, bool calc_energy>
__forceinline__ __device__
float pair_elec_force(const float r2, const float r, const float rinv, 
		      const float qq, float &pot_elec) {

  float fij_elec;

  if (elec_model == EWALD_LOOKUP) {
    fij_elec = qq*lookup_force(r, d_setup.hinv);
  } else if (elec_model == EWALD) {
    float erfc_val = fasterfc(d_setup.kappa*r);
    float exp_val = expf(-d_setup.kappa2*r2);
    if (calc_energy) {
      pot_elec = qq*erfc_val*rinv;
    }
    const float two_sqrtpi = 1.12837916709551f;    // 2/sqrt(pi)
    fij_elec = qq*(two_sqrtpi*d_setup.kappa*exp_val + erfc_val*rinv);
  } else if (elec_model == GSHFT) {
    // GROMACS style shift 1/r^2 force
    // MGL special casing ctonnb=0 might speed this up
    // NOTE THAT THIS EXPLICITLY ASSUMES ctonnb = 0
    //ctofnb4 = ctofnb2*ctofnb2
    //ctofnb5 = ctofnb4*ctofnb
    fij_elec = qq*(rinv - (5.0f*d_setup.roffinv4*r - 4.0f*d_setup.roffinv5*r2)*r2 );
    //d = -qscale*(one/r2 - 5.0*r2/ctofnb4 +4*r2*r/ctofnb5)
    if (calc_energy) {
      pot_elec = qq*(rinv - d_setup.GAconst + (d_setup.GBcoef*r - d_setup.roffinv5*r2)*r2);
      //e = qscale*(one/r - GAconst + r*r2*GBcoef - r2*r2/ctofnb5)
    }
  } else if (elec_model == NONE) {
    fij_elec = 0.0f;
    if (calc_energy) {
      pot_elec = 0.0f;
    }
  }

  return fij_elec;
}

//
// Calculates electrostatic force & energy for 1-4 interactions and exclusions
//
template <int elec_model, bool calc_energy>
__forceinline__ __device__
float pair_elec_force_14(const float r2, const float r, const float rinv,
			 const float qq, const float e14fac, float &pot_elec) {

  float fij_elec;

  if (elec_model == EWALD) {
    float erfc_val = fasterfc(d_setup.kappa*r);
    float exp_val = expf(-d_setup.kappa2*r2);
    float qq_efac_rinv = qq*(erfc_val + e14fac - 1.0f)*rinv;
    if (calc_energy) {
      pot_elec = qq_efac_rinv;
    }
    const float two_sqrtpi = 1.12837916709551f;    // 2/sqrt(pi)
    fij_elec = -qq*two_sqrtpi*d_setup.kappa*exp_val - qq_efac_rinv;
  } else if (elec_model == NONE) {
    fij_elec = 0.0f;
    if (calc_energy) {
      pot_elec = 0.0f;
    }
  }

  return fij_elec;
}

//
// 1-4 interaction force
//
template <typename AT, typename CT, int vdw_model, int elec_model, 
	  bool calc_energy, bool calc_virial, bool tex_vdwparam>
__device__ void calc_in14_force_device(
#ifdef USE_TEXTURE_OBJECTS
				       const cudaTextureObject_t tex,
#endif
				       const int pos, const xx14list_t* in14list,
				       const int* vdwtype, const float* vdwparam14,
				       const float4* xyzq, const int stride, AT *force,
				       double &vdw_pot, double &elec_pot) {

  int i = in14list[pos].i;
  int j = in14list[pos].j;
  int ish = in14list[pos].ishift;
  float3 sh_xyz = calc_box_shift(ish, d_setup.boxx, d_setup.boxy, d_setup.boxz);
  // Load atom coordinates
  float4 xyzqi = xyzq[i];
  float4 xyzqj = xyzq[j];
  // Calculate distance
  CT dx = xyzqi.x - xyzqj.x + sh_xyz.x;
  CT dy = xyzqi.y - xyzqj.y + sh_xyz.y;
  CT dz = xyzqi.z - xyzqj.z + sh_xyz.z;
  CT r2 = dx*dx + dy*dy + dz*dz;
  CT qq = ccelec*xyzqi.w*xyzqj.w;
  // Calculate the interaction
  CT r = sqrtf(r2);
  CT rinv = ((CT)1)/r;

  int ia = vdwtype[i];
  int ja = vdwtype[j];
  int aa = max(ja, ia);

  CT c6, c12;
  if (tex_vdwparam) {
    int ivdw = (aa*(aa-3) + 2*(ja + ia) - 2) >> 1;
    //c6 = __ldg(&vdwparam14[ivdw]);
    //c12 = __ldg(&vdwparam14[ivdw+1]);
#ifdef USE_TEXTURE_OBJECTS
    float2 c6c12 = tex1Dfetch<float2>(tex, ivdw);
#else
    float2 c6c12 = tex1Dfetch(VDWPARAM14_TEXREF, ivdw);
#endif
    c6  = c6c12.x;
    c12 = c6c12.y;
  } else {
    int ivdw = (aa*(aa-3) + 2*(ja + ia) - 2);
    c6 = vdwparam14[ivdw];
    c12 = vdwparam14[ivdw+1];
  }

  CT rinv2 = rinv*rinv;

  float dpot_vdw;
  CT fij_vdw = pair_vdw_force<vdw_model, calc_energy>(r2, r, rinv, rinv2, c6, c12, dpot_vdw);
  if (calc_energy) vdw_pot += (double)dpot_vdw;

  float dpot_elec;
  CT fij_elec = pair_elec_force_14<elec_model, calc_energy>(r2, r, rinv, qq,
  							    d_setup.e14fac, dpot_elec);
  if (calc_energy) elec_pot += (double)dpot_elec;

  CT fij = (fij_vdw + fij_elec)*rinv2;

  // Calculate force components
  AT fxij, fyij, fzij;
  calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);

  // Store forces
  write_force<AT>(fxij, fyij, fzij,    i, stride, force);
  write_force<AT>(-fxij, -fyij, -fzij, j, stride, force);
  
  // Store shifted forces
  if (calc_virial) {
    //sforce(is)   = sforce(is)   + fijx
    //sforce(is+1) = sforce(is+1) + fijy
    //sforce(is+2) = sforce(is+2) + fijz
  }

}

//
// 1-4 exclusion force
//
template <typename AT, typename CT, int elec_model, bool calc_energy, bool calc_virial>
__device__ void calc_ex14_force_device(const int pos, const xx14list_t* ex14list,
				       const float4* xyzq, const int stride, AT *force,
				       double &elec_pot) {

  int i = ex14list[pos].i;
  int j = ex14list[pos].j;
  int ish = ex14list[pos].ishift;
  float3 sh_xyz = calc_box_shift(ish, d_setup.boxx, d_setup.boxy, d_setup.boxz);
  // Load atom coordinates
  float4 xyzqi = xyzq[i];
  float4 xyzqj = xyzq[j];
  // Calculate distance
  CT dx = xyzqi.x - xyzqj.x + sh_xyz.x;
  CT dy = xyzqi.y - xyzqj.y + sh_xyz.y;
  CT dz = xyzqi.z - xyzqj.z + sh_xyz.z;
  CT r2 = dx*dx + dy*dy + dz*dz;
  CT qq = ccelec*xyzqi.w*xyzqj.w;
  // Calculate the interaction
  CT r = sqrtf(r2);
  CT rinv = ((CT)1)/r;
  CT rinv2 = rinv*rinv;
  float dpot_elec;
  CT fij_elec = pair_elec_force_14<elec_model, calc_energy>(r2, r, rinv, qq,
							    0.0f, dpot_elec);
  if (calc_energy) elec_pot += (double)dpot_elec;
  CT fij = fij_elec*rinv2;
  // Calculate force components
  AT fxij, fyij, fzij;
  calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);

  // Store forces
  write_force<AT>(fxij, fyij, fzij,    i, stride, force);
  write_force<AT>(-fxij, -fyij, -fzij, j, stride, force);
  // Store shifted forces
  if (calc_virial) {
    //sforce(is)   = sforce(is)   + fijx
    //sforce(is+1) = sforce(is+1) + fijy
    //sforce(is+2) = sforce(is+2) + fijz
  }

}

//
// 1-4 exclusion and interaction calculation kernel
//
template <typename AT, typename CT, int vdw_model, int elec_model, 
	  bool calc_energy, bool calc_virial, bool tex_vdwparam>
__global__ void calc_14_force_kernel(
#ifdef USE_TEXTURE_OBJECTS
				     const cudaTextureObject_t tex,
#endif
				     const int nin14list, const int nex14list,
				     const int nin14block,
				     const xx14list_t* in14list, const xx14list_t* ex14list,
				     const int* vdwtype, const float* vdwparam14,
				     const float4* xyzq, const int stride, AT *force) {
  // Amount of shared memory required:
  // blockDim.x*sizeof(double2)
  extern __shared__ double2 shpot[];

  if (blockIdx.x < nin14block) {
    double vdw_pot, elec_pot;
    if (calc_energy) {
      vdw_pot = 0.0;
      elec_pot = 0.0;
    }

    int pos = threadIdx.x + blockIdx.x*blockDim.x;
    if (pos < nin14list) {
      calc_in14_force_device<AT, CT, vdw_model, elec_model, calc_energy, calc_virial, tex_vdwparam>
	(
#ifdef USE_TEXTURE_OBJECTS
	 tex,
#endif
	 pos, in14list, vdwtype, vdwparam14, xyzq, stride, force, vdw_pot, elec_pot);
    }

    if (calc_energy) {
      shpot[threadIdx.x].x = vdw_pot;
      shpot[threadIdx.x].y = elec_pot;
      __syncthreads();
      for (int i=1;i < blockDim.x;i *= 2) {
	int t = threadIdx.x + i;
	double val1 = (t < blockDim.x) ? shpot[t].x : 0.0;
	double val2 = (t < blockDim.x) ? shpot[t].y : 0.0;
	__syncthreads();
	shpot[threadIdx.x].x += val1;
	shpot[threadIdx.x].y += val2;
	__syncthreads();
      }
      if (threadIdx.x == 0) {
	atomicAdd(&d_energy_virial.energy_vdw,  shpot[0].x);
	atomicAdd(&d_energy_virial.energy_elec, shpot[0].y);
      }
    }

  } else {
    double excl_pot;
    if (calc_energy) excl_pot = 0.0;

    int pos = threadIdx.x + (blockIdx.x-nin14block)*blockDim.x;
    if (pos < nex14list) {
      calc_ex14_force_device<AT, CT, elec_model, calc_energy, calc_virial>
	(pos, ex14list, xyzq, stride, force, excl_pot);
    }

    if (calc_energy) {
      shpot[threadIdx.x].x = excl_pot;
      __syncthreads();
      for (int i=1;i < blockDim.x;i *= 2) {
	int t = threadIdx.x + i;
	double val = (t < blockDim.x) ? shpot[t].x : 0.0;
	__syncthreads();
	shpot[threadIdx.x].x += val;
	__syncthreads();
      }
      if (threadIdx.x == 0) {
	atomicAdd(&d_energy_virial.energy_excl,  shpot[0].x);
      }
    }

  }

}

//
// Nonbonded force kernel
//
template <typename AT, typename CT, int tilesize, int vdw_model, int elec_model,
	  bool calc_energy, bool calc_virial, bool tex_vdwparam>
__global__ void calc_force_kernel(
#ifdef USE_TEXTURE_OBJECTS
				  const cudaTextureObject_t tex,
#endif
				  const int base,
				  const int n_ientry, const ientry_t* __restrict__ ientry,
				  const int* __restrict__ tile_indj,
				  const tile_excl_t<tilesize>* __restrict__ tile_excl,
				  const int stride,
				  const float* __restrict__ vdwparam, const int nvdwparam,
				  const float4* __restrict__ xyzq, const int* __restrict__ vdwtype,
#ifdef USE_BLOCK
				  const int numBlock,
				  const float* __restrict__ bixlam,
				  const int* __restrict__ blocktype,
				  AT* __restrict__ biflam,
				  AT* __restrict__ biflam2,
#ifdef USE_TEXTURE_OBJECTS
				  const cudaTextureObject_t block_tex,
#endif
#endif
				  AT* __restrict__ force) {

  // Pre-computed constants
  const int num_excl = ((tilesize*tilesize-1)/32 + 1);
  const int num_thread_per_excl = (32/num_excl);

  //
  // Shared data, common for the entire block
  //
  extern __shared__ char shmem[];
  
  //const unsigned int sh_start = tilesize*threadIdx.y;

  // Warp index (0...warpsize-1)
  const int wid = threadIdx.x % warpsize;

  // Load index (0...15 or 0...31)
  const int lid = (tilesize == 16) ? (wid % tilesize) : wid;

  int shmem_pos = 0;
  //
  // Shared memory requirements:
  // sh_xi, sh_yi, sh_zi, sh_qi: (blockDim.x/warpsize)*tilesize*sizeof(float)
  // sh_vdwtypei               : (blockDim.x/warpsize)*tilesize*sizeof(int)
  // sh_blocktypei               : (blockDim.x/warpsize)*tilesize*sizeof(int)
  // sh_fix, sh_fiy, sh_fiz    : (blockDim.x/warpsize)*warpsize*sizeof(AT)
  // sh_vdwparam               : nvdwparam*sizeof(float)
  //
  // ## For USE_BLOCK ##
  // sh_blocktypei             : (blockDim.x/warpsize)*tilesize*sizeof(int)
  // sh_bixlam                 : numBlock*sizeof(float)
  //
  // (x_i, y_i, z_i, q_i, vdwtype_i) are private to each warp
  // (fix, fiy, fiz) are private for each warp
  // vdwparam_sh is for the entire thread block
#if __CUDA_ARCH__ < 300
  float *sh_xi = (float *)&shmem[shmem_pos + (threadIdx.x/warpsize)*tilesize*sizeof(float)];
  shmem_pos += (blockDim.x/warpsize)*tilesize*sizeof(float);
  float *sh_yi = (float *)&shmem[shmem_pos + (threadIdx.x/warpsize)*tilesize*sizeof(float)];
  shmem_pos += (blockDim.x/warpsize)*tilesize*sizeof(float);
  float *sh_zi = (float *)&shmem[shmem_pos + (threadIdx.x/warpsize)*tilesize*sizeof(float)];
  shmem_pos += (blockDim.x/warpsize)*tilesize*sizeof(float);
  float *sh_qi = (float *)&shmem[shmem_pos + (threadIdx.x/warpsize)*tilesize*sizeof(float)];
  shmem_pos += (blockDim.x/warpsize)*tilesize*sizeof(float);
  int *sh_vdwtypei = (int *)&shmem[shmem_pos + (threadIdx.x/warpsize)*tilesize*sizeof(int)];
  shmem_pos += (blockDim.x/warpsize)*tilesize*sizeof(int);
#ifdef USE_BLOCK
  int *sh_blocktypei = (int *)&shmem[shmem_pos + (threadIdx.x/warpsize)*tilesize*sizeof(int)];
  shmem_pos += (blockDim.x/warpsize)*tilesize*sizeof(int);
#endif
#endif

#ifdef USE_BLOCK
#ifndef NUMBLOCK_LARGE
  // For large numBlock values, to conserve shared memory, we don't store bixlam into shared memory.
  int *sh_bixlam = (int *)&shmem[shmem_pos];
  shmem_pos += numBlock*sizeof(float);
#endif
#endif
  
  volatile AT *sh_fix = (AT *)&shmem[shmem_pos + (threadIdx.x/warpsize)*warpsize*sizeof(AT)];
  shmem_pos += (blockDim.x/warpsize)*warpsize*sizeof(AT);
  volatile AT *sh_fiy = (AT *)&shmem[shmem_pos + (threadIdx.x/warpsize)*warpsize*sizeof(AT)];
  shmem_pos += (blockDim.x/warpsize)*warpsize*sizeof(AT);
  volatile AT *sh_fiz = (AT *)&shmem[shmem_pos + (threadIdx.x/warpsize)*warpsize*sizeof(AT)];
  shmem_pos += (blockDim.x/warpsize)*warpsize*sizeof(AT);

  float *sh_vdwparam;
  if (!tex_vdwparam) {
    sh_vdwparam = (float *)&shmem[shmem_pos];
    shmem_pos += nvdwparam*sizeof(float);
  }

  // Load ientry. Single warp takes care of one ientry
  const int ientry_ind = (threadIdx.x + blockDim.x*blockIdx.x)/warpsize + base;

  int indi, ish, startj, endj;
  if (ientry_ind < n_ientry) {
    indi   = ientry[ientry_ind].indi;
    ish    = ientry[ientry_ind].ish;
    startj = ientry[ientry_ind].startj;
    endj   = ientry[ientry_ind].endj;
  } else {
    indi = 0;
    ish  = 1;
    startj = 1;
    endj = 0;
  }

  // Calculate shift for i-atom
  // ish = 0...26
  int ish_tmp = ish;
  float shz = (ish_tmp/9 - 1)*d_setup.boxz;
  ish_tmp -= (ish_tmp/9)*9;
  float shy = (ish_tmp/3 - 1)*d_setup.boxy;
  ish_tmp -= (ish_tmp/3)*3;
  float shx = (ish_tmp - 1)*d_setup.boxx;

  // Load i-atom data to shared memory (and shift coordinates)
  float4 xyzq_tmp = xyzq[indi + lid];
#if __CUDA_ARCH__ >= 300
  float xi = xyzq_tmp.x + shx;
  float yi = xyzq_tmp.y + shy;
  float zi = xyzq_tmp.z + shz;
  float qi = xyzq_tmp.w*ccelec;
  int vdwtypei = vdwtype[indi + lid];
#ifdef USE_BLOCK
  int blocktypei = blocktype[indi + lid];
#endif
#else
  sh_xi[lid] = xyzq_tmp.x + shx;
  sh_yi[lid] = xyzq_tmp.y + shy;
  sh_zi[lid] = xyzq_tmp.z + shz;
  sh_qi[lid] = xyzq_tmp.w*ccelec;
  sh_vdwtypei[lid] = vdwtype[indi + lid];
#ifdef USE_BLOCK
  sh_blocktypei[lid] = blocktype[indi + lid];
#endif
#endif

  sh_fix[wid] = (AT)0;
  sh_fiy[wid] = (AT)0;
  sh_fiz[wid] = (AT)0;

  if (!tex_vdwparam) {
    // Copy vdwparam to shared memory
    for (int i=threadIdx.x;i < nvdwparam;i+=blockDim.x)
      sh_vdwparam[i] = vdwparam[i];
    __syncthreads();
  }

#ifdef USE_BLOCK
#ifndef NUMBLOCK_LARGE
    // Copy bixlam to shared memory
    for (int i=threadIdx.x;i < numBlock;i+=blockDim.x)
      sh_bixlam[i] = bixlam[i];
    __syncthreads();
  }
#endif
#endif
  
  double vdwpotl;
  double coulpotl;
  if (calc_energy) {
    vdwpotl = 0.0;
    coulpotl = 0.0;
  }

  for (int jtile=startj;jtile <= endj;jtile++) {

    // Load j-atom starting index and exclusion mask
    unsigned int excl;
    if (tilesize == 16) {
      // For 16x16 tile, the exclusion mask per is 8 bits per thread:
      // NUM_THREAD_PER_EXCL = 4
      excl = tile_excl[jtile].excl[wid/num_thread_per_excl] >> 
	((wid % num_thread_per_excl)*num_excl);
    } else {
      excl = tile_excl[jtile].excl[wid];
    }
    int indj = tile_indj[jtile];

    // Skip empty tile
    if (__all(~excl == 0)) continue;

    float4 xyzq_j = xyzq[indj + lid];
    int ja = vdwtype[indj + lid];
#ifdef USE_BLOCK
    int jb = blocktype[indj + lid];
#endif

    // Clear j forces
    AT fjx = (AT)0;
    AT fjy = (AT)0;
    AT fjz = (AT)0;

    for (int t=0;t < num_excl;t++) {
      
      int ii;
      if (tilesize == 16) {
	ii = (wid + t*2 + (wid/tilesize)*(tilesize-1)) % tilesize;
      } else {
	ii = ((wid + t) % tilesize);
      }

#if __CUDA_ARCH__ >= 300
      float dx = __shfl(xi, ii) - xyzq_j.x;
      float dy = __shfl(yi, ii) - xyzq_j.y;
      float dz = __shfl(zi, ii) - xyzq_j.z;
#else
      float dx = sh_xi[ii] - xyzq_j.x;
      float dy = sh_yi[ii] - xyzq_j.y;
      float dz = sh_zi[ii] - xyzq_j.z;
#endif
	
      float r2 = dx*dx + dy*dy + dz*dz;

#if __CUDA_ARCH__ >= 300
      float qq = __shfl(qi, ii)*xyzq_j.w;
#else
      float qq = sh_qi[ii]*xyzq_j.w;
#endif

#if __CUDA_ARCH__ >= 300
      int ia = __shfl(vdwtypei, ii);
#else
      int ia = sh_vdwtypei[ii];
#endif

#ifdef USE_BLOCK
#if __CUDA_ARCH__ >= 300
      int ib = __shfl(blocktypei, ii);
#else
      int ib = sh_blocktypei[ii];
#endif
#endif

      if (!(excl & 1) && r2 < d_setup.roff2) {

	float rinv = rsqrtf(r2);
	float r = r2*rinv;
	
	float dpot_elec;
	float fij_elec = pair_elec_force<elec_model, calc_energy>(r2, r, rinv, qq, dpot_elec);
#ifndef USE_BLOCK
	if (calc_energy) coulpotl += (double)dpot_elec;
#endif
	int aa = (ja > ia) ? ja : ia;      // aa = max(ja,ia)
	
	float c6, c12;
	if (tex_vdwparam) {
	  int ivdw = (aa*(aa-3) + 2*(ja + ia) - 2) >> 1;
	  //c6 = __ldg(&vdwparam[ivdw]);
	  //c12 = __ldg(&vdwparam[ivdw+1]);
#ifdef USE_TEXTURE_OBJECTS
	  float2 c6c12 = tex1Dfetch<float2>(tex, ivdw);
#else
	  float2 c6c12 = tex1Dfetch(VDWPARAM_TEXREF, ivdw);
#endif
	  c6  = c6c12.x;
	  c12 = c6c12.y;
	} else {
	  int ivdw = (aa*(aa-3) + 2*(ja + ia) - 2);
	  c6 = sh_vdwparam[ivdw];
	  c12 = sh_vdwparam[ivdw+1];
	}
	
	float rinv2 = rinv*rinv;
	float dpot_vdw;
	float fij_vdw = pair_vdw_force<vdw_model, calc_energy>(r2, r, rinv, rinv2,
							       c6, c12, dpot_vdw);
#ifndef USE_BLOCK
	if (calc_energy) vdwpotl += (double)dpot_vdw;
#endif
	
	float fij = (fij_vdw - fij_elec)*rinv*rinv;

#ifdef USE_BLOCK
	// ib: highest 16 bits is the site number (isitemld), lowest 16 bits is the block number
	int ib_site = ib & 0xffff0000;
	int jb_site = jb & 0xffff0000;
	ib &= 0xffff;
	jb &= 0xffff;
	int bb = (jb > ib) ? jb : ib;      // bb = max(jb,ib)
	int iblock = (bb*(bb-3) + 2*(jb + ib) - 2);
#ifdef USE_TEXTURE_OBJECTS
	float scale = tex1Dfetch<float>(block_tex, iblock);
#else
	float scale = tex1Dfetch(blockparam_texref, iblock);
#endif
	fij *= scale;
	if (calc_energy) {
	  if (scale != 1.0f && scale != 0.0f) {
	    float dpot = (dpot_elec + dpot_vdw)*FORCE_SCALE_VIR;
	    int ibb = (ib == jb) ? ib : ( ib == 0 ? jb : (jb == 0 ? ib : -1) );

	    //float dpot_scale = 1.0f;
	    //if (ibb < 0) {
	    //  dpot_scale = bixlam[ib];
	    //  ibb += 
	    //}
	    //biflam[ibb] += (double)dpot;
	    
	    if (ibb >= 0) {
	      //biflam[ibb] += (double)dpot;
	      AT dpotAT = lliroundf(dpot);
	      atomicAdd((unsigned long long int *)&biflam[ibb], llitoulli(dpotAT));
	    } else if (ib_site != jb_site) {
#ifdef NUMBLOCK_LARGE
	      AT dpotiAT = lliroundf(bixlam[ib]*dpot);
	      AT dpotjAT = lliroundf(bixlam[jb]*dpot);
#else
	      AT dpotiAT = lliroundf(sh_bixlam[ib]*dpot);
	      AT dpotjAT = lliroundf(sh_bixlam[jb]*dpot);
#endif
	      atomicAdd((unsigned long long int *)&biflam2[ib], llitoulli(dpotiAT));
	      atomicAdd((unsigned long long int *)&biflam2[jb], llitoulli(dpotjAT));
	    }

	    	  /*
      if (ibl.eq.jbl) then
         biflam(ibl) = biflam(ibl) + energy
      elseif (ibl.eq.1) then
         biflam(jbl) = biflam(jbl) + energy
      elseif (jbl.eq.1) then
         biflam(ibl) = biflam(ibl) + energy
      elseif (isitemld(ibl).ne.isitemld(jbl)) then
         biflam2(jbl) = biflam2(jbl) + bixlam(ibl)*energy
         biflam2(ibl) = biflam2(ibl) + bixlam(jbl)*energy
      endif
	  */

	  }
	  
	  coulpotl += (double)(dpot_elec*scale);
	  vdwpotl += (double)(dpot_vdw*scale);
	}
#endif

	AT fxij;
	AT fyij;
	AT fzij;
	calc_component_force<AT, CT>(fij, dx, dy, dz, fxij, fyij, fzij);
	
	fjx -= fxij;
	fjy -= fyij;
	fjz -= fzij;
	
	if (tilesize == 16) {
	  // We need to re-calculate ii because ii must be warp sized in order to
	  // prevent race condition
	  int tmp = (wid + t*2) % 16 + (wid/16)*31;
	  ii = tilesize*(threadIdx.x/warpsize)*2 + (tmp + (tmp/32)*16) % 32;
	}
	
	sh_fix[ii] += fxij;
	sh_fiy[ii] += fyij;
	sh_fiz[ii] += fzij;
      } // if (!(excl & 1) && r2 < d_setup.roff2)

      // Advance exclusion mask
      excl >>= 1;
    }

    // Dump register forces (fjx, fjy, fjz)
    write_force<AT>(fjx, fjy, fjz, indj + lid, stride, force);
  }

  // Dump shared memory force (fi)
  // NOTE: no __syncthreads() required here because sh_fix is "volatile"
  write_force<AT>(sh_fix[wid], sh_fiy[wid], sh_fiz[wid], indi + lid, stride, force);

  if (calc_virial) {
    // Virial is calculated from (sh_fix[], sh_fiy[], sh_fiz[])
    // Variable "ish" depends on warp => Reduce within warp
    // NOTE: we skip the center element because it doesn't contribute to the virial
    if (ish != 13) {
      // Convert into double
      volatile double *sh_sfix = (double *)sh_fix;
      volatile double *sh_sfiy = (double *)sh_fiy;
      volatile double *sh_sfiz = (double *)sh_fiz;

      sh_sfix[wid] = ((double)sh_fix[wid])*INV_FORCE_SCALE;
      sh_sfiy[wid] = ((double)sh_fiy[wid])*INV_FORCE_SCALE;
      sh_sfiz[wid] = ((double)sh_fiz[wid])*INV_FORCE_SCALE;

      for (int d=16;d >= 1;d/=2) {
	if (wid < d) {
	  sh_sfix[wid] += sh_sfix[wid + d];
	  sh_sfiy[wid] += sh_sfiy[wid + d];
	  sh_sfiz[wid] += sh_sfiz[wid + d];
	}
      }
      if (wid == 0) {
	atomicAdd(&d_energy_virial.sforcex[ish], sh_sfix[0]);
	atomicAdd(&d_energy_virial.sforcey[ish], sh_sfiy[0]);
	atomicAdd(&d_energy_virial.sforcez[ish], sh_sfiz[0]);
      }
    }
  }

  if (calc_energy) {
    // Reduce energies across the entire thread block
    // Shared memory required:
    // blockDim.x*sizeof(double)*2
    __syncthreads();
    double2* sh_pot = (double2 *)(shmem);
    sh_pot[threadIdx.x].x = vdwpotl;
    sh_pot[threadIdx.x].y = coulpotl;
    __syncthreads();
    for (int i=1;i < blockDim.x;i *= 2) {
      int pos = threadIdx.x + i;
      double vdwpot_val  = (pos < blockDim.x) ? sh_pot[pos].x : 0.0;
      double coulpot_val = (pos < blockDim.x) ? sh_pot[pos].y : 0.0;
      __syncthreads();
      sh_pot[threadIdx.x].x += vdwpot_val;
      sh_pot[threadIdx.x].y += coulpot_val;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicAdd(&d_energy_virial.energy_vdw,  sh_pot[0].x);
      atomicAdd(&d_energy_virial.energy_elec, sh_pot[0].y);
    }
  }

}
