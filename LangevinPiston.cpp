#include "LangevinPiston.h"


//
// Class creator
//
LangevinPiston::LangevinPiston() {
}

//
// Class destructor
//
LangevinPiston::~LangevinPiston() {
}

//
// Apply pressure
//
void LangevinPiston::applyPressure() {

  // Calculate the volume at step n-1
  double volume = domdec.get_boxx()*domdec.get_boxy()*domdec.get_boxz();

  double REFKE = 0.5*NDEGF*KBOLTZ*REFT;
  int NATOM3=NATOM*3;
  double DELTA2=DELTA*DELTA;
  double HALFD=0.5/DELTA;
  double VCELL = 0.25*PATMOS/VOLUME/DELTA2;
  // SFACT converts from dyne/cm to Atm*Angstroms
  double SFACT = 98.6923;
  
  // Copy the thermal piston velocity and acceleration at previous step
  double PNHVP = PNHV;
  double PNHFP = PNHF;

  double SURFI = 0.5*XTLOLD(6)*(EPRESS(PIZZ)-0.5*(
						  EPRESS(PIYY)+EPRESS(PIXX)))/SFACT;

  /*
    !
    IF(QCONZ) THEN
       REFP(RPZZ) = EPRESS(PIZZ)
    ENDIF
    !
    IF(QSURF) THEN
       REFP(RPXX) = REFP(RPZZ) - SFACT*SURFT/XTLABC(6)
       REFP(RPYY) = REFP(RPXX)
    ENDIF
    !
    RVAL = ZERO
    !
  */
  // Compute pressure difference matrix.

  double delpLocal[3][3];
  calcDelpLocal(delpLocal);

  /*
  !                put h at n-1 into a holding array 
    DO I=1,6
       XTLOLD(I) = XTLABC(I)
       HDOLD(I) = HDOT(I)
    ENDDO
  */

  //
  // This loop does iterations to ensure than the corrected velocity is
  // in agreement with the pressure.  For starters we try 3 iterations.
  // This should be replaced with a tolerance criterion.
  //
  for (int jit=1;jit <= PITER;jit++) {
       DO I=1,6
          XTLABC(I) = XTLOLD(I)
          HDOT(I) = HDOLD(I)
       ENDDO
       !
       ! Compute inverse of h matrix.
       ! We have h at step n-1, this gives h inverse at step n-1
       !
       CALL INVT33S(XTLINV,XTLABC,OK)
       CALL MULNXNFL(WRK1,DELP,XTLINV,3)
       CALL LATTRN(XTLTYP,WRK1,DELPX,XTLREF)
       !
       ! Update the HDOT matrix.
       ! This sets hdotp to hdot at step n-3/2, hdoti is hdot at n-1/2
       ! The force on the piston comes from P at step n-1
       !
       FACT=EPROP(VOLUME)*PBFACT*ATMOSP
       !
       DO I=1,XDIM
          HDOTP(I)=HDOT(I)
          HDOT(I) = PALPHA*HDOT(I) +  &
               PWINV(I)*DELPX(I)*FACT + &
               PBFACT*BMGAUS(PRFWD(I),ISEED)
       ENDDO
       !
       !     Make sure that every process has exactly the same value
#if KEY_PARALLEL==1
       CALL PSND8(HDOT,XDIM)                        
#endif
       !
       DO I=1,XDIM
          HDAV(I) = (HDOTP(I)+HDOT(I))*PVFACT*DELTA
       ENDDO
       !
       ! Propogate the h matrix
       ! This calculates h at step n, using the h velocity we just found
       CALL GETXTL(SF,XTLABC,XTLTYP,XTLREF)
       !
       DO I=1,XDIM
          SF(I)=SF(I)+HDOT(I)
       ENDDO
       !
       !     Make sure that every process has exactly the same value
#if KEY_PARALLEL==1
       CALL PSND8(SF,XDIM)                          
#endif
       !
       DO I=1,XDIM
          SH(I)=SF(I)-HALF*HDOT(I)
       ENDDO
       !
       IF(QCONZ) THEN
          SF(XDIM) = XTLOLD(1)*XTLOLD(3)*XTLOLD(6)/SF(1)**2 
       ENDIF
       !
       ! warning: the isotropic and constant volume options
       ! work for tetragonal and orhtorombic only
       !
       !     this puts the new hvalues into the xtlabc matrix
       CALL SETXTL(HDAV,AVABC,XTLTYP,XTLREF)
       CALL MULNXNLL(WRK2,AVABC,XTLINV,3)
       !
       CALL SETXTL(HDOT,HDABC,XTLTYP,XTLREF)
       CALL PUTXTL(SH,XTLABC,XTLTYP,XTLREF)
       CALL INVT33S(XTLINV,XTLABC,OK)
       CALL MULNXNLL(WRK3,HDABC,XTLINV,3)
       CALL PUTXTL(SF,XTLABC,XTLTYP,XTLREF)
       !
       ! Calculate the thermal piston velocity and position
       IF(QNPT) THEN
          RVAL=HALF*RVAL
          PNHF = TWO*DELTA*(RVAL-REFKE)/TMASS
          IF(PNHFP == ZERO) THEN
             PNHFP = PNHF
          ENDIF
          PNHV = PNHVP + HALF*(PNHF + PNHFP)
          !     Make sure all processes have the same value
#if KEY_PARALLEL==1
          CALL PSND8(PNHV,1)                          
#endif
          !
          RVAL = ZERO
       ENDIF
       !
       !=======================================================================
       ! Scale the coordinates and velocities.
       !
       IF(.NOT.PTYPE) THEN
          ! Calculate corrected pressure difference for next iteration
          DELP(1,1) = ZERO
          DELP(1,2) = ZERO
          DELP(1,3) = ZERO
          DELP(2,1) = ZERO
          DELP(2,2) = ZERO
          DELP(2,3) = ZERO
          DELP(3,1) = ZERO
          DELP(3,2) = ZERO
          DELP(3,3) = ZERO

#if KEY_DOMDEC==1
          if (q_domdec) then
             i00 = 1
             i01 = natoml
          else
#endif 
             i00=1
             i01=natom
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
             I00 = 1 + IPARPT(MYNOD)
             i01 = IPARPT(MYNODP)
#endif 
#endif 
#if KEY_DOMDEC==1
          endif  
#endif

!$omp parallel do schedule(static) &
!$omp& private(ia, i, vxf, vyf, vzf, sfxx, sfyy, sfzz, svxx, svyy, svzz, rvc) &
!$omp& reduction(+:rval, delp)
          do ia=i00,i01
#if KEY_DOMDEC==1 /*domdec*/
             if (q_domdec) then
                i = atoml(ia)
             else
#endif /* (domdec)*/
                i = ia
#if KEY_DOMDEC==1
             endif  
#endif
#if KEY_PARALLEL==1
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) /= MYNOD) cycle
#endif 
#endif 
             ! Modify the forward half step velocity by -(v+f)*hdot/h
             !
             VXF = (XOLD(I) + VX(I)) * HALFD
             VYF = (YOLD(I) + VY(I)) * HALFD
             VZF = (ZOLD(I) + VZ(I)) * HALFD
             !
             SFXX = WRK2(1,1)*VXF &
                  + WRK2(1,2)*VYF &
                  + WRK2(1,3)*VZF
             SFYY = WRK2(2,1)*VXF &
                  + WRK2(2,2)*VYF &
                  + WRK2(2,3)*VZF
             SFZZ = WRK2(3,1)*VXF &
                  + WRK2(3,2)*VYF &
                  + WRK2(3,3)*VZF
             !
             IF(QCONZ) THEN
                SFZZ = -( WRK2(1,1) + WRK2(2,2) )*VZF
             ENDIF
             !
             VX(I) = XNEW(I) - DELTA*SFXX 
             VY(I) = YNEW(I) - DELTA*SFYY
             VZ(I) = ZNEW(I) - DELTA*SFZZ 
             !
             IF(QNPT) THEN
                VX(I) = VX(I) - DELTA2*VXF*PNHV
                VY(I) = VY(I) - DELTA2*VYF*PNHV
                VZ(I) = VZ(I) - DELTA2*VZF*PNHV
                !brb                  VXF = VXF - HALF*(SFXX + VXF*PNHV*DELTA)
                !brb                  VYF = VYF - HALF*(SFYY + VYF*PNHV*DELTA)
                !brb                  VZF = VZF - HALF*(SFZZ + VZF*PNHV*DELTA) 
                RVAL = RVAL + AMASS(I)*(VXF**2 + VYF**2 + VZF**2) 
             ENDIF
             !            
             VXF = (XCOMP(I) + X(I)) * HALF
             VYF = (YCOMP(I) + Y(I)) * HALF
             VZF = (ZCOMP(I) + Z(I)) * HALF
             !             
             SVXX = WRK3(1,1)*VXF &
                  + WRK3(1,2)*VYF &
                  + WRK3(1,3)*VZF
             SVYY = WRK3(2,1)*VXF &
                  + WRK3(2,2)*VYF &
                  + WRK3(2,3)*VZF
             SVZZ = WRK3(3,1)*VXF &
                  + WRK3(3,2)*VYF &
                  + WRK3(3,3)*VZF
             !
             IF(QCONZ) THEN
                SVZZ = -(WRK3(1,1) + WRK3(2,2))*VZF
             ENDIF
             !
             X(I) = VX(I) + SVXX + XCOMP(I)
             Y(I) = VY(I) + SVYY + YCOMP(I)
             Z(I) = VZ(I) + SVZZ + ZCOMP(I)
             !
             ! Calculate positive of the velocity contribution to the pressure
             !  to get new corrected pressure
             RVC=VCELL*AMASS(I)
             DELP(1,1)=DELP(1,1) + RVC*VX(I)**2
             DELP(1,2)=DELP(1,2) + RVC*VX(I)*VY(I)
             DELP(1,3)=DELP(1,3) + RVC*VX(I)*VZ(I)
             DELP(2,1)=DELP(2,1) + RVC*VY(I)*VX(I)
             DELP(2,2)=DELP(2,2) + RVC*VY(I)**2
             DELP(2,3)=DELP(2,3) + RVC*VY(I)*VZ(I)
             DELP(3,1)=DELP(3,1) + RVC*VZ(I)*VX(I)
             DELP(3,2)=DELP(3,2) + RVC*VZ(I)*VY(I)
             DELP(3,3)=DELP(3,3) + RVC*VZ(I)**2
          ENDDO
!$omp end parallel do
#if KEY_PARALLEL==1
!!$          CALL GCOMB(DELP,9)
!!$          CALL GCOMB(RVAL,1)
          gcarr(1:3) = delp(1:3,1)
          gcarr(4:6) = delp(1:3,2)
          gcarr(7:9) = delp(1:3,3)
          gcarr(10) = rval
          call gcomb(gcarr, 10)
          delp(1:3,1) = gcarr(1:3)
          delp(1:3,2) = gcarr(4:6)
          delp(1:3,3) = gcarr(7:9)
          rval = gcarr(10)
#endif 
          DELP(1,1) = DELPR(1,1) + DELP(1,1)
          DELP(1,2) = DELPR(1,2) + DELP(1,2)
          DELP(1,3) = DELPR(1,3) + DELP(1,3)
          DELP(2,1) = DELPR(2,1) + DELP(2,1)
          DELP(2,2) = DELPR(2,2) + DELP(2,2)
          DELP(2,3) = DELPR(2,3) + DELP(2,3)
          DELP(3,1) = DELPR(3,1) + DELP(3,1)
          DELP(3,2) = DELPR(3,2) + DELP(3,2)
          DELP(3,3) = DELPR(3,3) + DELP(3,3)
       ENDIF
    ENDDO jit_loop

}


