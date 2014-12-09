//
// Base class for Langevin Piston
// (c) Antti-Pekka Hynninen
//
// NOTE: Much of the code is copied and modified from CHARMM source/dynamc/dynutil.src
//
// Original reference: S. E. Feller et al., J. Chem. Phys. 103 (11) p.4613 (1995)
//
#include "Domdec.h"

class LangevinPiston {
 private:
  Domdec& domdec;

  double delp[3][3];
  double delpr[3][3];

  virtual void calcDelpLocal(double )=0;
  
 public:
  LangevinPiston();
  ~LangevinPiston();

  applyPressure();
};
