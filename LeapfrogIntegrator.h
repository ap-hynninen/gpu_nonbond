#ifndef LEAPFROGINTEGRATOR_H
#define LEAPFROGINTEGRATOR_H

//
// Leap frog Verlet integrator base class
//

class LeapfrogIntegrator {

  //friend class LangevinPiston;

private:

  // Settings
  bool const_temperature;
  bool const_pressure;
  bool holoconst;

  virtual void swap_step()=0;
  virtual void take_step()=0;
  virtual void calc_step()=0;
  virtual void calc_force(const bool calc_energy, const bool calc_virial)=0;

protected:

  // timestep in ps (10e-12 s)
  double timestep;

public:

  //
  // Class creator
  //
  LeapfrogIntegrator() {
  }

  //
  // Class destructor
  //
  ~LeapfrogIntegrator() {
  }

  //
  // Initialize
  //
  virtual void init(const int ncoord,
		    const double *x, const double *y, const double *z,
		    const double *dx, const double *dy, const double *dz)=0;

  //
  // Runs dynamics for nstep steps
  //
  void run(const int nstep) {

    for (int istep=0;istep < nstep;istep++) {
      
      // Take a step: x = x_prev + dx
      take_step();
      
      // Pressure scaling
      if (const_pressure) {
      }
      
      // Calculate forces
      // NOTE: If applicable, does neighbor list search
      bool calc_energy = false;
      bool calc_virial = const_pressure;
      calc_force(calc_energy, calc_virial);
      
      // Calculate step vector: dx = dx_prev - fx*dt*dt/mass
      calc_step();
      
      // Do holonomic constraints
      if (holoconst) {
      }
      
      // Constant temperature
      if (const_temperature) {
      }
      
      // Calculate temperature
      
      // Calculate RMS gradient and work
      
      // Calculate pressure
      
      // Swap: dx <=> dx_prev
      swap_step();
      
    }

  }

};

#endif // LEAPFROGINTEGRATOR_H
