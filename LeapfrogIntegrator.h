#ifndef LEAPFROGINTEGRATOR_H
#define LEAPFROGINTEGRATOR_H

#include "Forcefield.h"

//
// Leap frog Verlet integrator base class
//

class LeapfrogIntegrator {

  //friend class LangevinPiston;

private:

  virtual void swap_step()=0;
  virtual void take_step()=0;
  virtual void calc_step()=0;
  virtual void calc_force(const bool calc_energy, const bool calc_virial)=0;
  virtual void do_holoconst()=0;
  virtual void do_pressure()=0;
  virtual void do_temperature()=0;
  virtual bool const_pressure()=0;
  virtual void do_print_energy()=0;

protected:

  // timestep in ps (10e-12 s)
  double timestep;

  // Forcefield
  Forcefield *forcefield;

public:

  //
  // Class creator
  //
  LeapfrogIntegrator() {
    forcefield = NULL;
  }

  //
  // Class destructor
  //
  ~LeapfrogIntegrator() {
  }

  //
  // Sets forcefield value
  //
  void set_forcefield(Forcefield *forcefield) {
    this->forcefield = forcefield;
  }

  //
  // Sets timestep
  //
  void set_timestep(const double timestep) {
    this->timestep = timestep;
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
  void run(const int nstep, const int print_freq) {

    for (int istep=0;istep < nstep;istep++) {

      // Take a step: x = x_prev + dx_prev
      take_step();
      
      // Pressure scaling (if needed)
      do_pressure();

      bool print_energy = (istep % print_freq) == 0;

      // Calculate forces
      // NOTE: If applicable, does neighbor list search
      bool calc_energy = print_energy;
      bool calc_virial = const_pressure() || print_energy;
      calc_force(calc_energy, calc_virial);

      // Calculate step vector: dx = dx_prev - fx*dt*dt/mass
      calc_step();
      
      // Do holonomic constraints (if necessary)
      do_holoconst();
      
      // Constant temperature (if needed)
      do_temperature();
      
      // Calculate temperature
      
      // Calculate RMS gradient and work
      
      // Calculate pressure

      // Print energies & other values on screen
      if (print_energy) {
	do_print_energy();
      }
      
      // Swap: dx <=> dx_prev
      swap_step();
      
    }

  }

};

#endif // LEAPFROGINTEGRATOR_H
