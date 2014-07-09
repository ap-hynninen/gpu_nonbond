#ifndef LEAPFROGINTEGRATOR_H
#define LEAPFROGINTEGRATOR_H

#include <stdio.h>
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
  virtual void do_print_energy(int step)=0;
  virtual void get_restart_data(double *x, double *y, double *z,
				double *dx, double *dy, double *dz,
				double *fx, double *fy, double *fz)=0;

protected:

  // timestep in fs (10e-15 s)
  double timestep;

  // Forcefield
  Forcefield *forcefield;

  // Host buffers for saving integrator status
  // NOTE: Must be size of the global array length
  double *x;
  double *y;
  double *z;

  double *dx;
  double *dy;
  double *dz;

  double *fx;
  double *fy;
  double *fz;

public:

  //
  // Class creator
  //
  LeapfrogIntegrator() {
    forcefield = NULL;
    x = NULL;
    y = NULL;
    z = NULL;
    dx = NULL;
    dy = NULL;
    dz = NULL;
    fx = NULL;
    fy = NULL;
    fz = NULL;
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
  // Set host coordinate buffers
  //
  void set_coord_buffers(double *x_in, double *y_in, double *z_in) {
    x = x_in;
    y = y_in;
    z = z_in;
  }

  //
  // Set host step buffers
  //
  void set_step_buffers(double *x_in, double *y_in, double *z_in) {
    dx = x_in;
    dy = y_in;
    dz = z_in;
  }

  //
  // Set host force buffers
  //
  void set_force_buffers(double *x_in, double *y_in, double *z_in) {
    fx = x_in;
    fy = y_in;
    fz = z_in;
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
  void run(const int nstep, const int print_freq, const int restart_freq) {

    printf("DYNA DYN: Step         Time      TOTEner        TOTKe       ENERgy  TEMPerature\n");
    printf("DYNA PROP:             GRMS      HFCTote        HFCKe       EHFCor        VIRKe\n");
    printf("DYNA INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals    IMPRopers\n");
    printf("DYNA EXTERN:        VDWaals         ELEC       HBONds          ASP         USER\n");
    printf("DYNA EWALD:          EWKSum       EWSElf       EWEXcl       EWQCor       EWUTil\n");
    printf("DYNA PRESS:            VIRE         VIRI       PRESSE       PRESSI       VOLUme\n");
    printf("----------       ---------    ---------    ---------    ---------    ---------\n");

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
	do_print_energy(istep);
      }

      if ((istep % restart_freq) == 0) {
	get_restart_data(x, y, z, dx, dy, dz, fx, fy, fz);
      }

      // Swap: dx <=> dx_prev
      swap_step();
      
    }

  }

};

#endif // LEAPFROGINTEGRATOR_H
