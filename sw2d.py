#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from clawpack import petclaw as pyclaw
from clawpack import riemann

periodic_bc_time        = 10.
ambient_surface_height  = 0.
pulse_width             = 5

def qinit(state,pulse_amplitude):
    "Gaussian surface perturbation"
    x0=0.
    y0=0.

    b = state.aux[0,:,:] # Bathymetry

    X,Y = state.grid.p_centers
    surface = ambient_surface_height+pulse_amplitude*np.exp(-(X/pulse_width)**2)
    state.q[0,:,:] = surface - b
    state.q[1,:,:] = 0.
    state.q[2,:,:] = 0.
    
def bathymetry(y,bjump=1.0,bathymetry_type='pwc'):
    if bathymetry_type == 'pwc':
        b_A = -1. - bjump/2
        b_B = -1. + bjump/2
        return (y>0)*b_A + (y<=0)*b_B
    elif bathymetry_type == 'sinusoidal':
        #return -1. - bjump/2 + bjump*np.sin(2*np.pi*y)
        return -1. + bjump*np.sin(2*np.pi*y)

def switch_to_periodic_BCs(solver,state):
    from clawpack import pyclaw
    #Change to periodic BCs after initial pulse 
    if state.t>periodic_bc_time and solver.bc_lower[0]==pyclaw.BC.wall:
        solver.bc_lower[0]=pyclaw.BC.periodic
        solver.bc_upper[0]=pyclaw.BC.periodic
        solver.aux_bc_lower[0]=pyclaw.BC.periodic
        solver.aux_bc_upper[0]=pyclaw.BC.periodic
        
def setup(bjump=1.0,cells_per_period=20,tfinal=300,solver_type='classic',
          riemann_solver='fwave',outdir='./_output',btype='pwc',channel_width=0.1,
          num_output_times=200,L=200.,pulse_amplitude=0.05):


    if riemann_solver == 'fwave':
        rs = riemann.shallow_bathymetry_fwave_2D
    elif riemann_solver == 'augmented':
        rs = riemann.sw_aug_2D
        
    if solver_type == 'classic':
        solver = pyclaw.ClawSolver2D(rs)
        solver.limiters = pyclaw.limiters.tvd.minmod
        solver.dimensional_split = True
        solver.cfl_max     = 0.45
        solver.cfl_desired = 0.4
        #solver.limiters = pyclaw.tvd.MC
    elif solver_type == 'sharpclaw':
        solver = pyclaw.SharpClawSolver2D(rs)
        #solver.cfl_max     = 1.95
        #olver.cfl_desired = 1.9

    
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic

    solver.aux_bc_lower[0] = pyclaw.BC.wall
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[1] = pyclaw.BC.periodic
    solver.aux_bc_upper[1] = pyclaw.BC.periodic

    solver.fwave = True
    solver.before_step = switch_to_periodic_BCs 

    # Domain:
    xlower =   0.;  xupper =  L/2.
    ylower = -channel_width/2.;  yupper =  channel_width/2.

    
    mx = (xupper-xlower)*cells_per_period
    my = (yupper-ylower)*cells_per_period

    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    y = pyclaw.Dimension(ylower,yupper,my,name='y')
    domain = pyclaw.Domain([x,y])

    num_aux = 1
    state = pyclaw.State(domain,solver.num_eqn,num_aux)
    state.aux[:,:,:] = bathymetry(state.p_centers[1],bjump=bjump,bathymetry_type=btype)

    grav = 9.8 # Parameter (global auxiliary variable)
    state.problem_data['grav'] = grav
    state.problem_data['dry_tolerance'] = 1.e-3
    state.problem_data['sea_level'] = 0.

    qinit(state,pulse_amplitude)

    #===========================================================================
    # Set up controller and controller parameters
    #===========================================================================
    claw = pyclaw.Controller()
    claw.tfinal = tfinal
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.num_output_times = num_output_times
    claw.keep_copy = False
    claw.write_aux_init = True
    claw.outdir = outdir

    return claw


def get_offset(i):
    if i<30:
        return 0
    elif i < 40:
        raise(Exception)
    elif i<60:
        return 1
    elif i < 70:
        raise(Exception)
    elif i < 95:
        return 2
    elif i < 105:
        raise(Exception)
    elif i < 120:
        return 3
    elif i < 135:
        raise(Exception)
    elif i < 155:
        return 4
    elif i < 165:
        raise(Exception)
    elif i < 190:
        return 5
    elif i < 200:
        raise(Exception)
    elif i < 220:
        return 6
    elif i < 230:
        raise(Exception)
    elif i < 250:
        return 7
    elif i < 265:
        raise(Exception)
    elif i < 280:
        return 8
    elif i < 295:
        raise(Exception)
    else:
        return 9


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup)
