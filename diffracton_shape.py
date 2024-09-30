#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from clawpack import petclaw as pyclaw
from clawpack import riemann
from homog_pseudospectral import bracket

g = 9.8

def pwc_homog_coeffs(b_1, b_2, Y):
    """
    Compute homogenized equation coefficients for piecewise-constant bathymetry with alpha=0.5.
    This function IS NOT CORRECT for other kinds of bathymetry.
    It assumes that the bathymetry takes value b_1 over the first half-interval and b_2 over the second half.
    It also assumes the unperturbed surface value is zero.
    Finally, it assumes that the channel goes from y=-1/2 to y=1/2.
    """
    H_1 = -b_1
    H_2 = -b_2
    bd4=(b_1-b_2)/4
    bracket_H = -1*((Y<0.)*bd4*(0.5+2*Y) + (Y>=0.)*bd4*(0.5-2*Y))

    # The next formula is based on the interval [0,1] but we instead use [-1/2,1/2]
    # so we need to shift the y values.
    Y = Y+1/2
    dd = -bd4/2
    brHinvbrH = (1/192)*(2-H_1/H_2 - H_2/H_1) + (Y<0.5)*dd*Y*(2*Y-1)/H_1 - (Y>=0.5)*dd*(2*Y**2-3*Y+1)/H_2

    return bracket_H, brHinvbrH


def initialize_solitary_wave(A,X,Y,b,fac,expon=2.,btype="pwc",bjump=None):
    #fac = 4.85 # A = 0.05
    #fac = 6.  # A = 0.2
    dx = X[1,0]-X[0,0]

    if btype == "pwc":
        bH, bHH = pwc_homog_coeffs(b[0,0], b[0,-1], Y)
    elif btype == "sinusoidal":
        y1 = Y[0,:]

        H = lambda y : -1 - bjump/2 + bjump*np.sin(2*np.pi*y)
        bH = bracket(H)
        HinvbH = lambda y : bH(y) / H(y)
        bHH = bracket(HinvbH) 
        bHH = np.vectorize(bHH)
        bHH1 = bHH(y1)
        bHH = np.tile(bHH1,(Y.shape[0],1))

        bH = np.vectorize(bH)
        bH1 = bH(y1)
        bH = -np.tile(bH1,(Y.shape[0],1))
    else:
        raise Exception("unrecognized btype; valid choices are pwc or sinusoidal")

    f = A/(np.cosh(fac*np.sqrt(A)*X))**expon
    fpp = np.diff(f,2,axis=0,prepend=0,append=0)/dx**2
    if A >= 0.2:
        denom = 4
    else:
        denom = 1

    surface = f - bHH*fpp/denom
    h = surface - b
    u = np.sqrt(g)*surface
    hu = h*u

    umean = u.mean(axis=1)
    dudx = np.diff(umean,prepend=0)/dx

    hv = -np.expand_dims(dudx,1)*bH

    return h, hu, hv

def qinit(state,A,fac,bathymetry_type='pwc',bjump=1.2,expon=2.):
    "Gaussian surface perturbation"
    b = state.aux[0,:,:] # Bathymetry
    X,Y = state.grid.p_centers
    dx = X[1,0]-X[0,0]

    bH, bHH = pwc_homog_coeffs(b[0,0], b[0,-1], Y)
    h, hu, hv = initialize_solitary_wave(A, X, Y, b, fac, expon)

    state.q[0,:,:] = h
    state.q[1,:,:] = hu
    state.q[2,:,:] = hv

    
def bathymetry(y,bjump=1.0,bathymetry_type='pwc'):
    if bathymetry_type == 'pwc':
        b_A = -1. - bjump/2
        b_B = -1. + bjump/2
        return (y>0)*b_A + (y<=0)*b_B
    elif bathymetry_type == 'sinusoidal':
        return -1. - bjump/2 + bjump*np.sin(2*np.pi*y)

def setup(A=0.05,bjump=1.2,cells_per_period=20,tfinal=30,solver_type='classic',
          riemann_solver='fwave',outdir='./_output',btype='pwc',channel_width=1.0,
          num_output_times=20,L=100.,fac=4.85, expon=2.):


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
    elif solver_type == 'sharpclaw':
        solver = pyclaw.SharpClawSolver2D(rs)

    solver.fwave = True
    
    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic

    solver.aux_bc_lower[0] = pyclaw.BC.periodic
    solver.aux_bc_upper[0] = pyclaw.BC.periodic
    solver.aux_bc_lower[1] = pyclaw.BC.periodic
    solver.aux_bc_upper[1] = pyclaw.BC.periodic


    # Domain:
    xlower = -L/2;  xupper =  L/2.
    ylower = -channel_width/2.;  yupper =  channel_width/2.

    
    mx = (xupper-xlower)*cells_per_period
    my = (yupper-ylower)*cells_per_period

    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    y = pyclaw.Dimension(ylower,yupper,my,name='y')
    domain = pyclaw.Domain([x,y])

    num_aux = 1
    state = pyclaw.State(domain,solver.num_eqn,num_aux)
    state.aux[:,:,:] = bathymetry(state.p_centers[1],bjump=bjump,bathymetry_type=btype)

    grav = g # Parameter (global auxiliary variable)
    state.problem_data['grav'] = grav
    state.problem_data['dry_tolerance'] = 1.e-3
    state.problem_data['sea_level'] = 0.

    qinit(state,A,fac,bathymetry_type=btype,bjump=bjump,expon=expon)

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


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup)
