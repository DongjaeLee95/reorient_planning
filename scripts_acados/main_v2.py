#!/usr/bin/env python3
import numpy as np
import math
import reorient_planner_v2
import math_lib
import pytictoc
import time
import matplotlib.pyplot as plt
import param as param_class
import matplotlib.pyplot as plt
from plot_result_v2 import plot

if __name__ == '__main__':
    
    deg2rad = math.pi/180.0
    rad2deg = 180.0/math.pi

    dt = 0.01
    ur_max = 16.0*np.ones((6,1))
    ur_min = np.zeros((6,1))
    phidot_lb = deg2rad*-30.0
    phidot_ub = deg2rad*30.0
    m = 2.8
    g = 9.81
    mu_dot = 1.0
    L = 0.258
    kf = 0.016
    tilt_ang = deg2rad*30
    Am = np.zeros((6,6))
    Am_inv = np.zeros((6,6))

    param_ = param_class.Param(dt=dt, Am=Am, Am_inv=Am_inv, ur_max=ur_max, ur_min=ur_min,
                               phidot_lb=phidot_lb, phidot_ub=phidot_ub, m=m, g=g, mu_dot=mu_dot, 
                               L=L, kf=kf, tilt_ang=tilt_ang)

    reorient_plan_obj = reorient_planner_v2.reorient_planner(param = param_)
    param_.Am = reorient_plan_obj.set_Am(param_)
    param_.Am_inv = np.linalg.inv(param_.Am)

    tf = 20.0
    x0 = np.zeros((2))
    dt_ = param_.dt
    Nsim = int(tf/dt_)
    ts = np.linspace(0.0,tf,Nsim+1)

    ### algorithm input: 
    rho = 0.5
    nParam = 12 # Rf, tau, phi, phi_d
    param_val = np.zeros((Nsim,nParam))
    Rf = np.zeros((Nsim,3))
    tau = np.zeros((Nsim,3))
    phi_d = np.zeros((Nsim,3))
    phi = np.zeros((Nsim+1,3))
    for k in range(Nsim):
        if k < int(tf/(2.0*dt_)):
            Rf[k,:] = np.array([rho*2.0/tf*ts[k]*param_.m*param_.g,
                                -rho*2.0/tf*ts[k]*param_.m*param_.g,
                                param_.m*param_.g])
        else:
            Rf[k,:] = np.array([rho*param_.m*param_.g,
                                -rho*param_.m*param_.g,
                                param_.m*param_.g])
        # Rf = np.array([[rho*m_*g_],[0.0],[m_*g_]])
        tau[k,:] = np.zeros((1,3))

    plt.figure()
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(ts,np.append(Rf[0,i], Rf[:,i]),color='black',marker='o',linewidth=2)
        plt.grid(True)
        plt.xlabel("time [s]")
    plt.show()

    ### CLOSED-LOOP SIMULATION
    ocp_solver, integrator = reorient_plan_obj.init_ocp(x0,param_val[0,:])

    # ocp_solver, integrator = reorient_plan_obj.set_ocp_param(param_val[0,:])
    ### MAY NEED TO INITIALIZE! ###

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
    N = ocp_solver.acados_ocp.dims.N

    simX = np.ndarray((Nsim+1,nx))
    simU = np.ndarray((Nsim,nu))
    t_compt = np.zeros((Nsim))
    simX[0,:] = x0

    for k in range(Nsim):
        print("step: ", k)

        phi[k,0:2] = simX[k,0:2] # Assumption: phi_r = phi
        phi[k,2] = phi_d[k,2]

        param_val[k,:] = np.hstack((Rf[k,:], tau[k,:], phi[k,:], phi_d[k,:]))
        
        for kk in range(N+1):
            ocp_solver.set(kk, "p", param_val[k,:])
        # ocp_solver, integrator = reorient_plan_obj.set_ocp_param(param_val[k,:])

        simU[k,:] = ocp_solver.solve_for_x0(x0_bar = simX[k,:])
        t_compt[k] = ocp_solver.get_stats('time_tot')

        # simulate system
        simX[k+1,:] = integrator.simulate(x=simX[k,:],u=simU[k,:]) ## FROM HERE!
    
    plot(ts, simU, simX, t_compt, latexify=False, plt_show=True )