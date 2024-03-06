#!/usr/bin/env python3
import numpy as np
import math as m
from casadi import *
import reorient_planning
import math_lib
import pytictoc

def set_Am(L,kf,tilt_ang):
    ca = m.cos(tilt_ang)
    sa = m.sin(tilt_ang)

    P1 = L*ca - kf*sa
    P2 = L*sa + kf*ca
    return np.array(
        [ [-0.5*sa, -0.5*sa, sa, -0.5*sa, -0.5*sa, sa],
          [-m.sqrt(3)/2*sa, m.sqrt(3)/2*sa, 0, -m.sqrt(3)/2*sa, m.sqrt(3)/2*sa, 0],
          [ca, ca, ca, ca, ca, ca],
          [-0.5*P1, 0.5*P1, P1, 0.5*P1, -0.5*P1, -P1],
          [-m.sqrt(3)/2*P1, -m.sqrt(3)/2*P1, 0, m.sqrt(3)/2*P1, m.sqrt(3)/2*P1, 0],
          [-P2, P2, -P2, P2, -P2, P2]
        ]
    )

if __name__ == '__main__':

    time_check = pytictoc.TicToc()
    cm = math_lib.math_lib_wCasadi()

    """
    Code validation using arbitrary input
    """

    deg2rad = m.pi/180.0
    rad2deg = 180.0/m.pi

    # parameters
    dt_ = 0.01
    m_ = 2.8
    g_ = 9.81
    L_ = 0.258
    kf_ = 0.016
    tilt_ang_ = deg2rad*30.0
    Am_ = set_Am(L_,kf_,tilt_ang_)
    umax_ = 16.0*np.ones((6,1))
    umin_ = np.zeros((6,1))
    phidot_lb_ = deg2rad*-30.0
    phidot_ub_ = deg2rad*30.0

    # input
    R_ = np.eye(3)
    phi_d_ = np.zeros((3,1))
    u_ = 6.0*np.ones((6,1))

    reorient_plan = reorient_planning.reorient_planning(dt_, m_, g_, R_, phi_d_, u_,
                                                        Am_, umax_, umin_, phidot_lb_, phidot_ub_)

    tf = 1.0
    ts = np.linspace(0.0,int(tf/dt_),tf,endpoint=False)    
    rho = 1.0
    for k in range(len(ts)):
        # control input in the world frame
        if k < int(tf/(2.0*dt_)):
            Rf = np.array([[rho*2.0/tf*ts[k]*m_*g_],[0.0],[m_*g_]])
        else:
            Rf = np.array([[rho*m_*g_],[0.0],[m_*g_]])

        tau = np.zeros((3,1))
        u_ = np.linalg.inv(Am_) @ np.vstack((R_@Rf,tau))

        reorient_plan.set_variable(phi_d_,u_)
        reorient_plan.set_cost()
        reorient_plan.set_constraints()                                                        

        time_check.tic()
        sol = reorient_plan.solve()
        time_check.toc()

        # assumption: R = Rc
        R_ = cm.Rzyx_numeric(np.array([sol[0], sol[1], sol[2]]))

        # from here!!!!
        print("solution: ", sol)