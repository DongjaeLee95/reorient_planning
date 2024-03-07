#!/usr/bin/env python3
import numpy as np
import math as m
from casadi import *
import reorient_planning
import math_lib
import pytictoc
import time
import matplotlib.pyplot as plt
import shelve

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
    u0 = 6.0*np.ones((6,1))

    reorient_plan = reorient_planning.reorient_planning(dt_, m_, g_, R_, phi_d_, u0,
                                                        Am_, umax_, umin_, phidot_lb_, phidot_ub_)

    tf = 10 # 1.0
    ts = np.linspace(0.0,tf,int(tf/dt_))    
    rho = 1.0

    # data collection
    compt_time = np.zeros(len(ts))
    phi_r = np.zeros((3,len(ts)))
    u = np.zeros((6,len(ts)))

    for k in range(len(ts)):
        print("step: ",k)
        # control input in the world frame
        if k < int(tf/(2.0*dt_)):
            Rf = np.array([[rho*2.0/tf*ts[k]*m_*g_],[0.0],[m_*g_]])
        else:
            Rf = np.array([[rho*m_*g_],[0.0],[m_*g_]])

        tau = np.zeros((3,1))
        u[:,k] = (np.linalg.inv(Am_) @ np.vstack((R_.T@Rf,tau))).reshape(6,)

        reorient_plan.set_variable(phi_d_,u[:,k])
        reorient_plan.set_cost()
        reorient_plan.set_constraints()                                                        

        # time_check.tic()
        t_start = time.time()
        sol = reorient_plan.solve()
        # compt_time = time_check.toc()
        compt_time[k] = time.time() - t_start

        phi_r[:,k] = sol[0].reshape(3,)
        x_sol = sol[1]

        # assumption: R = Rc
        R_ = cm.Rzyx_numeric(np.array([phi_r[0,k].item(), phi_r[1,k].item(), phi_r[2,k].item()]))

        print("solution: ", sol)

    #%% code block for figures
    # figure plot
    plt1 = plt.figure()
    plt.plot(ts,compt_time,color='black',marker='o',linewidth=2)
    plt.title("computation time")
    plt.xlabel("time [s]")
    plt.ylabel("compt. time [s]")
    plt.grid(True)
    plt.savefig('figures/compt_time.png')

    plt2 = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(ts,rad2deg*phi_r[0,:],color='black',marker='o',linewidth=2)
    # plt.xlabel("time [s]")
    plt.ylabel("roll [deg]")
    plt.grid(True)
    plt.subplot(2,1,2)    
    plt.plot(ts,rad2deg*phi_r[1,:],color='red',marker='o',linewidth=2)
    plt.xlabel("time [s]")
    plt.ylabel("pitch [deg]")
    plt.grid(True)
    plt.savefig('figures/roll,pitch.png')

    plt3 = plt.figure()
    for kp in range(6):
        plt.subplot(3,2,kp+1)
        plt.plot(ts,u[kp,:],color='black',marker='o',linewidth=2)
        plt.grid(True)
        plt.ylabel("u"+str(kp+1))

        if k == 5:
            plt.xlabel("time [s]")
    plt.savefig('figures/rotor thrust.png')

    plt.show()
# %%
"""
- [해결] 뭔가 이상함.. constraint가 이상하게 들어갔나? 왜 roll이 같이 움직여
- [TODO] 그리고 계산속도 드리게 느림.. sqp로 바꾸는 법 알아내야 함
- [TODO] hoverability constraint relaxation 한것도 추가할 것
- matlab에서는 계산한 데이터가 workspace에 저장되어 있어서 figure plot 하는 block 돌리면 바로 결과 얻었는데, 그 비슷한거 안되나?
"""