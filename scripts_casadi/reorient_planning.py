#!/usr/bin/env python3
import numpy as np
# import math
# import pytictoc
import math_lib
import sys
import traceback
from casadi import *

"""
# opti variable: x = [alpha, beta, eps1 ,eps2]
# input: 
#   x_prev      : previous opti. variable
#   phi_r_prev  : previous phi_r
#   phi_d       : unmodified orientation reference
#   R           : rotation matrix
#   u           : rotor thrust, u=[u1,...,u6]

# output:
#   phi_r       : modified orientation reference
#   alpha       : \dot{phi_r(0)}
#   beta        : \dot{phi_r(1)}
#   eps1,2

# parameter:
#   DT          : discrete time interval
#   Am          : ctrl allocation matrix
#   umax        : rotor thurst maximum   
#   umin        : rotor thrust minimum
#   phidot_lb   : lower bound of alpha, beta
#   phidot_ub   : upper bound of alpha, beta
#   m           : mass of the multirotor
#   g           : gravitational accel.
#   mu_dot      : J_reg_phidot weight
#   mu_ddot     : J_reg_phiddot weight
"""

class reorient_planning:
    def __init__(self, dt=0.01, m_=1.0, g_=9.81, R=np.eye(3), phi_d=np.zeros((3,1)), u=np.zeros((6,1)),
                 Am=np.ones((6,6)), umax=np.ones((6,1)), umin=np.zeros((6,1)),
                 phidot_lb=-1.0, phidot_ub=1.0):
        self.dt_ = dt
        self.m_ = m_
        self.g_ = g_
        self.R_ = R
        self.Am_ = Am
        self.umax = umax
        self.umin = umin
        self.phidot_lb = phidot_lb
        self.phidot_ub = phidot_ub
        self.mu_dot = 3.0*self.dt_
        self.mu_ddot = 1.0

        self.cm = math_lib.math_lib_wCasadi()
        self.Am_inv = np.linalg.inv(self.Am_)

        # input
        self.x_prev = np.zeros((4,1))
        self.phi_r_prev = phi_d
        self.phi_d = phi_d
        self.u = u

        # ouput
        self.phi_r = np.zeros((3,1))

        # optimization problem
        self.opti = Opti()
        self.x = self.opti.variable(4,1)
        self.alpha = self.x[0]
        self.beta = self.x[1]
        self.eps1 = self.x[2]
        self.eps2 = self.x[3]
   
        self.J = 0.0

        # p_opts = {"expand":True}
        p_opts = {"expand":True, "print_iteration": False}

        # self.opti.solver("ipopt",p_opts)
        self.opti.solver("snopt",p_opts)

    def set_variable(self,phi_d,u):
        self.phi_d = phi_d
        self.u = u
        self.phi_r_casadi = self.phi_r_prev + self.dt_*vertcat(self.alpha,self.beta,0.0)
        # self.phi_r_casadi[0] = self.phi_r_prev[0] + self.dt_*self.alpha
        # self.phi_r_casadi[1] = self.phi_r_prev[1] + self.dt_*self.beta
        # self.phi_r_casadi[2] = self.phi_r_prev[2]

    def set_cost(self): 
        #### something wrong in here! J_phi is only defined with numeric values and not by casadi values...
        J_phi = (self.phi_r - self.phi_d).T@(self.phi_r - self.phi_d)

        Rc = self.cm.Rzyx_casadi(self.phi_r_casadi)
        # ustar = self.Am_inv @ ( np.block([[Rc.T@self.R_, np.zeros((3,3))], [np.zeros((3,3)), np.eye(3)]]) @ self.Am_ @ self.u )
        ustar = self.Am_inv @ ( blockcat(Rc.T@self.R_, np.zeros((3,3)), np.zeros((3,3)), np.eye(3)) @ self.Am_ @ self.u )

        temp = 0
        for k in range(6):
            temp = temp + power( (2.0/(self.umax[k] - self.umin[k])) * (ustar[k] - (self.umin[k] + self.umax[k])/2.0), 6)

        J_u = 1.0/6.0*temp

        J_eps = power(self.eps1,2) + power(self.eps2,2)

        alpha_prev = self.x_prev[0]
        beta_prev = self.x_prev[1]

        J_reg_phidot = self.mu_dot*( power(self.alpha,2) + power(self.beta,2))
        J_reg_phiddot = self.mu_ddot*power((alpha_prev - self.alpha),2) + power((beta_prev - self.beta),2)

        self.J = J_phi + J_u + J_eps + J_reg_phidot + J_reg_phiddot

        self.opti.minimize(simplify(self.J))
                 
    def set_constraints(self):
        self.opti.subject_to()

        # ustar_eps
        Rc = self.cm.Rzyx_casadi(self.phi_r_casadi)
        Au = self.Am_ @ self.u
        f = Au[0:3]
        tau = Au[3:6]
        ustar_eps = self.Am_inv @ ( vertcat( (Rc.T @ (self.R_@f - vertcat(self.eps1, self.eps2, 0.0))), tau ) )

        # uhover
        e3 = np.zeros((3,1))
        e3[2] = 1.0
        uhover = self.m_ * self.g_ * self.Am_inv @ (vertcat( Rc.T @ e3, np.zeros((3,1))))

        # umin <= ustar_eps <= umax
        # umin <= uhover <= umax
        for k in range(6):
            temp_c1 = (ustar_eps[k] - self.umax[k])/(self.umax[k] - self.umin[k])
            temp_c2 = (self.umin[k] - ustar_eps[k])/(self.umax[k] - self.umin[k])
            temp_c3 = (uhover[k] - self.umax[k])/(self.umax[k] - self.umin[k])
            temp_c4 = (self.umin[k] - uhover[k])/(self.umax[k] - self.umin[k])
            self.opti.subject_to( simplify(temp_c1) <= 0 )
            self.opti.subject_to( simplify(temp_c2) <= 0 )
            self.opti.subject_to( simplify(temp_c3) <= 0 )
            self.opti.subject_to( simplify(temp_c4) <= 0 )

        # alpha, beta, eps1, eps2 lower & upper bound
        self.opti.subject_to( self.opti.bounded( self.phidot_lb, self.alpha, self.phidot_ub ) )
        self.opti.subject_to( self.opti.bounded( self.phidot_lb, self.beta, self.phidot_ub ) )
        self.opti.subject_to( self.opti.bounded( 0.0, self.eps1, np.inf ) )
        self.opti.subject_to( self.opti.bounded( 0.0, self.eps2, np.inf ) )
 
    def solve(self):
        self.opti.set_initial(self.x,self.x_prev)

        sol_numeric = np.zeros(4)
        sol = self.opti.solve()
        sol_numeric = sol.value(self.x)
        try:
            sol = self.opti.solve()
            sol_numeric = sol.value(self.x)
            print("sol_numeric: ", sol_numeric)
        except:
            sys.stdout = open('solver_output.txt', 'w')
            traceback.print_exc(file=sys.stdout)
            sys.stdout.close()
            print("optimization failed")

        # update previous values for next time use
        self.phi_r[0] = self.phi_r_prev[0] + self.dt_*sol_numeric[0]
        self.phi_r[1] = self.phi_r_prev[1] + self.dt_*sol_numeric[1]
        self.phi_r[2] = self.phi_d[2]

        self.x_prev = sol_numeric
        self.phi_r_prev = self.phi_r

        return np.array([self.phi_r, sol.value(self.x)])