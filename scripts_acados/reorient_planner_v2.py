#!/usr/bin/env python3
import numpy as np
# import math
# import pytictoc
import math_lib
import math
import sys
# sys.path.append("/home/dj/acados/interfaces/acados_template/acados_template")
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, AcadosSimSolver
import traceback
from casadi import SX, vertcat, blockcat, power, dot, simplify
import param as param_class

"""
Modification from the original code
1) erase epsilon 
"""

""" OCP formulation
# state variable: x = [phi_r1, phi_r2]
# input variable: u = [alpha, beta]
# algorithm input: 
#   x_prev      : previous step x
#   phi_d       : unmodified orientation reference
#   R           : rotation matrix
#   ur          : rotor thrust, ur=[ur1,...,ur6]

# output:
#   x,u

# parameter:
#   dt          : discrete time interval
#   Am          : ctrl allocation matrix
#   ur_max      : rotor thurst maximum   
#   ur_min      : rotor thrust minimum
#   phidot_lb   : lower bound of alpha, beta
#   phidot_ub   : upper bound of alpha, beta
#   m           : mass of the multirotor
#   g           : gravitational accel.
#   mu_dot      : J_reg_phidot weight
#   mu_ddot     : J_reg_phiddot weight
"""

class reorient_planner:
    def __init__(self, param, hover_shrink_factor = 1.0):
        # self.param = param_class.Param()
        self.param = param
        self.hover_shrink_factor = hover_shrink_factor

        self.param.Am = self.set_Am(param)
        self.param.Am_inv = np.linalg.inv(self.param.Am)

        self.ocp = AcadosOcp()

    def set_Am(self,param): 
        L = param.L
        kf = param.kf
        tilt_ang = param.tilt_ang
        ca = math.cos(tilt_ang)
        sa = math.sin(tilt_ang)

        P1 = L*ca - kf*sa
        P2 = L*sa + kf*ca
        return np.array(
            [ [-0.5*sa, -0.5*sa, sa, -0.5*sa, -0.5*sa, sa],
            [-math.sqrt(3)/2*sa, math.sqrt(3)/2*sa, 0, -math.sqrt(3)/2*sa, math.sqrt(3)/2*sa, 0],
            [ca, ca, ca, ca, ca, ca],
            [-0.5*P1, 0.5*P1, P1, 0.5*P1, -0.5*P1, -P1],
            [-math.sqrt(3)/2*P1, -math.sqrt(3)/2*P1, 0, math.sqrt(3)/2*P1, math.sqrt(3)/2*P1, 0],
            [-P2, P2, -P2, P2, -P2, P2]
            ]
        )

    # def set_ocp_param(self,param_val):
    #     self.ocp.parameter_values = param_val # [] Rf, tau, phi, phi_d

    #     solver_json = 'acados_ocp_' + self.ocp.model.name + '.json'
    #     acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = solver_json)
    #     acados_integrator = AcadosSimSolver(self.ocp, json_file = solver_json)

    #     return acados_ocp_solver, acados_integrator

    def find_ustar(self,phi_r,Rf,tau):
        param_ = self.param
        cm = math_lib.math_lib_wCasadi()

        Rc = cm.Rzyx_numeric(phi_r)
        ustar = param_.Am_inv @ ( np.vstack(( (Rc.T @ Rf), tau )) )
        return ustar

    def init_ocp(self,x0,param_val): # called only once
        # parameter handle
        param_ = self.param

        self.ocp.model = self.ocp_model()
        self.ocp.dims.N = 1

        ############### CONSTRAINT ###############
        deg2rad = math.pi/180.0
        x1 = self.hover_shrink_factor*deg2rad*26.3819
        y1 = self.hover_shrink_factor*deg2rad*14.3216
        y2 = self.hover_shrink_factor*deg2rad*-29.6985

        # 1. input lower & upper bounds
        self.ocp.constraints.lbu = np.array([[param_.phidot_lb],[param_.phidot_lb]]).flatten()
        self.ocp.constraints.ubu = np.array([[param_.phidot_ub],[param_.phidot_ub]]).flatten()
        self.ocp.constraints.idxbu = np.array([0, 1])

        # 2. hoverability cosntraint approximation
        self.ocp.constraints.C_e = np.array([[0.0, 1.0],
                                        [(y1-y2)/x1, -1.0], 
                                        [-(y1-y2)/x1, -1.0]])
        self.ocp.constraints.lg_e = np.array([[y2], [-(2.0*y1-y2)], [-(2.0*y1-y2)]]).flatten()
        self.ocp.constraints.ug_e = np.array([[y1], [-y2], [-y2]]).flatten()
        # 3. rotor thrust min & max constraint
        self.ocp.constraints.lh_e = param_.ur_min.flatten()
        self.ocp.constraints.uh_e = param_.ur_max.flatten()

        # 4. initial value
        self.ocp.constraints.x0 = x0

        ############### COST OPTION ###############
        self.ocp.cost.cost_type = 'EXTERNAL'
        self.ocp.cost.cost_type_e = 'EXTERNAL'

        ############### SOLVER OPTION ###############
        self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES' # FULL_CONDENSING_QPOASES
            # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
            # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        # ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        self.ocp.solver_options.ext_cost_num_hess = 0 # EXACT hessian if 0 
        self.ocp.solver_options.integrator_type = 'ERK' # ERK (Euler), IRK, GNSF, DISCRETE
        # ocp.solver_options.print_level = 1
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.tf = self.param.dt

        self.ocp.solver_options.sim_method_newton_iter = 10
        self.ocp.solver_options.qp_solver_cond_N = 1

        self.ocp.parameter_values = param_val # [] Rf, tau, phi, phi_d

        solver_json = 'acados_ocp_' + self.ocp.model.name + '.json'
        acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = solver_json)
        acados_integrator = AcadosSimSolver(self.ocp, json_file = solver_json)

        return acados_ocp_solver, acados_integrator

    def ocp_model(self) -> AcadosModel:

        model_name = 'reorient_model'

        # set up states & controls
        phi_r1 = SX.sym('phi_r1')
        phi_r2 = SX.sym('phi_r2')

        x = vertcat(phi_r1,phi_r2)

        alpha = SX.sym('alpha')
        beta = SX.sym('beta')

        u = vertcat(alpha,beta)

        # xdot
        phi_r1_dot = SX.sym('phi_r1_dot')
        phi_r2_dot = SX.sym('phi_r2_dot')

        xdot = vertcat(phi_r1_dot,phi_r2_dot)

        # dynamics
        f_expl = vertcat(alpha, beta)
        f_impl = xdot - f_expl

        # set up parameters
        rf1 = SX.sym("rf1") # force 1
        rf2 = SX.sym("rf2") # force 2
        rf3 = SX.sym("rf3") # force 3
        tau1 = SX.sym("tau1") # torque 1
        tau2 = SX.sym("tau2") # torque 2
        tau3 = SX.sym("tau3") # torque 3
        phi_d1 = SX.sym("phi_d1") # phi_d1
        phi_d2 = SX.sym("phi_d2") # phi_d2
        phi_d3 = SX.sym("phi_d3") # phi_d3
        phi1 = SX.sym("phi1") # phi1
        phi2 = SX.sym("phi2") # phi2
        phi3 = SX.sym("phi3") # phi3
        p = vertcat(rf1,rf2,rf3,
                    tau1,tau2,tau3,
                    phi1,phi2,phi3,
                    phi_d1,phi_d2,phi_d3)

        # nonlinear (terminal) constraints
        Rf = vertcat(rf1,rf2,rf3)
        tau = vertcat(tau1,tau2,tau3)
        phi = vertcat(phi1,phi2,phi3)
        cm = math_lib.math_lib_wCasadi()
        R = cm.Rzyx_casadi(phi)

        phi_r = vertcat(phi_r1,phi_r2,phi_d3)
        Rc = cm.Rzyx_casadi(phi_r)
        param_ = self.param
        ustar = param_.Am_inv @ ( vertcat( (Rc.T @ Rf), tau ) )
        con_h_expr_e = ustar

        ############### COST ###############
        # u_casadi = (param_.Am_inv @ vertcat(R.T@Rf, tau))

        # J_phi: orientation error w.r.t. phi_d
        phi_re = vertcat(phi_r1-phi_d1,phi_r2-phi_d2,0.0)
        J_phi = dot(phi_re,phi_re)

        # J_u: avoid rotor saturation
        # ustar = param_.Am_inv @ ( blockcat(Rc.T@R, np.zeros((3,3)), np.zeros((3,3)), np.eye(3)) @ param_.Am @ u_casadi )

        temp = 0
        for k in range(6):
            temp = temp + power( (2.0/(param_.ur_max[k] - param_.ur_min[k])) * (ustar[k] - (param_.ur_min[k] + param_.ur_max[k])/2.0), 6)

        J_u = 1.0/6.0*temp
        
        # J_reg_phidot: phi_r variation regulation
        J_reg_phidot = param_.mu_dot*( power(alpha,2) + power(beta,2))

        J = J_reg_phidot            # running cost
        J_e = J_phi + J_u           # terminal cost

        # set model
        model = AcadosModel()

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = p
        model.name = model_name
        model.con_h_expr_e = con_h_expr_e
        # set cost
        model.cost_expr_ext_cost = J
        model.cost_expr_ext_cost_e = J_e

        return model