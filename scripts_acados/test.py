#!/usr/bin/env python3
import sys
# sys.path.append("/home/dj/acados/interfaces/acados_template/acados_template")
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
import pytictoc
import math_lib
import math
import matplotlib.pyplot as plt
from casadi import SX, vertcat, blockcat, power, dot

def set_Am(L,kf,tilt_ang):
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

def main():
    model_name = 'zyxEuler_kinematics'

    # set up states & controls
    phi_r1 = SX.sym('phi_r1')
    phi_r2 = SX.sym('phi_r2')
    eps1 = SX.sym('eps1')
    eps2 = SX.sym('eps2')

    x = vertcat(phi_r1,phi_r2,eps1,eps2)

    alpha = SX.sym('alpha')
    beta = SX.sym('beta')
    deps1 = SX.sym('deps1')
    deps2 = SX.sym('deps2')

    u = vertcat(alpha,beta,deps1,deps2)

    # xdot
    phi_r1_dot = SX.sym('phi_r1_dot')
    phi_r2_dot = SX.sym('phi_r2_dot')
    eps1_dot = SX.sym('eps1_dot') 
    eps2_dot = SX.sym('eps2_dot')

    xdot = vertcat(phi_r1_dot,phi_r2_dot,eps1_dot,eps2_dot)

    # dynamics
    f_expl = vertcat(alpha,
                    beta,
                    deps1,
                    deps2)
    f_impl = xdot - f_expl

    # set up parameters
    rf1 = SX.sym("rf1") # force 1
    rf2 = SX.sym("rf2") # force 2
    rf3 = SX.sym("rf3") # force 3
    tau1 = SX.sym("tau1") # torque 1
    tau2 = SX.sym("tau2") # torque 2
    tau3 = SX.sym("tau3") # torque 3
    p = vertcat(rf1,rf2,rf3,tau1,tau2,tau3)

    deg2rad = math.pi/180.0
    rad2deg = 180.0/math.pi
    L_ = 0.258
    kf_ = 0.016
    tilt_ang_ = deg2rad*30.0
    Am_ = set_Am(L_,kf_,tilt_ang_)
    Am_inv = np.linalg.inv(Am_)

    # nonlinear (terminal) constraints
    Rf = vertcat(rf1,rf2,rf3)
    tau = vertcat(tau1,tau2,tau3)
    cm = math_lib.math_lib_wCasadi()

    phi_r = vertcat(phi_r1,phi_r2,0.0)
    Rc = cm.Rzyx_casadi(phi_r)
    ustar_eps = Am_inv @ ( vertcat( (Rc.T @ (Rf - vertcat(eps1, eps2, 0.0))), tau ) )
    con_h_expr_e = ustar_eps

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

    ###############################################
    ocp = AcadosOcp()
    ocp.model = model

    Tf = 0.01
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    N = 1

    # set dimensions
    ocp.dims.N = N

    # external input & parameters
    deg2rad = math.pi/180.0
    rad2deg = 180.0/math.pi
    m_ = 2.8
    g_ = 9.81
    L_ = 0.258
    kf_ = 0.016
    tilt_ang_ = deg2rad*30.0
    Am_ = set_Am(L_,kf_,tilt_ang_)
    Am_inv = np.linalg.inv(Am_)
    umax_ = 16.0*np.ones((6,1))
    umin_ = np.zeros((6,1))
    phidot_lb_ = deg2rad*-30.0
    phidot_ub_ = deg2rad*30.0
    mu_dot_ = 3.0*Tf

    phi_d = np.zeros((3,1))
    R_ = np.eye(3)

    rho = 0.1
    Rf = np.array([[rho*m_*g_],[0.0],[m_*g_]])
    tau = np.zeros((3,1))
    u_numeric = (np.linalg.inv(Am_) @ np.vstack((R_.T@Rf,tau))).reshape(6,)

    # variable defintion
    phi_r = vertcat(phi_r1,phi_r2,phi_d[2])
    phi_re = vertcat(phi_r1-phi_d[0],phi_r2-phi_d[1],0.0)
    cm = math_lib.math_lib_wCasadi()

    # set cost
    J_phi = dot(phi_re,phi_re)

    Rc = cm.Rzyx_casadi(phi_r)
    ustar = Am_inv @ ( blockcat(Rc.T@R_, np.zeros((3,3)), np.zeros((3,3)), np.eye(3)) @ Am_ @ u_numeric )

    temp = 0
    for k in range(6):
        temp = temp + power( (2.0/(umax_[k] - umin_[k])) * (ustar[k] - (umin_[k] + umax_[k])/2.0), 6)

    J_u = 1.0/6.0*temp
    J_eps = power(eps1,2) + power(eps2,2)
    J_reg_phidot = mu_dot_*( power(alpha,2) + power(beta,2))

    J =  J_eps + J_reg_phidot #+ J_reg_phiddot
    J_e = J_u + J_phi

    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = J
    ocp.model.cost_expr_ext_cost_e = J_e

    # set constraints
    x1 = deg2rad*26.3819
    y1 = deg2rad*14.3216
    y2 = deg2rad*-29.6985

    ocp.constraints.lbu = np.array([phidot_lb_, phidot_lb_])
    ocp.constraints.ubu = np.array([phidot_ub_,phidot_ub_])
    ocp.constraints.idxbu = np.array([0, 1])

    ##### 얘네가 문제네!!!! 뭐가 문제야 도대체!!!! - math.inf 가 문제인가? 그런듯!!
    ocp.constraints.C_e = np.array([[0.0, 1.0, 0.0, 0.0],
                                    [(y1-y2)/x1, -1.0, 0.0, 0.0], 
                                    [-(y1-y2)/x1, -1.0, 0.0, 0.0]])
    ocp.constraints.lg_e = np.array([[y2], [-(2.0*y1-y2)], [-(2.0*y1-y2)]]).flatten()
    ocp.constraints.ug_e = np.array([[y1], [-y2], [-y2]]).flatten()
    ocp.constraints.lh_e = umin_.flatten()
    ocp.constraints.uh_e = umax_.flatten()

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0]) # plug in previous value!

    ####### external signal into ocp (e.g. force,torque from the controller!)
    ocp.parameter_values = np.array([Rf[0], Rf[1], Rf[2], tau[0], tau[1], tau[2]])

    ####### set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    # ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.ext_cost_num_hess = 1 # EXACT hessian if 0 
    ocp.solver_options.integrator_type = 'ERK' # ERK (Euler), IRK, GNSF, DISCRETE
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.tf = Tf

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')
    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
            raise Exception(f'acados returned status {status}.')

    # get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")

if __name__ == '__main__':
    main()