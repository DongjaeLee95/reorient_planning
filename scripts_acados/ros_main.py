#!/usr/bin/env python3
import numpy as np
import math
import reorient_planner_v2
import math_lib
import pytictoc
import time
import param as param_class
# ros-related
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped as Wrench
from ctrller_msgs.msg import PoseTarget as PoseSp
from ctrller_msgs.msg import Pwm as rotorThurst
from reorient_planning.msg import compt_time
from reorient_planning.msg import opti_variable

from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest

"""
subscribe: 
    1. rotation matrix R (& Euler angle phi)
    2. control input (force f, torque tau)
    3. desired Euler angle, phi_d
publish:
    1. computation time t_compt
    2. optimization variable: alpha, beta
    3. modified orientation angle: phi_{r1,2}
    4. corresponding rotor thrust: ustar
"""

# GLOBAL VARIABLES
ros_rate_g = 100
quat_g = np.zeros((4,1))
R_g = np.eye(3)
phi_g = np.zeros((3,1))
force_g = np.zeros((3,1))
torque_g = np.zeros((3,1))
phi_d_g = np.zeros((3,1))
cm_g = math_lib.math_lib_wCasadi()

odom_cb_flag = False
poseSp_cb_flag = False
wrench_cb_flag = False

initial_run_flag = True
start_flag = False

def odom_cb(msg: Odometry):
    odom_cb_flag = True
    quat_g[0] = msg.pose.pose.orientation.w
    quat_g[1] = msg.pose.pose.orientation.x
    quat_g[2] = msg.pose.pose.orientation.y
    quat_g[3] = msg.pose.pose.orientation.z

    R_g = cm_g.q2R_numeric(quat_g)
    phi_g = cm_g.R2rpy_numeric(R_g)

def poseSp_cb(msg: PoseSp):
    poseSp_cb_flag = True
    quat_d = np.zeros((4))
    
    quat_d[0] = msg.orientation.w
    quat_d[1] = msg.orientation.x
    quat_d[2] = msg.orientation.y
    quat_d[3] = msg.orientation.z

    R_d = cm_g.q2R_numeric(quat_d)
    phi_d_g = cm_g.R2rpy_numeric(R_d)

def wrench_cb(msg: Wrench):
    wrench_cb_flag = True
    force_g[0] = msg.wrench.force.x
    force_g[1] = msg.wrench.force.y
    force_g[2] = msg.wrench.force.z
    
    torque_g[0] = msg.wrench.torque.x
    torque_g[1] = msg.wrench.torque.y
    torque_g[2] = msg.wrench.torque.z

def start_planning_cb(req):
    start_flag = req.data
    return SetBoolResponse(success=True)

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

    x0 = np.zeros((2))
    dt_ = param_.dt

    ### algorithm input: 
    rho = 0.5
    nParam = 12 # Rf, tau, phi, phi_d
    Rf_ = np.zeros((1,3))
    Rf_[0,2] = m*g
    tau_ = np.zeros((1,3))
    phi_d_ = np.zeros((1,3))
    phi_ = np.zeros((1,3))
    param_val = np.hstack((Rf_[0,:], tau_[0,:], phi_[0,:], phi_d_[0,:]))

    ### OCP initialization
    ocp_solver, integrator = reorient_plan_obj.init_ocp(x0,param_val)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
    N = ocp_solver.acados_ocp.dims.N

    for k in range(N+1):
        ocp_solver.set(k, "p", param_val)
    ocp_solver.solve_for_x0(x0_bar = x0)
    
    t_compt = 0.0
    X_sol = np.ndarray((1,nx))
    U_sol = np.ndarray((1,nu))

    ### ROS-RELATED
    try:
        rospy.init_node('reorient_planning', anonymous=False)
        
        odom_sub =rospy.Subscriber('/state_estimator/filtered/lpf', Odometry, odom_cb, tcp_nodelay=True)
        wrench_sub = rospy.Subscriber('/ctrller/wrench', Wrench, wrench_cb, tcp_nodelay=True)
        poseT_sub = rospy.Subscriber('/ref_planner/poseTarget',PoseSp, poseSp_cb, tcp_nodelay=True)

        compt_time_pub = rospy.Publisher('/reorient_planning/compt_time', compt_time, queue_size=10, tcp_nodelay=True)
        ustar_pub = rospy.Publisher('/reorient_planning/ustar', rotorThurst, queue_size=10, tcp_nodelay=True)
        opti_variable_pub = rospy.Publisher('/reorient_planning/opti_variable', opti_variable, queue_size=10, tcp_nodelay=True)
        
        start_srv_server = rospy.Service('/reorient_planning/start',SetBool, start_planning_cb)

        compt_time_msg = compt_time()
        opti_variable_msg = opti_variable()
        rotorThurst_msg = rotorThurst()

        compt_time_msg.header.seq = 0
        compt_time_msg.header.frame_id = "map"
        opti_variable_msg.header.seq = 0
        opti_variable_msg.header.frame_id = "map"
        rotorThurst_msg.header.seq = 0
        rotorThurst_msg.header.frame_id = "map"

        rotorThurst_msg.pwm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        opti_variable_msg.state  = [0.0, 0.0]
        opti_variable_msg.input  = [0.0, 0.0]
        
        rate = rospy.Rate(ros_rate_g)
       
        while not rospy.is_shutdown():

            if (odom_cb_flag and poseSp_cb_flag and wrench_cb_flag):
                # ocp input setting
                Rf_ = (R_g @ force_g).T
                tau_ = torque_g.T
                phi_ = phi_g.T
                phi_d_ = phi_d_g.T
                
                param_val = np.hstack((Rf_[0,:], tau_[0,:], phi_[0,:], phi_d_[0,:]))
                for k in range(N+1):
                    ocp_solver.set(k, "p", param_val)

                if initial_run_flag:
                    initial_run_flag = False
                    x0 = phi_d_[0,0:2]
                    # solve ocp
                    U_sol = ocp_solver.solve_for_x0(x0_bar = x0)
                elif start_flag:
                    # solve ocp
                    U_sol = ocp_solver.solve_for_x0(x0_bar = x0)
                    t_compt = ocp_solver.get_stats('time_tot')
                    # define new x0 (test) -- TODO: should this be ran in MPC manner with feedback?
                    X_sol = integrator.simulate(x = x0,u = U_sol)
                    x0 = X_sol

                    compt_time_msg.header.stamp = rospy.Time.now()
                    opti_variable_msg.header.stamp = rospy.Time.now()
                    rotorThurst_msg.header.stamp = rospy.Time.now()

                    compt_time_msg.header.seq += 1
                    opti_variable_msg.header.seq += 1 
                    rotorThurst_msg.header.seq += 1

                    compt_time_msg.compt_time = t_compt
                    rotorThurst_msg.pwm = reorient_plan_obj.find_ustar(np.vstack((X_sol,phi_d_[0,2])),
                                                                       Rf_.T, tau_.T)
                    opti_variable_msg.state = X_sol
                    opti_variable_msg.input = U_sol

                    compt_time_pub.publish(compt_time_msg)
                    opti_variable_pub.publish(opti_variable_msg)
                    ustar_pub.publish(rotorThurst_msg)

            rate.sleep()


    except rospy.ROSInterruptException:
        pass