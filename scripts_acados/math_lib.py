import numpy as np
import math as m
from casadi import *

class math_lib_wCasadi:
    def Ry_casadi(self, ang):
        return vertcat( horzcat(cos(ang),0,sin(ang)),np.array([[0,1,0]]),horzcat(-sin(ang),0,cos(ang)))
    
    def Ry_numeric(self, ang):
        return np.array( [[m.cos(ang),0,m.sin(ang)],[0,1,0],[-m.sin(ang),0,m.cos(ang)]])

    def Rx_casadi(self, ang):
        return vertcat( np.array([[1,0,0]]),horzcat(0,cos(ang),-sin(ang)),horzcat(0,sin(ang),cos(ang)))
    
    def Rx_numeric(self, ang):
        return np.array( [[1,0,0],[0,m.cos(ang),-m.sin(ang)],[0,m.sin(ang),m.cos(ang)]])

    def Rz_casadi(self, ang):
        return vertcat(horzcat(cos(ang),-sin(ang),0),horzcat(sin(ang),cos(ang),0),np.array([[0,0,1]]))
    
    def Rz_numeric(self, ang):
        return np.array( [[m.cos(ang),-m.sin(ang),0],[m.sin(ang),m.cos(ang),0],[0,0,1]])

    def Rzyx_casadi(self, angs):
        return self.Rz_casadi(angs[2]) @ self.Ry_casadi(angs[1]) @ self.Rx_casadi(angs[0])

    def Rzyx_numeric(self, angs):
        return self.Rz_numeric(angs[2].item()) @ self.Ry_numeric(angs[1].item()) @ self.Rx_numeric(angs[0].item())

    def q2R_numeric(self, quat):
        quat_scalar = quat[0]
        quat_vector = quat[1:]
        quat_vector_outer = np.outer(quat_vector,quat_vector)
        quat_vector_hat = self.Hat_numeric(quat_vector)
        out_rot = (quat_scalar**2 - quat_vector.T@quat_vector) * np.eye(3) + 2 * quat_vector_outer + 2 * quat_scalar * quat_vector_hat
        return out_rot

    def Hat_numeric(self, vec):
        return np.array([ [0, -vec[2].item(), vec[1].item()], [vec[2].item(), 0, -vec[0].item()], [-vec[1].item(), vec[0].item(), 0]])

    def R2rpy_numeric(self, R):
        r = m.atan2(R[2,1], R[2,2])
        p = m.atan2(-R[2,0], m.sqrt( abs(R[2,1]*R[2,1] + R[2,2]*R[2,2]) ))
        y = m.atan2(R[1,0], R[0,0])

        return np.array([[r], [p], [y]])
