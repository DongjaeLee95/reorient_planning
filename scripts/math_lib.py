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
        return self.Rz_numeric(angs[2]) @ self.Ry_numeric(angs[1]) @ self.Rx_numeric(angs[0])

