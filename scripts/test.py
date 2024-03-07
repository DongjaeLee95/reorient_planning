#!/usr/bin/env python3
from casadi import *
import numpy as np
import pytictoc
import matplotlib.pyplot as plt

# Declare variables
x = SX.sym("x",2)

# Form the NLP
f = x[0]**2 + x[1]**2 # objective
g = x[0]+x[1]-10      # constraint
nlp = {'x':x, 'f':f, 'g':g}

# Pick an NLP solver
# MySolver = "ipopt"
#MySolver = "worhp"
MySolver = "sqpmethod"

# Solver options
opts = {}
if MySolver=="sqpmethod":
  opts["qpsol"] = "qpoases" # qpoases
  opts["qpsol_options"] = {"printLevel":"none"}

# Allocate a solver
solver = nlpsol("solver", MySolver, nlp, opts)

# Solve the NLP
time_check = pytictoc.TicToc()

time_check.tic()
sol = solver(lbg=0)
time_check.toc()

# Print solution
print("-----")
print("objective at solution = ", sol["f"])
print("primal solution = ", sol["x"])
print("dual solution (x) = ", sol["lam_x"])
print("dual solution (g) = ", sol["lam_g"])

vec = np.array([1, 2, 3, 4, 5, 6])
vec2 = np.array([1,2,3])
print("vec: ", vec[0:3] @ vec2)
print("vec: ", vec[3:6].T @ vec2)

print("a", np.array([[1, 2, 3],[4, 5, 6]]))
# print("vec: ", vec[1:])

X = [1,2 ,3, 4, 5]
Y = [1, 4, 9, 16 ,25]

plt1 = plt.figure()
plt.plot(X,Y,color='red',marker='o',alpha=0.5,linewidth=2)
plt.title("mohand's code block")
plt.xlabel("x example")
plt.ylabel("y example")
plt.grid(True)
# plt1.show()

plt2 = plt.figure()
plt.plot(X,Y,color='red',marker='o',alpha=0.5,linewidth=2)
plt.title("mohand's code block")
plt.xlabel("x example")
plt.ylabel("y example")
plt.grid(True)
# plt2.show()

plt.show()

# input()