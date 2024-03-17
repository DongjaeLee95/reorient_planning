#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import math
from acados_template import latexify_plot

def plot(ts, U, X, t_compt, latexify=False, plt_show=True):

    if latexify:
        latexify_plot()

    N_sim = X.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]

    rad2deg = 180.0/math.pi

    Tf = ts[N_sim-1]

    input_lables = [r'$\alpha$', r'$\beta$']

    plt.figure()
    for k in range(nu):
        plt.subplot(nu, 1, k+1)
        # line, = plt.step(ts, np.append([U[0,k]], U[:,k]))
        # if k < 2:
        line, = plt.plot(ts, rad2deg*np.append([U[0,k]], U[:,k]), label='true')
        # else:
        #     line, = plt.plot(ts, np.append([U[0,k]], U[:,k]), label='true')
            
        line.set_color('r')

        plt.ylabel(input_lables[k])
        plt.xlabel('$t$')
        # plt.hlines(u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
        # plt.hlines(-u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
        # plt.ylim([-1.2*u_max, 1.2*u_max])
        plt.xlim(ts[0], ts[-1])
        plt.grid(True)   
    # plt.savefig('figures/input.png')

    state_lables = [r'$\phi_{r,1}$', r'$\phi_{r,2}$']

    plt.figure()
    for k in range(nx):
        plt.subplot(nx, 1, k+1)
        # if k < 2:
        line, = plt.plot(ts, rad2deg*X[:,k], label='true')
        # else:
        #     line, = plt.plot(ts, X[:,k], label='true')

        plt.ylabel(state_lables[k])
        plt.xlabel('$t$')
        plt.grid(True)
        plt.legend(loc=1)
        plt.xlim(ts[0], ts[-1])
    # plt.savefig('figures/state.png')
    
    plt.figure()
    plt.plot(ts,np.append(t_compt[0], t_compt),color='black',marker='o',linewidth=2)
    plt.grid(True)
    # plt.savefig('figures/compt_time.png')

    if plt_show:
        plt.show()