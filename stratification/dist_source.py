import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import root

from kaye_et_al import calc_new_qv

def dist_source_only(t,y, R, qv):
    delta = y[0]
    gamma = y[1]
    ddelta = gamma
    dgam = R*gamma*qv
    return [ddelta, dgam]

def fun_dist_bvp(x, y, p):
    delta = y[0]
    gamma = y[1]
    R = p[0]
    qv = p[1]
    ddelta = gamma
    dgam = R*gamma*qv
    return np.vstack([ddelta, dgam])


if __name__ == '__main__':
    # vertical points
    z = np.linspace(0,1,200)
    # Parameters
    R = 500
    eps = 0.1
    eff_A = 2.12e-3 
    # Init conditions
    delta0 = 0
    gamma0 = -R

    qv0 = 0.2
    q_tol = 1e-7
    tol = 1
    y0=[delta0,gamma0]
    count = 0
    while tol > q_tol:
        count += 1
        print(f' Iteration: {count}, qv0: {qv0:0.3f}')
        solver = solve_ivp(fun=lambda t, y: dist_source_only(t, y, R, qv0),
                            t_span=(z[0], z[-1]),
                            y0=y0,
                            t_eval=z,
                            # args=(R,qv0),
                            method='BDF')


        delta = solver.y[0,:]
        breakpoint()
        qvnew = calc_new_qv(delta=delta, z_arr=z, eps=eps, eff_A=eff_A)
        tol = np.abs(qvnew - qv0)
        qv0 = qvnew
