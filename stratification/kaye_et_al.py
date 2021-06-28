import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

def eq(t,y, R, qv):
    q = y[0]
    m = y[1]
    f = y[2]
    delta = y[3]
    gamma = y[4]
    dq = m**0.5
    dm = q*f/m
    ddelta = gamma
    df = -q*ddelta
    dgam = R*ddelta*(qv - np.pi*q)
    return [dq, dm, df, ddelta, dgam]

def optimize_for_f_top(soln, *data):
    gamma_0 = soln[0]
    z, y0, R, qv0 = data
    y0[-1] = gamma_0
    
    solver = solve_ivp(fun=lambda t, y: eq(t, y, R, qv0),
                    t_span=(z[0], z[-1]),
                    y0=y0,
                    t_eval=[z[0], z[-1]],
                    # args=(R,qv0),
                    method='RK45')
    if solver.success:
        # print(solver.message)
        return solver.y[2,-1]
    else:
        print(solver.message)
        return 10

def calc_new_qv(delta, z_arr, eps, eff_A):
    return 1/(4*eps**2) * eff_A * (np.trapz(delta, z_arr))**0.5

def LLSS_h(soln, *data):
    h = soln[0]
    eff_A, C = data
    return C**1.5 * (h**5/(1-h))**0.5 - eff_A

if __name__ == '__main__':
    # vertical points
    z = np.linspace(0,1,200)
    # Parameters
    R = 1000
    eff_A = 2.12e-3 
    eps = 0.1
    # Init conditions
    q0 = 1e-6
    m0 = 1e-6
    f0 = 1
    delta0 = 0
    gamma0 = 1e-15

    qv0 = 0.2
    q_tol = 1e-7
    tol = 1
    y0=[q0,m0,f0,delta0,gamma0]
    count = 0
    while tol > q_tol:
        count += 1
        print(f' Iteration: {count}, qv0: {qv0:0.3f}')
        soln = root(fun=optimize_for_f_top, x0=gamma0, args=(z, y0, R, qv0))
        if not soln.success:
            breakpoint()
        y0[-1] = soln.x[0]
        solver = solve_ivp(fun=lambda t, y: eq(t, y, R, qv0),
                            t_span=(z[0], z[-1]),
                            y0=y0,
                            t_eval=z,
                            # args=(R,qv0),
                            method='RK45')


        delta = solver.y[3,:]
        qvnew = calc_new_qv(delta=delta, z_arr=z, eps=eps, eff_A=eff_A)
        tol = np.abs(qvnew - qv0)
        qv0 = qvnew

    gamma = solver.y[4,:]
    q = solver.y[0,:]
    m = solver.y[1,:]
    f = solver.y[2,:]

    ## LLSS solution for comparison
    C = 6/5 * eps*(9*eps/10)**(1/3)*np.pi**(2/3)
    llss = root(fun=LLSS_h, x0=0.5, args=(eff_A, C))
    h_llss = llss.x[0]
    g_llss = ((2*eps)**(4/3)* np.pi**(2/3) )/ (C*h_llss**(5/3))
    g_llss_profile = np.zeros(shape=z.shape)
    g_llss_profile[z > h_llss] = g_llss
    plt.style.use('seaborn-deep')
    fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=True)
    ax[0].set_title(f'qv: {qvnew:0.4f}')
    try:
        ax[0].plot(q,z, label='q')
        ax[0].plot(m,z, label='m')
        ax[0].plot(f,z, label='f')
        ax[0].legend()
    except ValueError:
        breakpoint()
    ax[1].plot(delta,z, label=r'$\delta$')
    ax[1].plot(g_llss_profile,z, label=r'$\delta_{LLSS}$')
    ax[1].legend()
    ax[0].set_xlabel('q, m, f')
    ax[0].set_ylabel(r'$\zeta$')
    ax[1].set_xlabel(r'$\delta$')
    plt.show()
    plt.close()