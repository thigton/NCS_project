import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg.basic import solve
from scipy.optimize import root

from kaye_et_al import calc_new_qv

def wall_plume_equations(t, y, R, qv, alpha_w, alpha_p, C, f_source, room_width_height_ratio):
    print(t, end='\r')
    q = y[0]
    m = y[1]
    f  = y[2]
    delta = y[3]
    gam = y[4]
    dq = alpha_w / (2*alpha_p)**2 * m / q
    dm = f*q/m - C/(2*alpha_p)**2 * (m/q)**2
    df = f_source - q*gam
    ddelta = gam
    dgam = R * (qv - q*room_width_height_ratio) * gam
    return [dq, dm, df, ddelta, dgam]

def optimize_for_f_top(soln, *data):
    gamma_0 = soln[0]
    z, y0, R, qv0, alpha_w, alpha_p, C, f_source, room_width_height_ratio = data
    y0[-1] = gamma_0
    
    solver = solve_ivp(fun=lambda t, y: wall_plume_equations(t, y, R, qv0, alpha_w, alpha_p, C, f_source, room_width_height_ratio),
                    t_span=(z[0], z[-1]),
                    y0=y0,
                    t_eval=[z[0], z[-1]],
                    atol=1e-2,
                    method='RK45')
    if solver.success:
        # print(solver.message)
        return solver.y[2,-1]
    else:
        print(solver.message)
        return 10

def wall_plume_original_eq(t,y, alpha_w, f_source, q_source):
    q = y[0] ; m = y[1] ; f  = y[2] #; be = y[3]
    dq = alpha_w * m / q + q_source
    dm = f*q/m - C*(m/q)**2
    # dbe = 
    df = f_source
    return [dq, dm, df]

if __name__ == '__main__':
    # vertical points
    H = 1
    W = 1
    z = np.linspace(0.1,H,200)
    room_width_height_ratio = W/H
    # alpha_w = 0.048 #McConnochie & Kerr (2015) experiments included wall shear stress and no finite volume flux
    # C = 0.18 # skin friction coefficient no finite volume flux at the source
    theta = 1 # similarity condition
    f_source = 3.7e-5
    q_source = 1.33e-4
    alpha_w = 0.068 # Parker et al. Part 1 (2021) finite source volume flux
    C = 0.15 # skin friction coefficient Parker et al. part 1(2021)


    R = 500
    eff_A = 2.12e-3 
    alpha_p = 0.1

    # Init conditions
    q0 = q_source*z[0]
    m0 = (2/3)**0.5* z[0]**1.5 * (f_source * q_source)**0.5
    f0 = f_source*z[0]         
    print(q0, m0, f0)                                                                                                                                                                                                                                                                                                                                                                    
    delta0 = 0
    gamma0 = 1e-10
    plume_in_quiesent = solve_ivp(fun = wall_plume_original_eq,
                                  t_span=(z[0], z[-1]),
                                  y0=[q0, m0, f0],
                                #   t_eval=z,
                                  args=(alpha_w, f_source, q_source),
                                  method='RK45')
    if not plume_in_quiesent.success:
        print(plume_in_quiesent.message)
        breakpoint()
    else:
        z_sol = plume_in_quiesent.t
        q = plume_in_quiesent.y[0,:]
        m = plume_in_quiesent.y[1,:]
        f = plume_in_quiesent.y[2,:]
        plt.plot( q, z_sol, label='q')
        plt.plot( m, z_sol, label='m')
        plt.plot( f, z_sol, label='f')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()
        plt.close()
    qv0 = 0.2
    q_tol = 1e-7
    tol = 1
    y0=[q0,m0,f0,delta0,gamma0]
    count = 0
    while tol > q_tol:
        count += 1
        print(f' Iteration: {count}, qv0: {qv0:0.3f}')
        print(f'tol: {tol:0.2f}', end='\r')
        soln = root(fun=optimize_for_f_top, x0=gamma0, args=(z, y0, R, qv0, alpha_w, alpha_p, C, f_source, room_width_height_ratio))
        breakpoint()
        if not soln.success:
            exit()
            breakpoint()
        y0[-1] = soln.x[0]
        solver = solve_ivp(fun=lambda t, y: wall_plume_equations(t, y, R, qv0, alpha_w, alpha_p, C, f_source, room_width_height_ratio),
                            t_span=(z[0], z[-1]),
                            y0=y0,
                            t_eval=z,
                            # args=(R,qv0),
                            method='RK45')

        delta = solver.y[3,:]
        qvnew = calc_new_qv(delta=delta, z_arr=z, eps=alpha_p, eff_A=eff_A)
        tol = np.abs(qvnew - qv0)
        qv0 = qvnew

    gamma = solver.y[4,:]
    q = solver.y[0,:]
    m = solver.y[1,:]
    f = solver.y[2,:]