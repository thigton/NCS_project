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

def fun_dist_bvp(x, y):#, p):
    g = y[0]
    N = y[1]
    # qv = p[0]
    dg = N
    dN = N*qv/(kappa*A)
    return np.vstack([dg, dN])

def dist_bc(ya, yb):#, p):
    gH = yb[0]
    N0 = ya[1]
    # qv = p[0]
    return np.array([gH - F0/qv, N0 + F0/(kappa*A)]) #, qv - eff_a*((g0+gH)/2)**0.5])

def qv_from_buoyancy_profile(buoyancy_profile, vertical_coordinates, effective_area):
    # breakpoint()
    return effective_area * (np.trapz(buoyancy_profile, vertical_coordinates))**0.5

def effective_area(a1, a2):
    cd = 0.6
    return (1/(2*cd**2*a1**2) + 1/(2*cd**2*a2**2))**(-0.5)

def gladstone_woods_results(A_star, F0, H):
    Q = A_star**(2/3) * (F0*H)**(1/3)
    g = F0/Q
    return Q, g

if __name__ == '__main__':
    # Buoyancy flux params
    g = 9.81 # gravity ms^-2
    Wl = 30 # heat flux per unit area W/m^2
    rho_air = 1.225 # kg/m^3
    thermal_expansion_coeff = 3.43e-3 # K^-1
    specific_heat_capacity = 1006 # J/(kg K)
    # vertical points
    H = 3
    z = np.linspace(0,H,200)
    y_flow = np.ones((2, z.shape[0]))
    # Parameters
    A = 200 # m^2
    eff_a = effective_area(a1=0.5, a2=0.5)

    F0 = g*thermal_expansion_coeff*Wl*A/(rho_air*specific_heat_capacity)
    Q_gladstone, g_gladstone = gladstone_woods_results(eff_a, F0, H)
    kappas = [1e-1,1e-2,]
    # qv0 = 
    q_tol = 1e-7
    for kappa in kappas:
        N0 = F0 / (-kappa*A)
        qv = F0/g_gladstone
        count = 0
        tol = 1
        while tol > q_tol:
            print(f'iteration: {count} q_tol : {tol}, qv: {qv}')
            g0 =F0/qv
            y0=[g0,N0]

            strat_soln = solve_bvp(fun_dist_bvp, dist_bc, z, y_flow)#, p=[qv0])
            if strat_soln.success:
                g_profile = strat_soln.y[0,:]
                z_coords = strat_soln.x
                qv_new = qv_from_buoyancy_profile(buoyancy_profile=g_profile,
                                              vertical_coordinates=z_coords,
                                              effective_area=eff_a)
                tol = abs(qv_new-qv)
                qv = qv_new
            else:
                breakpoint()
                print('FAILED!!!')
            count +=1
            if count == 1000:
                break
        plt.plot(g_profile, z_coords,  label=f'$\kappa$:{kappa}')
        print(f'kappa$:{kappa} - Qv : {qv}')
        # plt.title(f'Gladstone Q: {Q_gladstone:0.4f}, Model Q: {strat_soln.p[0]:0.4f}')
    plt.axvline(g_gladstone, label='Gladstone & Wood (2001)', ls='--', color='k')
    print(f'Q_gladstone: {Q_gladstone}')
    plt.legend()
    plt.show()
    plt.close()
    exit()
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
        qvnew = calc_new_qv(delta=delta, z_arr=z, eps=eps, eff_A=eff_A)
        tol = np.abs(qvnew - qv0)
        qv0 = qvnew
