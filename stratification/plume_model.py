import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
import pandas as pd
import matplotlib.pyplot as plt

def plume(z, y, eps, g, rho_0, rho_amb_df):
    Q = y[0]
    M = y[1]
    F = y[2]
    spline = InterpolatedUnivariateSpline(rho_amb_df.index, rho_amb_df.to_numpy())
    dRhodz = spline.derivatives(z)[1]
    dQdz = 2 * eps * M**(1/2)
    dMdz = F*Q/M
    dFdz = g / rho_0 * dRhodz * Q
    return [dQdz, dMdz, dFdz]

def density_dry_air(temp):
    p = 101325 # Pa
    R = 287 # J/(kg.K)
    return p/(R*temp)

def ventilation_rate(eff_A, z_arr, rho_a_arr, rho_0, g):
    g_prime = (rho_a_arr - rho_0) / rho_0 * g
    g_integral = np.trapz(g_prime, z_arr)
    if g_integral < 0:
        # Upward flow through openings
        Q = eff_A * (np.abs(g_integral))**0.5
    if g_integral >= 0:
        # Downward flow through openings
        Q = - eff_A * g_integral**0.5
    return Q

if __name__ == '__main__':
    source_power = 160 # Watts
    g = 9.81
    cp = 1005 # specific heat capacity of air J / (kg. K)
    T_ref = 20+273.15 # [K]
    rho_0 = density_dry_air(T_ref)

    Q0 = 1e-10
    M0 = 1e-10
    F0 = source_power * g / (T_ref * rho_0 * cp)
    H = 5
    eff_A = 0.01 * H**2
    eps = 0.1
    z = np.linspace(0,H,200)
    temp_arr = np.linspace(21, 21, len(z)) + 273.15
    rho_amb = density_dry_air(temp_arr)
    rho_amb_df = pd.Series(data=rho_amb, index=z, name='rho')

    solver = solve_ivp(fun=plume, 
                     t_span = (0, H),
                     y0=[Q0, M0, F0],
                     t_eval=z,
                     args=(eps, g, rho_0, rho_amb_df),
                     method='RK45')
    if len(z) != len(solver.t):
        print('t_eval didnt work!')
        breakpoint()
    z_plot = solver.t
    rho_amb_plot = rho_amb
    Qz = solver.y[0]
    Mz = solver.y[1]
    Fz = solver.y[2]
    rho_p = rho_amb_plot + rho_0 * Fz / (g * Qz)

    # Get ventilation flow rate
    Qv = ventilation_rate(eff_A=eff_A,
                          z_arr=z_plot,
                          rho_a_arr=rho_amb_plot,
                          rho_0=rho_0,
                          g=g)

    # Determine boundary conditions for the top of the room
    Qr_H = Qz[-1] - Qv
    # Fr_H 
    # breakpoint()



    fig, ax = plt.subplots(2,2, figsize = (10,10))
    ax[0,0].plot(Qz, z_plot, label='Qp')
    ax[0,0].plot(Qz-Qv, z_plot, label='Qr')
    ax[0,0].axvline(Qv,label='Qv')
    ax[0,0].axvline(0,ls=':',color='k')
    ax[0,0].set_xlabel('Q(z)')
    ax[0,0].set_ylabel('z')
    ax[0,0].legend()
    ax[0,1].plot(Mz, z_plot)
    ax[0,1].set_xlabel('M(z)')
    ax[0,1].set_ylabel('z')
    ax[1,0].plot(Fz, z_plot)
    ax[1,0].set_xlabel('F(z)')
    ax[1,0].set_ylabel('z')

    ax[1,1].plot(rho_amb_plot, z_plot, label=r'$\rho_a$')
    ax[1,1].plot(rho_p, z_plot, label=r'$\rho_p$')
    ax[1,1].axvline(rho_0, label=r'$\rho_0$', color='k', ls='--')
    ax[1,1].set_xlabel(r'$\rho(z)$')
    ax[1,1].set_xlim([1.18, 1.3])
    ax[1,1].set_ylabel('z')
    ax[1,1].legend()   
    plt.show()
    plt.close()