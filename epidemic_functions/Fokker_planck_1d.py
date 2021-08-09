import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root
import de_oliveira_droplet_extra_funcs as funcs
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time

def reynolds_number(rho_g, diameter, abs_velocity, dynam_viscosity):
    return rho_g*(abs_velocity)*diameter/dynam_viscosity

def drag_coefficient(Re):
    if isinstance(Re, float):
        if Re == 0:
            return 1e10
        elif Re >= 1e3:
            return 0.424
        else:
            return (24/Re)*(1+(1/6)*Re**(2/3))
    elif isinstance(Re, np.ndarray):
        return np.where(Re >= 1e3, 0.424, np.where(Re==0, 1e20, (24/Re)*(1+(1/6)*Re**(2/3))))
    else:
        raise TypeError('Reynolds number should be either a float or a numpy.ndarray')

def steady_state_velocity(soln, *data):
    w = soln[0]
    rho_g, rho_l, D, w_v, muG, g = data
    Re = reynolds_number(rho_g=rho_g, diameter=D,
                         abs_velocity=abs(w_v-w),
                         dynam_viscosity=muG)
    Cd = drag_coefficient(Re)
    return 3*Cd/D * (rho_g/rho_l)*abs(w_v-w) * (w_v-w) + (1- rho_g/rho_l)*g

def steady_state_velocity_simple(soln, *data):
    w = soln[0]
    w_v, drag, g = data
    return drag*abs(w_v-w)*(w_v-w) + g

def oneDvelocityFPE(t, P, params, drag=None):
    print(f'time: {t:0.6f} secs', end='\r')
    dw = params.dw
    diff_eqs = []
    abs_w = np.abs(params.w_arr-params.vent_w)
    Red = reynolds_number(rho_g=params.rhoInf, diameter=params.diameter, abs_velocity=abs_w, dynam_viscosity=params.muG)
    Cd = drag_coefficient(Red)

    if drag:
        a = drag*abs(params.vent_w-params.w_arr)*(params.vent_w-params.w_arr) + params.g
    else:
        a = params.g*(1-params.rho_g/params.rho_l) + 3*Cd*params.rho_g*abs_w*(params.vent_w - params.w_arr)/(params.rho_l*diameter)
    # b = sigma + np.abs(params.w_arr)
    b = sigma
    # set value of ghost cells 
    f0 = 0
    f1 = 0
    P[0] = P[1]  - f0*dw / b
    P[-1] = P[-2] + f1*dw / b
    for i in range(len(params.w_arr)):
        # print(i, end='\r')
        if i == 0 or  i == len(params.w_arr)-1:
            dPdw = 0
            d2Pdw2 = 0
        else:
            dPdw = (a[i+1]*P[i+1] - a[i-1]*P[i-1]) / (2*dw) # First order central difference
            d2Pdw2 = (b*P[i+1] - 2*b*P[i] + b*P[i-1])/dw**2 # second order central difference

        dPdt = -dPdw + 0.5*d2Pdw2
        diff_eqs.append(dPdt)

    return diff_eqs


class SimulationParameters():
    def __init__(self, diameter, TG, Td, RH, vent_w, sigma):
        self.diameter = diameter
        self.TG = TG
        self.Td = Td
        self.RH = RH
        self.vent_w = vent_w
        self.sigma = sigma
        params = funcs.simulation_parameters(ventilation_velocity=self.vent_w)
        self.pG = params.get('pG')
        self.RR = params.get('RR')
        self.W_air = params.get('W_air') * 1e3
        self.W_h2o = params.get('W_h20') * 1e3
        self.g = params.get('g')
        self.x_h2o = RH*funcs.Psat_h2o(TG)/self.pG  # vapor pressure as a fraction of amb pressure [-]
        self.YG = self.x_h2o*self.W_h2o/(self.x_h2o*self.W_h2o + (1-self.x_h2o)*self.W_air) #[-]
        self.TR = (2*Td + 1*TG)/3 #[K]
        # YR = (2*Yseq + 1*YG)/3  # mass fraction [-]
        self.YR = self.YG  # mass fraction of water in air far away from
        self.WR = (self.YR/self.W_h2o + (1-self.YR)/self.W_air)**(-1) # [kg/kmol]

        self.rho_g = self.pG / (self.RR/(self.WR/1000) * self.TR)  # ideal gas law
        self.rhoInf = self.pG / (self.RR/(self.W_air/1000) * self.TG)
        self.rho_l = funcs.rhoL_h2o(self.Td)
        self.muG = self.YR*funcs.mu_h2o(self.TR) + (1-self.YR)*funcs.mu_air(self.TR)
        self.w0 = 0

    def calculate_steady_state_velocity(self, **kwargs):
        if 'drag_coeff' in kwargs:
            self.drag = kwargs['drag_coeff']
            self.steady_state_solution = root(fun=steady_state_velocity_simple, x0=[0],
                                  args=(self.vent_w, self.drag, self.g),
                                  method='hybr')
        else:
            self.steady_state_solution = root(fun=steady_state_velocity, x0=[0],
                                  args=(self.rho_g, self.rho_l, self.diameter, self.vent_w, self.muG, self.g),
                                  method='hybr')
        self.w_ss = self.steady_state_solution.x[0]

    def assign_w_arr(self, number_of_points):
        rang = abs(self.w_ss - self.w0)
        lb = min(self.w_ss, self.w0) - max(0.75*rang, 2*self.sigma)
        ub = max(self.w_ss, self.w0) + max(0.75*rang, 2*self.sigma)
        self.w_bounds = [lb, ub]
        print(f'velocity range: {self.w_bounds}')
        time.sleep(1)
        self.w_arr = np.linspace(lb, ub, number_of_points)
        self.dw = self.w_arr[1] - self.w_arr[0]

if __name__ == '__main__':
    N = 2001
    droplet_temperature = 20 + 273.15 # [K]
    ambient_temperature = 20 + 273.15 # [K]
    relative_humidity = 0.6 # [%]
    vent_ws = [-0.01, -0.005, 0.00, 0.01] # [m/s]
    diameter = 1e-5 # [m]
    # sigmas = [0.01,0.001] # constant randomness in the system
    sigma = 0.01 # constant randomness in the system
    drag = 1.5
    teval=None
    fig, ax = plt.subplots(figsize=(15,10))
    data = {}
    lines = {}
    sumP = {}
    method = 'BDF'
    for n, vent_w in enumerate(vent_ws):
        # simple class to store all the relevant parameters.
        params = SimulationParameters(diameter=diameter, TG=ambient_temperature, Td=droplet_temperature,
                                      RH=relative_humidity, vent_w=vent_w, sigma=sigma)
        params.calculate_steady_state_velocity()
        params.assign_w_arr(number_of_points=N)
        P0 = np.zeros(shape=params.w_arr.shape)
        w0_idx = np.abs(params.w_arr - params.w0).argmin()
        P0[w0_idx] = 1
        t_bounds = [0, 1.5]
        soln = solve_ivp(fun=oneDvelocityFPE,
                         t_span=t_bounds,
                         y0=P0,
                         method=method,
                         t_eval=teval,
                        #  t_eval=np.geomspace(1e-6, 1.5, 200),
                         args=(params,))

        t = soln.t
        teval = t
        data[n] = soln.y
        sumP[n] = np.trapz(y=data[n],dx=1.0, axis=0)
        lines[n], = ax.plot(params.w_arr, P0, color=f'C{n}', label=f'$w_v$={vent_w:0.3f}m/s, $\sum$P={sumP[n][0]:0.5f}')
        ax.axvline(x=params.w_ss, ls='--', color=f'C{n}', label='steady_state w: ODE')
    ax.set_ylim([0,0.05])
    ax.set_xlim([-0.025,0.025])
    ax.set_ylabel('P(w,t)')
    ax.set_xlabel('w')
    ax.legend(loc='upper left')

    # tables
    column_headers = ['value']
    row_headers = ['ODE', 'FPE','a(w,t)', 'time', 'd', 'b(w,t)']
    cell_text = [[r'$\frac{\mathrm{d} w}{\mathrm{d} t} = \frac{3 C_D}{d}\left(\frac{\rho_g}{\rho_l}\right)\left|w_v - w\right|(w_v - w) + \left(1- \frac{\rho_g}{\rho_l}\right)g$'],
                 [r'$\frac{\partial P(w,t)}{\partial t} = -\frac{\partial}{\partial w}\left(a(w,t)P(w,t)\right) + \frac{1}{2}\frac{\partial^2}{\partial w^2}\left(b(w,t)P(w,t)\right)$'],
                 [r'$\frac{3 C_D}{d}\left(\frac{\rho_g}{\rho_l}\right)\left|w_v - w\right|(w_v - w) + \left(1- \frac{\rho_g}{\rho_l}\right)g$'],
                 [f't={t[0]:0.5f}s'],
                 [f'{diameter*1e6:0.01f}$\mu$m'],
                 [sigma],
                 ]
    the_table = ax.table(cellText=cell_text,
                      rowLabels=row_headers,
                       colLabels=column_headers,
                      loc='upper right',
                      cellLoc='left',
                      )
    the_table.auto_set_column_width(0)
    the_table.scale(1, 2)
    # breakpoint()

    def animate(i, ydata):
        the_table.get_celld()[(4,0)].get_text().set_text(f't={t[i]:0.5f}s')
        new_objs = [the_table]
        for n in range(len(vent_ws)):
            lines[n].set_ydata(ydata[n][:,i])
            lines[n].set_label(f'$w_v$={vent_w:0.3f}m/s, $\sum P={sumP[n][i]:0.5f}')
            new_objs.append(lines[n])
        return new_objs

    ani = animation.FuncAnimation(fig, animate,
                                    frames=len(t),
                                    fargs=(data,),
                                    interval=10,
                                    # repeat=True,
                                    blit=True,)


    writergif = animation.PillowWriter(fps=20) 
    save = input('Do you want to save the animation? [Y/N]')
    if 'y' in save.lower():
        fname = input('File name:')
        ani.save(f'/home/tdh17/Documents/BOX/NCS Project/models/figures/{fname}.gif', writer=writergif)
    plt.show()
    plt.close()