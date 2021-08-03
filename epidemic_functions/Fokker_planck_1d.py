import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import de_oliveira_droplet_extra_funcs as funcs
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def reynolds_number(rho_g, diameter, abs_velocity, dynam_viscosity):
    return rho_g*(abs_velocity)*diameter/dynam_viscosity

def drag_coefficient(Re):
    return (24/Re)*(1+(1/6)*Re**(2/3)) if Re < 1e3 else 0.424

def oneDvelocityFPE(t, P, diameter, TG, Td, RH, vent_w, sigma, w_arr):
    print(f'time: {t:0.6f} secs', end='\r')

    lw = w_arr[0]
    uw = w_arr[-1]
    delta_w = w_arr[1] - w_arr[0]

    params = funcs.simulation_parameters(ventilation_velocity=vent_w)
    pG = params.get('pG')
    RR = params.get('RR')
    W_air = params.get('W_air') * 1e3
    W_h2o = params.get('W_h20') * 1e3
    g = params.get('g')
    p_h2o = RH*funcs.Psat_h2o(TG)  # TG = air_temperature [Pa]
    x_h2o = p_h2o/pG  # vapor pressure as a fraction of amb pressure [-]
    YG = x_h2o*W_h2o/(x_h2o*W_h2o + (1-x_h2o)*W_air) #[-]
    TR = (2*Td + 1*TG)/3 #[K]
    # YR = (2*Yseq + 1*YG)/3  # mass fraction [-]
    YR = YG  # mass fraction of water in air far away from 
    WR = (YR/W_h2o + (1-YR)/W_air)**(-1) # [kg/kmol]

    rho_g = pG / (RR/(WR/1000) * TR)  # ideal gas law
    rhoInf = pG / (RR/(W_air/1000) * TG)
    rho_l = funcs.rhoL_h2o(Td)
    muG = YR*funcs.mu_h2o(TR) + (1-YR)*funcs.mu_air(TR)
    
    diff_eqs = []
    for i in range(len(w_arr)):
        # print(i, end='\r')
        abs_w = abs(w_arr[i]-vent_w)
        Red = reynolds_number(rho_g=rhoInf, diameter=diameter, abs_velocity=abs_w, dynam_viscosity=muG) 
        Cd = drag_coefficient(Red)
        a = g*(1-rho_g/rho_l) + 3*Cd*rho_g*abs_w*(vent_w - w_arr[i])/(rho_l*diameter) 
        
        b = sigma
        if i == 0:
            dPdw = (P[i+1] - P[i]) / (delta_w) # First order forward difference
            d2Pdw2 = (P[i+2] - 2*P[i+1] + P[i])/delta_w**2 # second order forward difference 
        
        elif i == len(w_arr)-1:
            dPdw = (P[i] - P[i-1]) / (delta_w) # First order backward difference
            d2Pdw2 = (P[i] - 2*P[i-1] + P[i-2])/delta_w**2 # second order backward difference 

        else:
            dPdw = (P[i+1] - P[i-1]) / (2*delta_w) # First order central difference
            d2Pdw2 = (P[i+1] - 2*P[i] + P[i-1])/delta_w**2 # second order central difference 
        dPdt = -a*dPdw + 0.5*sigma*d2Pdw2
        diff_eqs.append(dPdt)

    return diff_eqs


if __name__ == '__main__':
    droplet_temperature = 30 + 273.15 # [K]
    ambient_temperature = 20 + 273.15 # [K]
    relative_humidity = 0.6
    vent_w = 0.0127
    w_bounds = [-0.1, 0.1]
    sigma = 0.01
    w_vals = np.linspace(w_bounds[0], w_bounds[1], 201)
    w0 = 0
    P0 = np.zeros(shape=w_vals.shape)
    w0_idx = np.where(np.isin(w_vals, w0))[0]
    P0[w0_idx] = 1
    t_bounds = [0, 1]
    diameter = 1e-5
    soln = solve_ivp(fun=oneDvelocityFPE,
                     t_span=t_bounds,
                     y0=P0,
                     method='BDF',
                     t_eval=np.linspace(t_bounds[0], t_bounds[1], 100),
                     args=(diameter, ambient_temperature, droplet_temperature,
                            relative_humidity, vent_w, sigma, w_vals))

    t = soln.t
    fig, ax = plt.subplots()
    line, = ax.plot(w_vals, P0)

    def animate(i, ydata):
        line.set_ydata(ydata[:,i])
        return line
    
    ani = animation.FuncAnimation(fig, animate, 
                                    fargs=(soln.y,),
                                    repeat=True)




    plt.show()
    plt.close()