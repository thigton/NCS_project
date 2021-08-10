import numpy as np
import sdeint
import matplotlib.pyplot as plt
from scipy.stats import norm

# from sde import sde
from Fokker_planck_1d import SimulationParameters, reynolds_number, drag_coefficient

def a(X, t,):
    x = X[0]
    w = X[1]
    global diameter
    droplet_temperature = 20 + 273.15 # [K]
    ambient_temperature = 20 + 273.15 # [K]
    relative_humidity = 0.6 # [%]
    vent_w = 0.0 # [m/s]
    sigma = 0.01
    params = SimulationParameters(diameter=diameter, TG=ambient_temperature, Td=droplet_temperature,
                                      RH=relative_humidity, vent_w=vent_w, sigma=sigma)
    abs_w = np.abs(w-params.vent_w)
    Red = reynolds_number(rho_g=params.rhoInf, diameter=params.diameter, abs_velocity=abs_w, dynam_viscosity=params.muG)
    Cd = drag_coefficient(Red)
    try:
        return np.array([w, (params.g*(1-params.rho_g/params.rho_l) + 3*Cd*params.rho_g*abs_w*(params.vent_w - w)/(params.rho_l*diameter))])
    except (RuntimeWarning, FloatingPointError) as e:
        print(e)

def b(x, t):
    return sigma


    

if __name__ == '__main__':
    # All the change is at the start (0 < t < 1) and needs a smaller delta T
    sim_partition = 0.5 # second
    Lt = 15
    Nt = 500000
    tspan = np.linspace(0,Lt,Nt)

    # tspan_init = np.linspace(0.0, sim_partition, 10001)
    # tspan_ss = np.linspace(sim_partition, 10, 100001)
    # tspan = np.concatenate([tspan_init[:-1], tspan_ss])
    X0 = np.array([1.5, 1e-6])
    sigma = np.diag([0, 50])
    diameters = [5e-6, 1e-5] # [m]

    N = 25
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    for ii, diameter in enumerate(diameters):
        disp = np.zeros((Nt, N))
        print(f'diameter : {diameter*1e6}')
        for i in range(N):
            # second order runge kutta method
            print(f'simulation {i+1}')
            result = sdeint.itoint(a, b, X0, tspan)
            disp[:,i] = result[:,0]
            # result = sdeint.itoint(a, b, X0, tspan_init)
            # X0_2 = result[-1,:] # extract conditions at 1 second for 2nd integration
            # result = result[:-1,:] # remove final row as will be included in 2nd integration
            # print(f'2nd integration')
            # result_ss = sdeint.itoint(a, b, X0_2, tspan_ss)
            # result = np.vstack((result, result_ss))
            ax[0].plot(tspan, disp[:,i], color=f'C{ii}', alpha=0.2)
            # ax[1].plot(tspan, result[:,1], color=f'C{ii}', alpha=0.2)
        ax[0].plot(tspan, np.mean(disp, axis=1),color=f'C{ii}', label=r'd={}\$\mu\$m'.format(diameter*1e6))

        for pdf_time, ls in zip([2, 5, 10, 15],['-','--',':','-.']):
            idx = np.abs(tspan-pdf_time).argmin()
            mu, std = norm.fit(disp[idx, :])
            x = np.linspace(mu-4*std, mu+4*std, 200)
            p = norm.pdf(x, mu, std)
            ax[1].plot(x, p, color=f'C{ii}', ls=ls, label=f't={pdf_time}s')
        

    plt.show()
    plt.close()


