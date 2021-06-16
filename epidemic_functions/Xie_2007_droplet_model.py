import numpy as np
from scipy.integrate import solve_ivp
# from metpy.calc import saturation_vapor_pressure
import matplotlib.pyplot as plt

def sphere_volume(radius):
    return 4/3 * np.pi * radius**3

def saturation_vapor_pressure(temp):
    """saturation vapor pressure at a certain temperature according to Bolton (1980).
    Taken from metpy.calc.saturation_vapor_pressure.
    They just use the pint library for units which I cba to work with.

    Args:
        temp (float): temperature in Celcius

    Returns:
        (float): vapor presure in Pascals.
    """
    return 6.112*100 * np.exp(17.67 * temp
                                    / (temp + 243.5))

def celciusToKelvin(x):
    return x + 273.15

def kelvinToCelcius(x):
    return x - 273.15

def Xie_2007_no_respiratory_jet_ODEs(t, y, consts, amb_temp=20, relative_humidity=100, ventilation_velocity=-0.01):
    kinematic_viscosity, dynamic_viscosity, gamma, D_inf, air_density, drop_density, cg, cp, Kg, Lv, g, R, Mv, Lambda, p, p_v_inf = consts
    drop_radius = y[0] # drop radius [micro m]
    drop_temp = y[1] # drop temperature [C]
    drop_velocity =  y[2] # drop velocity [m/s]
    drop_displacement = y[3] # drop displacement [m]
    print(t)
    mp = drop_density * sphere_volume(drop_radius*1e-6) # mass of the drop [kg]

    Re_p = drop_radius*2*np.abs(drop_velocity - ventilation_velocity)/ kinematic_viscosity # Reynolds Number
    Sc = kinematic_viscosity / D_inf # Schmidt number
    Pr = cg*dynamic_viscosity/Kg # Prandtl number
    Sh = 1 + 0.3*Re_p**0.5*Sc**(1/3) # Sherwood number
    Nu = 1 + 0.3*Re_p**0.5*Pr**(1/3) # Nusselt number 
    if Re_p >= 1e3:
        Cd = 0.424
    else:
        Cd = 24/Re_p * (1 + 1/6*Re_p**(2/3))


    C = (amb_temp - drop_temp)/amb_temp**(Lambda-1) * (2-Lambda)/(amb_temp**(2-Lambda)- drop_temp**(2-Lambda)) # correction factor because of the temperature dependence of the diffusion coefficient
    p_va = saturation_vapor_pressure(kelvinToCelcius(drop_temp)) # vapor pressure in the gas at the droplet surface

    I =  - (4 * np.pi * drop_radius*1e-6 * C * Mv * D_inf * p * Sh) / (R * amb_temp)* np.log((p - p_va)/(p-p_v_inf)) # [kg/s]
    drdt = C * Mv * D_inf * p * Sh / (drop_density * (drop_radius)*1e-6 * R * amb_temp)* np.log((p - p_va)/(p-p_v_inf))
    # print(f'1:{3* Kg * (amb_temp - drop_temp)/(drop_density*cp*(drop_radius*1e-6)**2)*Nu}, 2:{Lv*I/(mp*cp)}, 3:{3*gamma*(drop_temp**4-amb_temp**4)/(drop_radius*1e-6*cp*drop_density)}')
    dTdt = 3* Kg * (amb_temp - drop_temp)/(drop_density*cp*(drop_radius*1e-6)**2)*Nu - Lv*I/(mp*cp) - 3*gamma*(drop_temp**4-amb_temp**4)/(drop_radius*1e-6*cp*drop_density)

    dVdt = g - 3*Cd*air_density * np.abs(drop_velocity - ventilation_velocity) * (drop_velocity - ventilation_velocity)/(8*drop_density*drop_radius*1e-6)
    # *(1 - drop_density/air_density)
    dxdt = -drop_velocity
    breakpoint()

    return [drdt, dTdt, dVdt, dxdt]

if __name__ == '__main__':
    drop_radius_0 = 5 # micro metres
    drop_temp_0 = celciusToKelvin(20.5) # degrees C
    amb_temp = celciusToKelvin(20) # degrees C
    drop_velocity_0 = 1e-6 # m/s
    drop_displacement_0 = 1.5 # m
    relative_humidity = 50 # % 

    kinematic_viscosity = 1e-5 # kinematic viscosity of air [m^2/s]
    dynamic_viscosity = 18.13e-6 # dynamic_viscosity of air [Pa.s]
    gamma = 5.670374419e-8 # W⋅m^−2⋅K^−4 Stefan-Boltzman constant
    D_inf = 0.146 * 1e-4 # binary diffusion coefficient far from the droplet [m^2/s] https://www.thermopedia.com/content/696/
    air_density = 1.225 # air density [kg/m^3] https://www.macinstruments.com/blog/what-is-the-density-of-air-at-stp/#:~:text=According%20to%20the%20International%20Standard,%3A%200.0765%20lb%2Fft%5E3
    drop_density = 998 # water density [kg/m^3]
    cg = 1.005*1e3 # [J/kg.K] specific heat capacity of air
    cp = 4200  # [J/kg.K] specific heat capacity of drop
    Kg = 25.87 * 1e-3 # Thermal conductivity of air [Watts/metre/Kelvin]
    Lv = 2260*1e3 # [J/kg] Latent heat of vaporization
    g = 9.81 # gravity [ms^-2]
    R = 8.31446261815324 # Ideal gas constant [J/mol/K]
    Mv = 18.01528 * 1e-3 # Molecular weight of vapor [kg/mol]
    Lambda = 1.6 # some correction specific for each substance with a value between 1.6 and 2.
    p = 101325 # total pressure [Pa] (assume this is atmospheric pressure)
    p_v_inf = relative_humidity/100 * saturation_vapor_pressure(kelvinToCelcius(amb_temp)) # patial pressure of the vapor of the droplet ...(water)
    consts = (kinematic_viscosity, dynamic_viscosity, gamma, D_inf, air_density, drop_density, cg, cp, Kg, Lv, g, R, Mv, Lambda, p, p_v_inf)
    soln = solve_ivp(fun=lambda t,y: Xie_2007_no_respiratory_jet_ODEs(t, y, consts=consts, amb_temp=20, relative_humidity=100, ventilation_velocity=0),
                     t_span = (0,1),
                     y0 = [drop_radius_0, drop_temp_0, drop_velocity_0, drop_displacement_0],
                     method='LSODA')
    time = soln.t
    radius = soln.y[0]
    temp = soln.y[1]
    velocity = soln.y[2]
    disp = soln.y[3]
    fig, ax = plt.subplots(2,2, figsize=(12,12))
    ax[0,0].plot(time, radius)
    ax[0,0].set_xlabel('time')
    ax[0,0].set_ylabel('radius')
    ax[1,0].plot(time, temp)
    ax[1,0].set_xlabel('time')
    ax[1,0].set_ylabel('temp')
    ax[0,1].plot(time, velocity)
    ax[0,1].set_xlabel('time')
    ax[0,1].set_ylabel('velocity')
    ax[1,1].plot(time, disp)
    ax[1,1].set_xlabel('time')
    ax[1,1].set_ylabel('disp')
    plt.show()
    plt.close()
