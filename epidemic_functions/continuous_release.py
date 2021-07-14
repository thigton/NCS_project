import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

import de_oliveira_droplet_extra_funcs as funcs
from de_oliveira_droplet_distribution import (
    get_particle_distribution, get_particle_distribution_parameters)
from de_oliveira_droplet_model import state_dot_AS_2, saliva_evaporation


if __name__ == '__main__':
    action = 'speaking'
    # Input parameters
    # Ambient conditions
    air_temperature = 20+273.15        # K ...  ambient temperature
    relative_humidity = [0, 0.2, 0.4, 0.6, 0.8, 1]             # (-) ... relative humidty

    # Droplet Diameter
    # droplet_sizes=np.array([10])*1e-6           # m ... initial droplet diameter

    # Saliva composition
    comp = 'high-pro'
    saliva = [945, 9.00, 76, 0.5]  # High protein sputum

    # SARS-CoV-1 Exponentional decay constant
    lambda_i = 0.636/3600  # (s^-1)
    n_v0 = (10**10)*10**6  # (copies/m^3 of liquid)
    
    # ventilation velocity
    vent_u = 0

    params = funcs.simulation_parameters(ventilation_velocity=vent_u)
    source_params = {'speaking': {'t': 30,
                                  'Q': 0.211},
                     'coughing': {'t': 0.5,
                                  'Q': 1.25}}  # in litres and seconds
    particle_distribution_params = get_particle_distribution_parameters()
    # droplet sizes [m] pdf in litre^-3 m^[-1]
    droplet_sizes, pdf = get_particle_distribution(params=particle_distribution_params,
                                                   modes=['1', '2', '3'],
                                                   source=source_params)
    Td_00 = params['Td_0']
    mdSMALL = params['mdSmall']
    x_00 = params['x_0'] # release height
    v_00 = params['v_0'] # release velocitys
    md_00 = np.array([])
    Nv_00 = np.array([])
    yw_00 = np.array([])
    for droplet in droplet_sizes:
        # Initial droplet mass from composition
        [md_0i, rho_n, yw_0i, Nv_0i] = funcs.saliva_mass(droplet, Td_00, saliva, n_v0)
        md_00 = np.append(md_00, md_0i)
        Nv_00 = np.append(Nv_00, Nv_0i)
        yw_00 = np.append(yw_00, yw_0i)
        # Simulation time
    t_0 = 0               # s ... initial time
    t_end = 300           # s ... end of simulation time
    delta_t = 0.1
    teval = np.arange(t_0,t_end, delta_t)

    delta_volume = source_params[action]['Q']*1e-3 * delta_t # volume of speech in time step
    for RH in relative_humidity:
        X_df = pd.DataFrame(x_00, index=teval, columns=droplet_sizes)
        v_df = pd.DataFrame(v_00, index=teval, columns=droplet_sizes)
        Td_df = pd.DataFrame(Td_00, index=teval, columns=droplet_sizes)
        md_df = pd.DataFrame(np.vstack([md_00]*len(teval)), index=teval, columns=droplet_sizes)
        yw_df = pd.DataFrame(np.vstack([yw_00]*len(teval)), index=teval, columns=droplet_sizes)
        Nv_df = pd.DataFrame(np.vstack([Nv_00]*len(teval)), index=teval, columns=droplet_sizes)
        # D_df = pd.DataFrame(np.vstack([droplet_sizes]*len(teval)), index=teval, columns=droplet_sizes)
        for t in teval: # t is the simulation time
            print(f'{t}secs')
            for t_emit in teval: # t_emit is the time when a row is emitted
                if t_emit >= t:
                    print('t emit it greater than simulation time', end='\r')
                    continue
                for droplet in droplet_sizes:
                    print(f'droplet size: {droplet*1e6:0.3f} micrometres', end='\r')
                    
                    # Set parameters
                    lambda_v = lambda_i
                    s_comp = saliva
                    # D_0 = droplet
                    TG = air_temperature
                    x_0 = X_df.loc[t_emit,droplet]
                    v_0 = v_df.loc[t_emit,droplet]
                    Td_0 = Td_df.loc[t_emit,droplet]
                    md_0 = md_df.loc[t_emit,droplet]
                    yw_0 = yw_df.loc[t_emit,droplet]
                    Nv_0 = Nv_df.loc[t_emit,droplet]
                    # D_0 = D_df.loc[t_emit,droplet]
                    state_0 = [x_0, v_0, Td_0, md_0, yw_0, Nv_0]  # initial state
                    soln = solve_ivp(fun=state_dot_AS_2,
                                 t_span=(t-delta_t, t),
                                 t_eval=[t-delta_t, t],
                                 y0=state_0,
                                 method='BDF',
                                 args=(TG, RH, s_comp, lambda_v, True),
                                 rtol=1e-10,
                                 atol=params['mdSmall'])
                    if soln.success:
                        X_df.at[t_emit, droplet] = soln.y[0,1]
                        v_df.at[t_emit, droplet] = soln.y[1,1]
                        Td_df.at[t_emit, droplet] = soln.y[2,1]
                        md_df.at[t_emit, droplet] = soln.y[3,1]
                        yw_df.at[t_emit, droplet] = soln.y[4,1]
                        Nv_df.at[t_emit, droplet] = soln.y[5,1]
                    else:
                        print(soln.message)
                        breakpoint()                   
            breakpoint()                   

            D_df = (((yw_df*md_df/funcs.rhoL_h2o(
                    Td_df))*6/np.pi + ((1-yw_df)*md_df/rho_n)*6/np.pi)**(1/3))