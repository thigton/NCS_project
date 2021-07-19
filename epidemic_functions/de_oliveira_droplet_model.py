from operator import index
import os
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from scipy import integrate
from scipy.integrate import solve_ivp
from datetime import date
import pickle
import de_oliveira_droplet_extra_funcs as funcs
from de_oliveira_droplet_distribution import (
    get_particle_distribution, get_particle_distribution_parameters)


def saliva_evaporation(yw, md, Td, Tinf, RH, saliva, t, ventilation_velocity):
    # yw = mass fraction of water in the droplet (-)
    # md = mass of droplet kg
    # Td = droplet temperature [K]
    # Tinf = ambient temperature [k]
    # s_comp = saliva composition [c_water c_salt c_pro c_surf] (kg/m3)

    # Sw = pw/psat,w (-)
    # dk = droplet droplet diameter (m)

    # Properties
    params = funcs.simulation_parameters(ventilation_velocity=ventilation_velocity)
    # Saliva composition
    c_water = saliva[0]
    c_salt = saliva[1]
    c_pro = saliva[2]
    c_surf = saliva[3]

    # Ideal gas constant  (J/molK)
    R = params['RR']

    # Densities of each component (kg/m3)
    rho_salt = params['rho_salt']  # NaCl
    rho_pro = params['rho_pro']  # BSA protein
    rho_surf = params['rho_surf']  # DPPC surfactant

    # Molecular weight of each component (kg/kmol)
    M_w = params['W_h20'] * 1e3
    M_s = params['W_salt']
    M_pro = params['W_pro']
    M_surf = params['W_surf']

    # Water density
    rho_w = funcs.rhoL_h2o(Td)

    # Property of "solid phase"
    # Total concentration
    c_tot = c_salt + c_pro + c_surf

    # Mass fractions of each component in the solid phase
    y_salt = c_salt/c_tot
    y_pro = c_pro/c_tot
    y_surf = c_surf/c_tot

    # Total mass of solid components
    m_n = (1-yw)*md  # mass of solids
    m_w = yw*md  # mass of water

    # Mass of each component in the droplet
    m_salt = y_salt*m_n
    m_pro = y_pro*m_n
    m_surf = y_surf*m_n

    # Volumes
    V_n = ((m_salt/rho_salt + m_pro/rho_pro + m_surf/rho_surf))  # of solid part
    V_w = m_w/rho_w  # of liquid part

    # Mean density of solid part
    rho_n = m_n/V_n

    # Mean diameter of the solid part
    d_n = (6*V_n/np.pi)**(1/3)

    # Aproximated volume assumption, V_d = Vw + Vn
    V_d = V_w + V_n
    D = (6*V_d/np.pi)**(1/3)  # droplet diameter

    # Other properties
    # Mole fraction
    sumA = (m_salt/M_s) + (m_pro/M_pro) + (m_surf/M_surf) + (m_w/M_w)
    xw = (m_w/M_w) / sumA  # mole fraction of water

    # Surface tension of water
    # Taken from this reference "Surface Tensions of Inorganic Multicomponent Aqueous Electrolyte Solutions and Melts"
    # https://pubs.acs.org/doi/pdf/10.1021/jp105191z
    TC = Td-273.15
    sig_w = 0.2358*((374.00-TC)/647.15)**1.256*(1-0.625*((374.00-TC)/647.15))
    # Surface tension
    sig_s = sig_w

    # mass of parts relative to water
    nu_ion = 2
    mf_salt = m_salt/m_w
    mf_pro = m_pro/m_w
    mf_surf = m_surf/m_w

    # Molalities (of binary solution) (#kg/mol)
    # more mass stuff
    molal_s = (m_salt/M_s)*1000/m_w
    molal_p = (m_pro/M_pro)*1000/m_w
    molal_surf = (m_surf/M_surf)*1000/m_w

    # Osmotic coefficient NaCl, strong electrolyte
    cf = 1000*molal_s*(M_s)/rho_salt  # molal_s here in kmol/kg
    Yf = 77.4e-3
    Aphi = 0.392
    bpit = 1.2
    Betaof = 0.018
    phi_s = 1 + 2*cf*Betaof*Yf - Aphi * \
        (cf**0.5)*(Yf**0.5)/(np.sqrt(2)+bpit*(cf**0.5)*(Yf**0.5))

    # Osmotic coefficient BSA and DPCC Surfactant

    # BSA protein
    gms = ((rho_pro/rho_w)/(molal_p*M_pro/1000) + 1)**(1/3)
    phi_p = 1 + (gms**(-3))*(3-gms**(-6))/(1-gms**(-3))**2

    # Surfactant
    gms_surf = ((rho_surf/rho_w)/(molal_surf*M_surf/1000) + 1)**(1/3)
    phi_surf = 1 + (gms_surf**(-3))*(3-gms_surf**(-6))/(1-gms_surf**(-3))**2

    # Evaluate Sw (equation 2.10)
    Sw = np.exp(((4*(M_w/1000)*sig_s)/(R*Tinf*rho_w*D)) - ((M_w*rho_n*d_n**3)/(rho_w*(D**3-d_n**3)))
                * (nu_ion*phi_s*mf_salt/M_s + phi_surf*mf_surf/M_surf + phi_p*mf_pro/M_pro))

    return [Sw, D]


def state_dot_AS_2(t, state, TG, RH, s_comp, lambda_v, integrate, ventilation_velocity,):
    """ ODEs for an evaporating droplet.
    Evap model: Using nomenclature from Miller, Harstad & Bellan (1998).
    Saliva model: Mikhailov et al. (2003)
    Virus decay: van Doremalen et al. (2020)
    TG = amb temp.
    RH - relative humidity


    """
    if integrate:
        print(f'time: {t:0.2f}', end='\r')

    X = state[0]  # vertical position
    v = state[1]  # vertical velocity
    Td = state[2]  # droplet temperature
    md = state[3]  # total droplet mass
    yw = state[4]  # water mass fraction in droplet
    Nv = state[5]  # viral load

    # parameters
    params = funcs.simulation_parameters(ventilation_velocity=ventilation_velocity)
    W_air = params['W_air']
    W_h2o = params['W_h20']
    RR = params['RR']
    g = params['g']
    pG = params['pG']
    uG = params['uG']
    # Initial floor check
    if X <= 0:
        X_dot = 0
        v_dot = 0
        Td_dot = 0
        md_dot = 0
        yw_dot = 0
        Nv_dot = 0
    else:
        # Evaluate droplet composition
        md_n = (1-yw)*md  # Mass of solid

        # Effect on evaporation
        # This func is to return equation 2.10 and the diameter
        [Sw, D] = saliva_evaporation(yw, md, Td, TG, RH, s_comp, t, ventilation_velocity)
        rhoL = md/((np.pi/6)*D**3)
        # Saturation pressure and surface concentration
        # Pure water
        pSat = funcs.Psat_h2o(Td)  # based on droplet temperature
        pw = Sw*pSat  # Vapour pressure at the surface rho_w from eq 2.10

        Xseq = pw/pG  # Molar fraction at the surface
        # Mass fraction at the surface
        Yseq = Xseq*W_h2o / (Xseq*W_h2o + (1-Xseq)*W_air)

        # Ambient Humidity
        p_h2o = RH*funcs.Psat_h2o(TG)  # TG = air_temperature
        x_h2o = p_h2o/pG  # vapor pressure as a fraction of amb pressure
        # mass fraction of water in air far away
        YG = x_h2o*W_h2o/(x_h2o*W_h2o + (1-x_h2o)*W_air)

        # Reference conditions
        # ("1/3 rule") This is something from the literature
        # reference temperature (combining the temp of the droplet and far away)
        TR = (2*Td + 1*TG)/3
        YR = (2*Yseq + 1*YG)/3  # mass fraction

        # Gas properties at reference conditions

        # Molar weight @ reference conditions
        # combining to get the average molar weight
        WR = (YR/W_h2o + (1-YR)/W_air)**(-1)

        # Gas properties @ reference conditions
        rhoG = pG / (RR/WR * TR)  # ideal gas law
        # combined specific heat capacity
        CpG = YR*funcs.CpV_h2o(TR) + (1-YR)*funcs.Cp_air(TR)
        # combined dynamic viscosity
        muG = YR*funcs.mu_h2o(TR) + (1-YR)*funcs.mu_air(TR)
        # thermal conductivity?
        lambdaG = YR*funcs.lambda_h2o(TR) + (1-YR)*funcs.lambda_air(TR)
        DDG = funcs.D_h2o_air(pG, TR)  # diffusivity

        # Liquid properties

        #rhoL= rhoL_h2o(Td)
        CL = funcs.CL_h2o(Td)
        # J/kg ... latent heat at boiling point
        LV = funcs.LV_c2h5oh(Td)

        # Derived quantities

        us = abs(v-uG)    # m/s ... difference between particle and gas velocity
        theta1 = CpG / CL
        taud = rhoL*D**2/(18*muG)      # particle time constant for Stokes flow

        # Non-dimensional numbers

        BMeq = (Yseq - YG)/(1 - Yseq)  # spalding mass transfer number

        rhoInf = pG / (RR/W_air * TG)  # density
        Red = rhoInf*us*D/muG  # Reynolds number
        PrG = CpG*muG/lambdaG  # Prandtl Number
        DDG = funcs.D_h2o_air(pG, TR)
        ScG = muG/rhoG/DDG  # Schmidt Number

        # Ranz-Marshall correlation # similar to in Xie 2007 (same power laws different constants)
        Nu = 2 + 0.552 * Red**(1/2) * PrG**(1/3)
        Sh = 2 + 0.552 * Red**(1/2) * ScG**(1/3)

        ## Mass & Temperature
        # Abramzon and Sirignano model
        # If mass fraction at the surface and far away are the same or humidity is maximum.
        if (abs(Yseq-YG) <= 1e-12) or RH == 1:
            md_dot = 0
            Td_dot = 0
        else:
            # Initialise BT
            # Frossling Correlation
            FM = (1+BMeq)**0.7 / BMeq * np.log(1+BMeq)
            FT = FM
            Phi = 1.0
            NuStar = 2 + (Nu-2)/FT
            ShStar = 2 + (Sh-2)/FM
            md_dot = 0 - np.pi*D*ShStar*DDG*rhoG * \
                np.log(1+BMeq)  # equation 2.4
            # specfic heat capacity of water at the reference temp
            CpV = funcs.CpV_h2o(TR)
            LeBar = ScG/PrG  # Schmidt / Prandtl = lewis number?

            # THis is some iterative routine mentioned after equation 2.7 to get a value for
            # the Spalding heat transfer number
            BTnew = (TG-Td)*(CpV/LV)
            BT = -100   # dummy value
            # BT iteration
            i = 0
            imax = 5
            try:
                while (abs((BTnew-BT)/BTnew) > 1e-3) and (i <= imax):
                    BT = BTnew
                    FT = (1+BT)**0.7 / BT * np.log(1+BT) # runtime warning
                    NuStar = 2 + (Nu-2)/FT # runtime warning
                    Phi = (CpV/CpG) * (ShStar/NuStar) * (1/LeBar)
                    BTnew = (1 + BMeq)**Phi - 1.0
                    i = i+1
            except ValueError as e:
                print(e)
            BT = BTnew

            # f2= -md_dot/(md*BT) * (3*PrG*taud/Nu)
            # HdT = 0.0
            Td_dot = md_dot/(md*CL) * (LV - CpV*(TG-Td)/BT)  # equation 2.8
        # global time_check,
        # .append(D)
        # time_check.append(t)
        # Water mass fraction
        yw_dot = (md_n/(md**2))*md_dot  # equation 2.11

        # Motion
        X_dot = v

        # Schiller and Naumann (1933) Drag
        if Red > 1000:
            Cd = 0.424
        else:
            Cd = (24/Red)*(1+(1/6)*Red**(2/3))

        v_dot = g*(1-rhoG/rhoL) - 3*Cd*rhoG*us*(v-uG)/(rhoL*D)  # equation 2.2

        # Viral activity
        Nv_dot = -lambda_v*Nv

    # Model output
    if integrate:
        return [X_dot, v_dot, Td_dot, md_dot, yw_dot, Nv_dot]
    else:
        return md_dot

class DataClass():
    def __init__(self, air_temp, RH, saliva, Lambda,
                t_end, vent_u, particle_distribution, state_0, results):
        self.air_temp = air_temp
        self.RH = RH
        self.saliva = saliva
        self.Lambda = Lambda
        self.simulation_length = t_end
        self.sim_time_resolution = results[0].shape[0]
        self.ventilation_velocity = vent_u
        self.init_state = state_0
        self.initial_particle_distribution = particle_distribution
        self.sim_droplet_resolution = len(self.initial_particle_distribution)
        X_df, v_df, Td_df, md_df, yw_df, Nv_df, D_df = results
        self.droplet_displacement = X_df
        self.droplet_velocity = v_df
        self.droplet_temp = Td_df
        self.droplet_mass = md_df
        self.droplet_mass_fraction = yw_df
        self.droplet_viral_load = Nv_df
        self.droplet_diameter = D_df

        self.sim_date = date.today()
        
        
if __name__ == '__main__':
    vent_u_arr = [-0.01, -0.005,0, 0.005, 0.01]
    plot = False
    # Input parameters
    # Ambient conditions
    air_temperature = 20+273.15
      # K ...  ambient temperature
    # relative_humidity = [0.4]             # (-) ... relative humidty
    relative_humidity = [0.6, 0.8, 1, 0.2,0, 0.4]             # (-) ... relative humidty

    # Droplet Diameter
    # droplet_sizes=np.array([10])*1e-6           # m ... initial droplet diameter

    # Saliva composition
    comp = 'high-pro'
    saliva = [945, 9.00, 76, 0.5]  # High protein sputum

    # SARS-CoV-1 Exponentional decay constant
    lambda_i = 0.636/3600  # (s^-1)
    n_v0 = (10**10)*10**6  # (copies/m^3 of liquid)
    for vent_u in vent_u_arr:
        # Load other parameters
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
        Td_0 = params['Td_0']
        mdSMALL = params['mdSmall']
        x_0 = params['x_0']
        v_0 = params['v_0']
    
        # Simulation time
        t_0 = 0               # s ... initial time
        t_end = 3600           # s ... end of simulation time
        teval = np.arange(0, 3600, 0.5)
        for RH in relative_humidity:
            print(f'Relative Humidity: {RH:0.0%}, vent velocity: {vent_u}')
            X_df = pd.DataFrame(index=teval)
            v_df = pd.DataFrame(index=teval)
            Td_df = pd.DataFrame(index=teval)
            md_df = pd.DataFrame(index=teval)
            yw_df = pd.DataFrame(index=teval)
            Nv_df = pd.DataFrame(index=teval)
            D_df = pd.DataFrame(index=teval)
            for droplet in droplet_sizes:
                print(f'droplet size: {droplet*1e6:0.3f} micrometres')
                # Solve
                # Set parameters
                lambda_v = lambda_i
                s_comp = saliva
                D_0 = droplet
                TG = air_temperature
                # RH = relative_humidity
    
                # Initial droplet mass from composition
                [md_0, rho_n, yw_0, Nv_0] = funcs.saliva_mass(D_0, Td_0, saliva, n_v0)
    
                # Integration
                state_0 = [x_0, v_0, Td_0, md_0, yw_0, Nv_0]  # initial state
                # Integration
                atol = params['mdSmall']
                while True:
                    try:
                        soln = solve_ivp(fun=state_dot_AS_2,
                                     t_span=(t_0, t_end),
                                     t_eval=teval,
                                     y0=state_0,
                                     method='BDF',
                                     args=(TG, RH, s_comp, lambda_v, True, vent_u),
                                     rtol=1e-10,
                                     atol=atol,
                                     )
                    except ValueError:
                        atol *= 2
                    if soln.success:
                        break
                    else:
                        atol *= 2
    
                # Save variables
                t = soln.t
                try:
                    X_df[droplet] = pd.Series(soln.y[0, :], index=t)
                    v_df[droplet] = pd.Series(soln.y[1, :], index=t)
                    Td_df[droplet] = pd.Series(soln.y[2, :], index=t)
                    md_df[droplet] = pd.Series(soln.y[3, :], index=t)
                    yw_df[droplet] = pd.Series(soln.y[4, :], index=t)
                    Nv_df[droplet] = pd.Series(soln.y[5, :], index=t)
                    D_t = (((yw_df[droplet]*md_df[droplet]/funcs.rhoL_h2o(
                        Td_df[droplet]))*6/np.pi + ((1-yw_df[droplet])*md_df[droplet]/rho_n)*6/np.pi)**(1/3))
                    D_df[droplet] = pd.Series(D_t, index=t)
                except:
                    breakpoint()
                # test = np.asarray([state_dot_AS_2(t[i], soln.y[:,i], D_0, TG, RH, md_0, s_comp, lambda_v, integrate=False) for i in range(len(t))])
    
                if plot:
                    # Plots
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
                    ax[0].plot(t, D_t*1e6, 'k-')
                    ax[0].set_xlabel('t (s)')
                    ax[0].set_ylabel('d (\mum)')
                    ax[0].set_xscale('log')
                    ax[0].set_xlim([1e-2, 1e3])
    
                    ax[1].plot(t, soln.y[0, :], 'k-')
                    ax[1].set_xlabel('t (s)')
                    ax[1].set_ylabel('X (m)')
    
                    ax[2].plot(t, soln.y[5, :], 'k-')
                    ax[2].set_xlabel('t (s)')
                    ax[2].set_ylabel('Nv(PFU)')
    
                    plt.show()
                    plt.close()
    
            # SAVE ZE DAATA
            obj = DataClass(air_temp=TG,
                            RH=RH,
                            state_0=state_0,
                            saliva=s_comp,
                            Lambda=lambda_v,
                            t_end=t_end,
                            vent_u=params['uG'],
                            particle_distribution=pdf, 
                            results=(X_df, v_df, Td_df, md_df, yw_df, Nv_df, D_df))
            if not os.path.exists(f'{os.path.dirname(os.path.realpath(__file__))}/data_files/'):
                os.mkdir(f'{os.path.dirname(os.path.realpath(__file__))}/data_files/')
            fname = f'{os.path.dirname(os.path.realpath(__file__))}/data_files/RH_{RH}_u_{params["uG"]}_T_{TG-273.15}_comp_{comp}'.replace('.','-')
            overwrite=True
            if os.path.isfile(f'{fname}.pickle'):
                with open(f'{fname}.pickle', 'rb') as pickle_in:
                    old_obj = pickle.load(pickle_in)
                print(f'File already exists...Date created {old_obj.sim_date}')
                print(f'simulation time resolution: Old:-{old_obj.sim_time_resolution}, New:-{len(teval)}')
                print(f'simulation droplet resolution: Old:-{old_obj.sim_droplet_resolution}, New:-{len(droplet_sizes)}')
                action = input('Do you want to overwrite? [Y/N]')
                if 'n' in action.lower():
                    overwrite=False
            if overwrite:    
                with open(f'{fname}.pickle', 'wb') as pickle_out:
                    pickle.dump(obj, pickle_out)


        # coughing_pdf = pdf * source_params['coughing']['Q'] * source_params['coughing']['t']

        # time = teval[[0, 150, 160, 170, 180, 250, 295]]
        # for t in time:
        #     plt.plot(D_df.loc[t,:], coughing_pdf)
        # plt.xscale('log')
        # plt.xlim([1e-7, 1e-3])
        # plt.yscale('log')
        # plt.ylim([1e-4, 1e6])

        # plt.show()
        # plt.close()
