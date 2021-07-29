import numpy as np


def simulation_parameters(ventilation_velocity=0.0):
    return {# Simulation Parameters
            'mdSmall': 1e-10,
            'DSMALL': 1e-10,
            # Initial conditions
            'x_0': 1.5,                 # m ... initial droplet position
            'v_0': -1e-7,                # m/s ... initial droplet velocity
            'Td_0': 273.15+33,          # K ... initial droplet temperature
            # Physical parameters
            'pG': 1.0*101325,           # Pa ... ambient pressure
            'Tb': 373.15,               # K ... boiling point at pG
            'uG': ventilation_velocity, # m/s ... velocity of gas stream
            'g': -9.81 ,                # m/s2 ... gravity
            # Contstants
            'RR': 8.31445985,           # J/(mol K) ... ideal gas constant
            # Thermodynamical properties
            # Water
            'W_h20': 18.015e-3 ,        # kg/mol ... molar mass
            'Tc': 647.096,              # K ... critical temperature
            'pc': 22.064*1e6,           # Pa ... critical pressure
            # Air
            'W_n2': 28.02*1e-3,         # kg/mol ... molar mass
            'W_air': 28.960*1e-3,       # kg/mol ... molar mass
            # Saliva components
            'W_salt': 58.4,             #NaCl # g/mol ... molar mass
            'rho_salt': 2160,
            'W_pro': 66500,             #BSA protein
            'rho_pro': 1362,
            'W_surf': 734,              #DPPC surfactant
            'rho_surf': 1000,
            }

def CL_h2o(T):
    """
    Droplet heat capacity
    D.W. Green, R.H. Perry. Perry's Chemical Engineers' Handbook, 8th
    edition. McGraw-Hill Inc. 2008.

    Input: T in K
    Output: CL in J/kg/K
    """
    params = simulation_parameters()
    W= params['W_h20']*1e3
    C1= 276370 ; C2= -2090.1  ; C3= 8.125 ; C4= -0.014116 ; C5= 9.3701E-06
    C = C1 + C2*T + C3*T**2 + C4*T**3 + C5*T**4 # J/kmol/K
    #  J/(kg K)  =  J/(kmol K)  *  kmol/kg  *  1/ (g/mol)
    return C / W


def Cp_air(T):
    """
    D.W. Green, R.H. Perry. Perry's Chemical Engineers' Handbook, 8th
    edition. McGraw-Hill Inc. 2008.

    Input: T in K
    Output: Cp in J/kg/K
    """
    params = simulation_parameters()
    W= params['W_air']*1e3
    C1= 0.28958e5 ; C2= 0.0939e5 ; C3= 3.012e3; C4= 0.0758e5 ; C5= 1484
    C= C1 + C2*( (C3/T)/np.sinh(C3/T) )**2 + C4*( (C5/T)/np.cosh(C5/T) )**2 # J/kmol/K
    #  J/(kg K)  =  J/(kmol K)  *  kmol/kg  *  1/ (g/mol)
    return C / W


def CpV_h2o(T):
    """
    D.W. Green, R.H. Perry. Perry's Chemical Engineers' Handbook, 8th
    edition. McGraw-Hill Inc. 2008.

    Input: T in K
    Output: Cp in J/kg/K
    """
    params = simulation_parameters()
    W= params['W_h20']*1e3
    C1= 0.33363e5 ; C2= 0.2679e5 ; C3= 2.6105e3 ; C4= 0.08896e5 ; C5= 1169
    C= C1 + C2*( (C3/T)/np.sinh(C3/T) )**2 + C4*( (C5/T)/np.cosh(C5/T) )**2 # J/kmol/K
    #  J/(kg K)  =  J/(kmol K)  *  kmol/kg  *  1/ (g/mol)
    return C / W

def D_h2o_air(P,T):
    """
    Properties of Gases and Liquids
    Fuller et al.

    Input: T in K, P in Pa
    Output: D in m**2/s
    Molar masses
    W_c2h5oh = 46.068  # g/mol ... molar mass
    """
    params = simulation_parameters()
    W_h2o = params['W_h20']*1e3
    W_air = params['W_air']*1e3   # g/mol ... molar mass
    MAB= 2* (1/W_air + 1/W_h2o)**(-1)
    # Pressure
    Pbar= P*1e-5
    # Diffusion volumes
    # #             2*C    + 6* H   + 1*O
    # sigma_c2h5oh= 2*15.9 + 6*2.31 + 1*6.11
    sigma_h2o= 13.1
    sigma_air= 19.7

    # D_AB in cm^2/s
    D_AB= (0.00143* T**1.75) / (Pbar * MAB**(1/2) * (sigma_air**(1/3) + sigma_h2o**(1/3))**2 )
    # m^2/s = cm^2/s * 1e-4 m^2/cm^2

    return D_AB * 1e-4

def lambda_air(T):
    """ D.W. Green, R.H. Perry. Perry's Chemical Engineers' Handbook, 8th
     edition. McGraw-Hill Inc. 2008.

     Input: T in K
     Output: mu in W/(m K)
    """
    C1= 0.00031417 ; C2= 0.7786 ; C3= -0.7116 ; C4= 2121.7
    return C1*T**C2 / (1 + C3/T + C4/T**2)

def lambda_h2o(T):
    """D.W. Green, R.H. Perry. Perry's Chemical Engineers' Handbook, 8th
    edition. McGraw-Hill Inc. 2008.

    Input: T in K
    Output: mu in W/(m K)
    """
    C1= 6.2041E-06 ; C2= 1.3973 ; C3= 0 ; C4= 0
    return C1*T**C2 / (1 + C3/T + C4/T**2)

def LV_c2h5oh(T):
    """ D.W. Green, R.H. Perry. Perry's Chemical Engineers' Handbook, 8th
     edition. McGraw-Hill Inc. 2008.

     Input: T in K ... boiling point
     Output: Latent heat LV in J/kg
    """
    params = simulation_parameters()
    W = params['W_h20']*1e3
    C1= 5.2053e7 ; C2= 0.3199 ; C3= -0.212 ; C4= 0.25795 ; C5= 0
    Tc =  647.096   # K ... critical temperature
    Tr= T/Tc
    L = C1* (1-Tr)**(C2 + C3*Tr + C4*Tr**2 + C5*Tr**3)   # J/kmol
    # J/kg = J/kmol * kmol/kg  *  1/(g/mol)
    return L / W

def mu_air(T):
    """ D.W. Green, R.H. Perry. Perry's Chemical Engineers' Handbook, 8th
     edition. McGraw-Hill Inc. 2008.

     Input: T in K
     Output: mu in Pa*s (= kg/(m s))
    """
    C1= 1.425E-06 ; C2= 0.5039 ; C3= 108.3 ; C4= 0.0
    return (C1* T**C2) / (1 + C3/T + C4/T**2)

def mu_h2o(T):
    """
    D.W. Green, R.H. Perry. Perry's Chemical Engineers' Handbook, 8th
    edition. McGraw-Hill Inc. 2008.

    Input: T in K
    Output: mu in Pa*s (= kg/(m s))
    """
    C1= 1.7096E-08 ; C2= 1.1146 ; C3= 0 ; C4= 0
    return (C1* T**C2) / (1 + C3/T + C4/T**2)

def Psat_h2o(T):
    """
    D.W. Green, R.H. Perry. Perry's Chemical Engineers' Handbook, 8th
    edition. McGraw-Hill Inc. 2008.

    Input: T in K
    Output: Psat in Pa
    """
    C1 = 73.649 ; C2 = -7258.2 ; C3 = -7.3037 ; C4 = 4.1653E-06 ; C5 = 2
    return np.exp(C1 + C2/T + C3*np.log(T) + C4*T**C5)



def rhoL_h2o(T):
    """
    D.W. Green, R.H. Perry. Perry's Chemical Engineers' Handbook, 8th
    edition. McGraw-Hill Inc. 2008.

    Input: T in K
    Output: rhoL in kg/m^3
    """
    params = simulation_parameters()
    W= params['W_h20']*1e3
    tau= 1 - T/647.096
    dens=  17.863 + 58.606*tau**0.35 - 95.396*tau**(2/3) + 213.89*tau - 141.26*tau**(4/3)  # mol/dm^3
    # kg/mol = g/mol * 0.001
    # kg/dm^3 = mol/dm^3 * kg/mol
    # kg/m^3 =  kg/dm^3                       * 1000 dm^3/m^3
    # kg/m^3 =  mol/dm^3 * ( g/mol * 0.001 )  * 1000 dm^3/m^3
    return dens * W


def saliva_mass(D,Td,saliva,n_v):
    """All units checked and correct
    Determine the initial mass, density,
    mass fraction and viral load of the droplet

    Args:
        D (float): droplet diameter units m ?
        Td (float): air temperature units K
        saliva (list): saliva composition concentration [water, salt, protein, surfacant] units mg/ml or kg/m3
        n_v (float): initial viral load concentration. copies/m^3
    Returns:
        list: droplet initial mass, density, mass fraction and viral load
    """
    # Saliva composition kg/m^3
    c_water=saliva[0]
    c_salt=saliva[1]
    c_pro=saliva[2]
    c_surf=saliva[3]
    params = simulation_parameters()
    # Densities of each component (kg/m3)
    rho_w = rhoL_h2o(Td) # water
    rho_salt=params['rho_salt'] #NaCl
    rho_pro=params['rho_pro']  #BSA protein
    rho_surf=params['rho_surf'] #DPPC surfactant

    ## Property of "solid phase"
    #Total concentration [kg/m3]
    c_tot = c_salt + c_pro +c_surf

    # Volumes
    V_d = (np.pi/6)*(D)**(3) #droplet diameter [m3]

    # Mass of each component in the droplet [kg]
    m_salt=c_salt*V_d
    m_pro=c_pro*V_d
    m_surf=c_surf*V_d

    # Volumes [m3]
    V_n=((m_salt/rho_salt + m_pro/rho_pro + m_surf/rho_surf)) #of solid part

    V_w = V_d - V_n # volume of water part [m3]
    m_w = V_w*rho_w # mass of water part [kg]

    m_n = m_salt + m_pro + m_surf # mass of solids [kg]
    md = m_n + m_w # total mass [kg]
    rho_n = m_n/V_n if V_n != 0  else 0 # density of solids [kg/m3]

    yw = m_w/(md) # mass fraction . no units

    ## Emitted virus
    Nv = V_d*n_v # copies
    return [md, rho_n, yw, Nv]


def saliva_evaporation(yw, md, Td, Tinf, RH, saliva,):
    """all units checked and correct!

    Args:
        yw (float): mass fraction of water in the droplet (-)
        md (float): mass of droplet kg
        Td (float): droplet temperature [K]
        Tinf (float): ambient temperature [k]
        RH (float): Relative humidity [0-1]
        saliva (list): saliva composition [c_water c_salt c_pro c_surf] (kg/m3)

    Returns:
        list: [Sw = pw/psat,w (-), dk = droplet droplet diameter (m)]
    """
    # yw = 
    # md = 
    # Td = 
    # Tinf = 
    # s_comp = s
    # Sw = pw/psat,w (-)
    # dk = droplet droplet diameter (m)

    # Properties
    params = simulation_parameters()
    # Saliva composition
    c_salt = saliva[1]
    c_pro = saliva[2]
    c_surf = saliva[3]

    # Ideal gas constant  (J/mol.K)
    R = params['RR']

    # Densities of each component (kg/m3)
    rho_w = rhoL_h2o(Td) # water
    rho_salt = params['rho_salt']  # NaCl
    rho_pro = params['rho_pro']  # BSA protein
    rho_surf = params['rho_surf']  # DPPC surfactant

    # Molecular weight of each component (kg/kmol)
    M_w = params['W_h20'] * 1e3
    M_s = params['W_salt']
    M_pro = params['W_pro']
    M_surf = params['W_surf']

    # Property of "solid phase"
    # Total concentration kg/m3
    c_tot = c_salt + c_pro + c_surf

    # Mass fractions of each component in the solid phase [-]
    y_salt = c_salt/c_tot if c_tot != 0 else 0
    y_pro = c_pro/c_tot if c_tot != 0 else 0
    y_surf = c_surf/c_tot if c_tot != 0 else 0

    # Total mass of solid components
    m_n = (1-yw)*md  # mass of solids kg
    m_w = yw*md  # mass of water kg

    # Mass of each component in the droplet [kg]
    m_salt = y_salt*m_n
    m_pro = y_pro*m_n
    m_surf = y_surf*m_n

    # Volumes [m3]
    V_n = ((m_salt/rho_salt + m_pro/rho_pro + m_surf/rho_surf))  # of solid part
    V_w = m_w/rho_w  # of liquid part

    # Mean density of solid part [kg/m3]
    rho_n = m_n/V_n if V_n != 0 else 0

    # Mean diameter of the solid part [m]
    d_n = (6*V_n/np.pi)**(1/3)

    # Aproximated volume assumption, V_d = Vw + Vn [m3]
    V_d = V_w + V_n
    D = (6*V_d/np.pi)**(1/3)  # droplet diameter [m]

    # Other properties
    # Mole fraction [kmol]
    sumA = (m_salt/M_s) + (m_pro/M_pro) + (m_surf/M_surf) + (m_w/M_w)
    xw = (m_w/M_w) / sumA  # mole fraction of water [-]

    # Surface tension of water
    # Taken from this reference "Surface Tensions of Inorganic Multicomponent Aqueous Electrolyte Solutions and Melts"
    # https://pubs.acs.org/doi/pdf/10.1021/jp105191z
    sig_w = 235.8*((647.15-Td)/647.15)**1.256*(1-0.625*((647.15-Td)/647.15)) # mN m-1
    # Surface tension 
    sig_s = sig_w*1e-3 # N m-1

    # mass of parts relative to water [-]
    nu_ion = 2
    mf_salt = m_salt/m_w
    mf_pro = m_pro/m_w
    mf_surf = m_surf/m_w

    # Molalities (of binary solution) (mol/kg)
    # more mass stuff
    molal_s = (m_salt/M_s)*1000/m_w
    molal_p = (m_pro/M_pro)*1000/m_w
    molal_surf = (m_surf/M_surf)*1000/m_w

    # Osmotic coefficient NaCl, strong electrolyte
    cf = 1e-3*molal_s*(M_s)/rho_salt  # here molal_s in kmol/kg : cf: m3/kg
    Yf = 77.4e-3 # mol/m3
    Aphi = 0.392 # (kg/mol)**(1/2)
    bpit = 1.2 # (kg/mol)**(1/2)
    Betaof = 0.018 # [kg/mol]
    phi_s = 1 + 2*cf*Betaof*Yf - Aphi * \
        (cf**0.5)*(Yf**0.5)/(np.sqrt(2)+bpit*(cf**0.5)*(Yf**0.5)) # [-]

    # Osmotic coefficient BSA and DPCC Surfactant

    # BSA protein
    if molal_p != 0:
        gms = ((rho_pro/rho_w)/(molal_p*M_pro/1000) + 1)**(1/3) # [-]
        phi_p = 1 + (gms**(-3))*(3-gms**(-6))/(1-gms**(-3))**2 # [-]
    else:
        gms = None
        phi_p = None

    # Surfactant
    if molal_surf != 0:
        gms_surf = ((rho_surf/rho_w)/(molal_surf*M_surf/1000) + 1)**(1/3) # [-]
        phi_surf = 1 + (gms_surf**(-3))*(3-gms_surf**(-6))/(1-gms_surf**(-3))**2 # [-]
    else:
        gms_surf = None
        phi_surf = None
    # Evaluate Sw (equation 2.10)
    if molal_p == 0  and molal_s == 0 and molal_surf == 0:
        Sw = np.exp(((4*(M_w/1000)*sig_s)/(R*Tinf*rho_w*D)))
    else:

        Sw = np.exp(((4*(M_w/1000)*sig_s)/(R*Tinf*rho_w*D)) - ((M_w*rho_n*d_n**3)/(rho_w*(D**3-d_n**3)))
                * (nu_ion*phi_s*mf_salt/M_s + phi_surf*mf_surf/M_surf + phi_p*mf_pro/M_pro))

    return [Sw, D]
