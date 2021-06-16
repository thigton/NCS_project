import numpy as np


def simulation_parameters(ventilation_velocity=0.0):                    
    return {# Simulation Parameters
            'mdSmall': 1e-20,
            'DSMALL': 1e-14,
            # Initial conditions
            'x_0': 1.5,                 # m ... initial droplet position
            'v_0': 1e-6,                # m/s ... initial droplet velocity
            'Td_0': 273.15+33,          # K ... initial droplet temperature
            # Physical parameters
            'pG': 1.0*101325,           # Pa ... ambient pressure
            'Tb': 373.15,               # K ... boiling point at pG
            'uG': ventilation_velocity,                  # m/s ... velocity of gas stream
            'g': -9.81 ,                # m/s2 ... gravity
            # Contstants
            'RR': 8.31445985,           # J/(mol K) ... ideal gas constant
            # Thermodynamical properties
            # Water
            'W_h20': 18.015e-3 ,        # kg/mol ... molar mass
            'Tc': 647.096,              # K ... critical temperature
            'pc': 22.064*1e6,           # Pa ... critical pressure
            # Air
            'W_n2': 28.02*1e-3,
            'W_air': 28.960*1e-3,       # kg/mol ... molar mass
            # Saliva components
            'W_salt': 58.4,             #NaCl
            'rho_salt': 2160,
            'W_pro': 66500,             #BSA protein
            'rho_pro': 1362, 
            'W_surf': 734,              #DPPC surfactant
            'rho_surf': 1000,
            }

def CL_h2o(T):
    """
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

    ## Properties 
    params = simulation_parameters()
    # Saliva composition
    c_water=saliva[0]
    c_salt=saliva[1]
    c_pro=saliva[2]
    c_surf=saliva[3]                      

    # Densities of each component (kg/m3)
    rho_salt=2160 #NaCl
    rho_pro=1362  #BSA protein
    rho_surf=1000 #DPPC surfactant

    #Molecular weight of each component (kg/kmol)

    #Water density
    rho_w = rhoL_h2o(Td)

    ## Property of "solid phase"
    #Total concentration
    c_tot = c_salt + c_pro +c_surf

    # Volumes
    V_d = (np.pi/6)*(D)**(3) #droplet diameter

    # Mass of each component in the droplet
    m_salt=c_salt*V_d
    m_pro=c_pro*V_d
    m_surf=c_surf*V_d

    # Volumes
    V_n=((m_salt/rho_salt + m_pro/rho_pro + m_surf/rho_surf)) #of solid part

    V_w = V_d - V_n
    m_w = V_w*rho_w

    m_n = m_salt + m_pro + m_surf
    md = m_n + m_w

    rho_n = m_n/V_n

    yw = m_w/(md)

    ## Emitted virus

    Nv = V_d*n_v

    return [md, rho_n, yw, Nv]



