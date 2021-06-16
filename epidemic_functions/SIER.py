from numpy.linalg.linalg import solve
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def SIER_model_Noakes_2006(tau, y, R0, sigma):
    """SEIR model from Noakes (2006) Eq. 22

    Args:
        tau (float): Dimensionless time = gamma * t
        y (array): [u, v, x, w] : [susceptables, Infectors, Exposed, Immune] all scaled on total population
        R0 (float): Reproductive ratio
        sigma (float): ratio between exposed to infector rate and recovery rate (alpha / gamma)

    Returns:
        [array]: differential equations
    """
    u = y[0] ; v = y[1] ; x = y[2] ; w = y[3]
    dydt = np.zeros(4)
    dydt[0] = 0-R0*u*v
    dydt[1] = sigma*x - v
    dydt[2] = R0*u*v - sigma*x
    dydt[3] = v
    return dydt

def SIER_model_reduced_occupancy(tau, y, R0, sigma, gamma, workingHrs=(0,24)):
    """SEIR model modified from Noakes 2006 to assume people are only suspectible during working hours 9-17

    Args:
        tau (float): Dimensionless time = gamma * t
        y (array): [u, v, x, w] : [susceptables, Infectors, Exposed, Immune] all scaled on total population
        R0 (float): Reproductive ratio
        sigma (float): ratio between exposed to infector rate and recovery rate (alpha / gamma)
        gamma (float): recovery rate (hr^-1)

    Returns:
        [array]: differential equations
    """
    u = y[0] ; v = y[1] ; x = y[2] ; w = y[3]
    t = tau/gamma
    dydt = np.zeros(4)
    if t % 24 > workingHrs[0] and t % 24 < workingHrs[1]:
        dydt[0] = 0-R0*u*v
        dydt[1] = sigma*x - v
    else:
        dydt[0] = 0
        dydt[1] = -v
    dydt[2] = R0*u*v - sigma*x
    dydt[3] = v
    return dydt

if __name__ == '__main__':
    p = 0.48 # pulmonary inhalation [m^3/hr]
    q = 10 # quanta production rate per infector [quanta/hr]
    N = 200 # Total population
    S = 0.99; I = 0.01; E = 0; R = 0 # [susceptables, Infectors, Exposed, Immune] all scaled on total population
    alpha = 1/24 # progression rate [1/hr]
    gamma = 0.5/24 # removal rate [1/hr]

    V = 7200 # room volume [m^3]
    A = 3 # ACH

    sigma = alpha/gamma
    R0 = p*q*N/(V*A*gamma)
    
    tend = 40
    baseModel = solve_ivp(fun=SIER_model_reduced_occupancy,
                        t_span=(0,tend),
                        t_eval = np.linspace(0,tend, 200),
                        y0=[S,I,E,R],
                        args=(R0, sigma, gamma),
                        method='RK45')

    workingHrs = solve_ivp(fun=lambda t,y: SIER_model_reduced_occupancy(t,y, R0, sigma, gamma, workingHrs=(8,18)),
                        t_span=(0,tend),
                        t_eval = np.linspace(0,tend, 200),
                        y0=[S,I,E,R],
                        method='RK45')
 

    df = pd.DataFrame(data=np.transpose(np.vstack((baseModel.y, workingHrs.y))),
                      columns=['S','I','E','R', 'S_1','I_1','E_1','R_1'],
                      index=baseModel.t/gamma)
    
    df.plot.line(xlabel='time [hr]', grid=True, style=['bs-','ro-','y^-', 'gx-', 'bs:','ro:','y^:', 'gx:'], markevery=10)
    plt.show()
    plt.close()