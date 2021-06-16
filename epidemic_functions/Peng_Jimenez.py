import numpy as np
import matplotlib.pyplot as plt

N = 10 # number of occupants
n_i = 0.001
V = 142 # room volume [m^3]
E_p = 100 # virus quanta [hr^-1]
E_co2 = 0.0203 # CO2 quanta [m^3hr^-1]
m_ex = 0.5 # mask efficiency for exhalation
gamma_0 = 3 # [hr^-1] co2 removal rate
gamma = 3.92 # [hr^-1] viral removal rate


t = np.linspace(0,2, 200)
c = n_i*(N-1)*E_p*(1-m_ex)/V * (1/gamma - (1-np.exp(-gamma*t))/(gamma**2*t))
c_co2 = N*E_co2/V * (1/gamma_0 - (1-np.exp(-gamma_0*t))/(gamma_0**2*t))
plt.plot(t,c, label='virus')
plt.plot(t,c_co2, label='co2')
plt.legend()
plt.show()
plt.close()