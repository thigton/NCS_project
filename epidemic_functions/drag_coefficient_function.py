import numpy as np
import matplotlib.pyplot as plt


Re = np.linspace(1, 2e3, 300)
Cd = 24/Re * (1 + Re**(2/3)/6)
plt.plot(Re[Re>1000], Cd[Re>1000], color='k', lw=2, alpha=0.5)
Cd[Re>1000] = 0.424
plt.plot(Re, Cd, color='k', lw=2)
plt.plot([1e3, 2e3], [0.424, 0.424], color='k', lw=2)
plt.axvline(1e3, color='k', ls=':')
plt.axhline(0.424, color='k', ls=':')
plt.xlabel('Re')
plt.ylim([0,1])
plt.ylabel('$C_d$')
plt.show()
plt.close()