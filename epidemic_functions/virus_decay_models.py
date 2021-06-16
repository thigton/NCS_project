import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Dabisch_et_al_decay(T, RH, S):
    """Empirical formula for the viral decay rate for SARS-CoV-2 from
    "The influence of temperature, humidity, and simulated sunlight on the infectivity of SARS-CoV-2 in aerosols"
    Dabisch et. al 2021

    Args:
        T (float, or m x n array which can do elementwise operations): temperature in degrees celcius
        RH (float, or m x n array which can do elementwise operations): relative humidity (%)
        S (float, or m x n array which can do elementwise operations): integrated UVB irradiance in W/m^2

    Returns:
        (float, or m x n array) : decay constant for viral infectivity in hr^-1
    """
    k = 0.16030 + 0.04018*((T-20.615)/10.585) \
    + 0.02176*((RH-45.235)/28.665) \
    + 0.14369*((S-0.95)/0.95) \
    + 0.02636*((T-20.615)/10.585)*((S-0.95)/0.95) # units min^-1
    return  k * 60

def jones_interpretation():
    sd = 0.43 * 60**2
    mean = 1.75 * 10**(-4) * 60**2
    print(sd, mean)
    dist = lognorm(s=sd, scale=np.exp(mean))
    # breakpoint()
    return dist
    


if __name__ == '__main__':
    n = 100 
    T_arr = np.linspace(10,30,n) # temperature range
    RH_arr = np.linspace(40, 60, n) # relative humidity range
    S_arr = np.linspace(0, 2, n)
    T_mg, RH_mg, S_mg = np.meshgrid(T_arr, RH_arr, S_arr)
    k = Dabisch_et_al_decay(T_mg, RH_mg, S_mg)

    dist = jones_interpretation()
    # Fix RH
    idx = [n//2]
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    for i in idx:
        ax1.plot_surface(T_mg[i,:,:], S_mg[i,:,:], k[i,:,:], alpha=0.5)
    ax1.set_xlabel('temperature')
    ax1.set_ylabel('UV-B irradiance')
    ax1.set_zlabel('decay constant [$hr^{-1}$]')

    ax2 = fig.add_subplot(122)
    x = np.linspace(0, 5, 200)
    ax2.plot(x, dist.pdf(x))
    plt.show()
    plt.close()

