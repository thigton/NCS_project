import numpy as np
from numpy.lib.npyio import save
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from savepdf_tex import savepdf_tex
from scipy.stats import lognorm


def get_particle_distribution_parameters():
    df = pd.read_csv('/Users/Tom/Box/NCS Project/models/epidemic_functions/tri-modal-logn-respiratory-particle-distribution.csv',
                     header=[0, 1],
                     index_col=[0],
                     )
    df = df.T
    return df


def eq2_12(cn, CMD, d, GSD, modes, ODE=False):
    """Eq 2.12 from De Oliviera 2020

    Args:
        cn (arr): droplet number concentration [cm^{-3}]
        CMD (arr): count median diameter [\mu m]
        d (float or arr): droplet diameter [\mu m]
        GSD (arr): geometric standard deviation
    """

    dd = 10**d if ODE else d
    if isinstance(d, float):
        sigma = 0
    elif isinstance(d, np.ndarray):
        sigma = np.zeros(shape=d.shape)
    else:
        raise TypeError('diameter type not recognised')
    for i in modes:
        cn_i = cn.loc[i]
        CMD_i = CMD.loc[i]
        GSD_i = GSD.loc[i]
        sigma = sigma + (cn_i / ((2*np.pi)**0.5 * np.log(GSD_i))) * np.exp(-(np.log(dd/CMD_i))**2 / (2*(np.log(GSD_i))**2))

    return np.log(10) * sigma


def get_particle_distribution(params, dia_range=[1, 1000], modes=['1', '2', '3'], plot=False, number_of_diameters=400, **kwargs):
    """Tri modal droplet distribution from either speaking or coughing

    Args:
        params (pd.DataFrame): constants for lognormal distribution fit
        dia_range (list, optional): min and maximume diameter to consider. Defaults to [1, 1000].
        modes (list, optional): modes of the tri modal distribution to include. Defaults to ['1', '2', '3'].
        plot (str, optional): Type of plot to produce current options ['pdf-tex', True]. Defaults to False
        number_of_diameters (int, optional): Number of different diameters to do. Defaults to 400.

    Returns:
        tup<array>: array of diameters [m], array of droplet number concentration [m^-3]
    """
    if plot:
        fig = particle_number_conc_pdf(save_format=plot)
    concentrations = {}
    for action in params.index.get_level_values(1).unique():
        if action == 'Na':
            continue
        param_num = params.xs(key=action, axis=0, level=1).sort_index(
            axis=0, ascending=True)
        if 'dia_eval' in kwargs:
            d = kwargs['dia_eval']
        else:
            d = np.logspace(np.log10(dia_range[0]), np.log10(dia_range[-1]), number_of_diameters)
        dC_dlogdk = eq2_12(
            param_num['Cn_i'], param_num['CMD'], d, param_num['GSD'], modes=modes)

        concentrations[action] = particle_numbers_from_dCndlogD(diameters=d, dCndlogD=dC_dlogdk) # number concentration per
        if plot:
            fig.add_data(d, concentrations, action)
    if plot:
        fig.save()
        
    return d*1e-6, concentrations

class particle_number_conc_pdf:
    def __init__(self, save_format):
        self.save_format = save_format
        self.fname = 'particle_distribution'
        self.f_loc = '/Users/Tom/Box/NCS Project/models/figures/'
        self.fig = plt.figure(figsize=(10, 4))
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.set_xlabel('d [m]')
        self.ax.set_xlim([1e-7, 1e-3])
        self.ax.set_ylim([0.1, 1e6])
        self.ylabel = r'\$(pdf) [m^{-1}]\$' if self.save_format == 'pdf-tex' else '(pdf) $[m^{-1}]$'
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_yscale('log')
        self.ax.set_xscale('log')
        self.ax.grid(True, alpha=0.5)

        if self.save_format == 'pdf-tex':
            self.ax.set_xticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
           [r'\$10^{-7}\$', r'\$10^{-6}\$', r'\$10^{-5}\$', r'\$10^{-4}\$', r'\$10^{-3}\$',])
            self.ax.set_yticks([1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6],
           [r'\$10^{-4}\$', r'\$10^{-2}\$', r'\$10^{0}\$', r'\$10^{2}\$', r'\$10^{4}\$',r'\$10^{6}\$'])

    def add_data(self, diameters, conc, action):
        self.ax.plot(diameters*1e-6, conc[action] , label=action)
        plt.tight_layout()
    
    def save(self):
        self.ax.legend(loc='lower left', frameon=False)
        if self.save_format == 'pdf-tex':
            savepdf_tex(fig=self.fig, fig_loc=self.f_loc, name=self.fname)
        else:
            plt.show()
        plt.close()

def particle_numbers_from_dCndlogD(diameters, dCndlogD, plot=False):
    boundaries = (diameters[:-1] + diameters[1:]) /2
    boundaries = np.insert(boundaries, 0, 2*diameters[0] - boundaries[0])
    boundaries = np.append(boundaries, 2*diameters[-1]-boundaries[-1])
    boundaries = np.log10(boundaries)
    return dCndlogD*(boundaries[1:] - boundaries[:-1])*1e6



def particle_dist_ODE(d, y, *args):
    params, modes = args
    dC_dlogdk = eq2_12(params['Cn_i'], params['CMD'],
                       d, params['GSD'], modes, ODE=True)
    return dC_dlogdk


def particle_cdf(params, modes=['1', '2', '3'], dia_range=[0.1, 1000]):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    for action in params.index.get_level_values(1).unique():
        cn_0 = [0]
        if action == 'Na':
            continue
        tspan = (np.log10(dia_range[0]), np.log10(dia_range[-1]))
        param_num = params.xs(key=action, axis=0, level=1).sort_index(
            axis=0, ascending=True)
        soln = solve_ivp(fun=lambda d, y: particle_dist_ODE(d, y, param_num, modes),
                         t_span=tspan,
                         t_eval=np.log10(np.logspace(
                             tspan[0], tspan[-1], 300)),
                         y0=cn_0,
                         method='RK45'
                         )

        d_k = 10**soln.t
        cn_k = soln.y[0]
        cm_k = cn_k * np.pi*d_k**3/6
        spline = InterpolatedUnivariateSpline(d_k, cn_k)
        pdf = spline.derivative().get_coeffs()
        ax[0].plot(d_k, cn_k, label=action)
        ax[1].plot(d_k, cm_k, label=action)
        ax[2].plot(d_k[:-1], pdf, label=action)
    ax[0].set_xscale('log')
    ax[0].set_xlim([0.1, 1000])
    ax[0].set_ylim([0, 0.25])
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylim([1e-4, 1e7])
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()
    plt.close()


if __name__ == '__main__':
    parameters = get_particle_distribution_parameters()

    source_params = {'speaking': {'t': 30, 'Q': 0.211}, 'coughing': {
        't': 0.5, 'Q': 1.25}}  # in litres and seconds
    # particle_cdf(params=parameters, modes=['1','2','3'])
    get_particle_distribution(params=parameters, modes=[
                              '1', '2', '3'], plot='show')
