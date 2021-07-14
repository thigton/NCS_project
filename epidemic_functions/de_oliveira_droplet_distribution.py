import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt


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
    
    cn = cn.loc[modes]
    CMD = CMD.loc[modes]
    GSD = GSD.loc[modes]

    dd = 10**d if ODE else d
    if isinstance(d, float):
        sigma = 0
    elif isinstance(d, np.ndarray):
        sigma = np.zeros(shape=d.shape)
    else:
        raise TypeError('diameter type not recognised')

    for cn_i,  CMD_i, GSD_i in zip(cn, CMD, GSD):
        sigma = sigma + (cn_i / ((2*np.pi)**0.5 * np.log(GSD_i))) * \
            np.exp(-(np.log(dd) - np.log(CMD_i))**2 / (2*(np.log(GSD_i))**2))

    return np.log(10) * sigma


def get_particle_distribution(params, source, dia_range=[1, 1000], modes=['1', '2', '3'], plot=False, **kwargs):
    if plot:
        fig = plt.figure(figsize=(10, 4))
    concentrations = {}
    for action in params.index.get_level_values(1).unique():
        if action == 'Na':
            continue
        param_num = params.xs(key=action, axis=0, level=1).sort_index(
            axis=0, ascending=True)
        
        if 'dia_eval' in kwargs:
            d = kwargs['dia_eval']
        else:
            d = np.logspace(np.log10(dia_range[0]), np.log10(dia_range[-1]), 400)

        dC_dlogdk = eq2_12(
            param_num['Cn_i'], param_num['CMD'], d, param_num['GSD'], modes=modes)
        
        if plot:
            print(action, source[action]['t']*source[action]['Q'])
            plt.plot(d*1e-6, dC_dlogdk*1e6 , label=action)
        concentrations[action] = dC_dlogdk*1e6
    if plot:
        plt.xlabel('d [m]')
        plt.xlim([1e-7, 1e-3])
        plt.ylim([1e-4, 1e6])
        plt.ylabel('dC_dlogdk [$cm^{-3}$]')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.show()
        plt.close()
    return d*1e-6, concentrations


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
                              '1', '2', '3'], source=source_params, plot=True)
