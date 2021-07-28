from typing import Type
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import uncertainties as unc
import uncertainties.unumpy as unp

from savepdf_tex import savepdf_tex

def func(Nv, k):
    return 1- np.exp(-Nv/k)


def get_response_model_data_sets():
    df = pd.read_csv('/Users/Tom/Box/NCS Project/models/epidemic_functions/response_model_data_sets.csv')
    df['prob'] = df['Positive'] / df['Tested']
    return df

def curve_fit_to_data(df, xcol, ycol):
    xdata = df.loc[df['Virus'].isin(['rSARS-CoV', 'MHV-1']), xcol]
    ydata = df.loc[df['Virus'].isin(['rSARS-CoV', 'MHV-1']), ycol]
    popt, pcov = curve_fit(func, xdata, ydata)
    return popt, pcov


def get_coeff(popt, pcov, with_uncertainties=True):
    return unc.correlated_values(popt, pcov) if with_uncertainties else popt[0]

def calculate_CI(k, Nv):
    if isinstance(k[0], unc.core.AffineScalarFunc):
        P = 1 - unp.exp(-Nv/k)
        nom = unp.nominal_values(P)
        std = unp.std_devs(P)
        lb = nom - 1.96 * std
        ub = nom + 1.96 * std
        return lb, ub
    else:
        raise TypeError('uncertainties.core.AffineScalarFunc type required')

def get_prob_CI(Nv):
    df = get_response_model_data_sets()
    xdata = 'Dose'
    ydata = 'prob'
    popt, pcov = curve_fit_to_data(df, xcol=xdata, ycol=ydata)
    coeff = get_coeff(popt, pcov, with_uncertainties=True)
    confidence_intervals = calculate_CI(k=coeff, Nv=Nv)
    return confidence_intervals

def get_k_CI():
    df = get_response_model_data_sets()
    xdata = 'Dose'
    ydata = 'prob'
    popt, pcov = curve_fit_to_data(df, xcol=xdata, ycol=ydata)
    coeff = get_coeff(popt, pcov, with_uncertainties=True)

    lb = coeff[0].nominal_value - 1.96 * coeff[0].std_dev
    ub = coeff[0].nominal_value + 1.96 * coeff[0].std_dev
    return lb, ub


if __name__ == '__main__':
    # CI = get_prob_CI(Nv = np.logspace(-1, 4, 500))
    # k_CI = get_k_CI()
    # print(CI)
    df = get_response_model_data_sets()
    xdata = df.loc[df['Virus'].isin(['rSARS-CoV', 'MHV-1']), 'Dose']
    ydata = df.loc[df['Virus'].isin(['rSARS-CoV', 'MHV-1']), 'prob']
    popt, pcov = curve_fit_to_data(df=df, xcol='Dose', ycol='prob')
    coeff = get_coeff(popt, pcov, with_uncertainties=True)

    px = np.logspace(-1, 4, 500)

    py = 1 - unp.exp(-px/coeff)
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)
    lb = nom - 1.96 * std
    ub = nom + 1.96 * std
    # plot data
    
    fig = plt.figure()
    # plot the regression
    plt.plot(px, nom, c='black', label='best fit')

    # uncertainty lines (95% confidence)
    plt.fill_between(x=px, y1=lb, y2=ub, color='k', alpha=0.1)

    plt.scatter(xdata, ydata, s=20, marker='x', color='black', label='rSARS-CoV1')
    plt.grid(True, alpha=0.5)

    plt.ylabel('Response (Illness)')
    plt.ylim([0, 1])
    plt.xlabel(r'\$N_v\$ (PFU)')
    plt.xscale('log')
    plt.xlim([1, 1e4])
    plt.legend(loc='best', frameon=False)

    # save and show figure
    plt.tight_layout()

    plt.xticks([1e0, 1e1, 1e2, 1e3, 1e4],
    [r'\$10^{0}\$', r'\$10^{1}\$', r'\$10^{2}\$', r'\$10^{3}\$', r'\$10^{4}\$',])
    fname = 'response_curve'
    f_loc =  '/Users/Tom/Box/NCS Project/models/figures/'
    savepdf_tex(fig=fig, fig_loc=f_loc, name=fname)
    # plt.show()
    plt.close()
# # Fit the function to the data

# k = popt[0]
# print('Optimal Values')
# print('k: ' + str(k))

# n = len(ydata)
# # compute r^2
# r2 = 1.0-(sum((ydata-func(xdata,k))**2)/((n-1.0)*np.var(ydata,ddof=1)))
# print('R^2: ' + str(r2))

# # calculate parameter confidence interval
# a = unc.correlated_values(popt, pcov)
# print('Uncertainty')
# print('a: ' + str(a))

# px = np.logspace(-1, 4, 500)
# py = 1 - unp.exp(-px/a)
# nom = unp.nominal_values(py)
# std = unp.std_devs(py)
# lb = nom - 1.96 * std
# ub = nom + 1.96 * std