import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def master_equation(t, y, km, gam_m, p_bounds):
    print(f'time: {t:0.2f} minutes', end='\r')
    lb = p_bounds[0]
    ub = p_bounds[1]
    diff_eqs = []
    try:
        for m in range(lb, ub):
            if m == lb:

                dPmdt = gam_m*(m+1)*y[m-lb+1] - km*y[m-lb]
            elif m == ub-1:
                dPmdt = km * y[m-lb-1] - gam_m*m*y[m-lb]
            else:
                dPmdt = km * y[m-lb-1] + gam_m * (m+1) * y[m-lb+1] - (km + gam_m*m) * y[m-lb]
            diff_eqs.append(dPmdt)
    except IndexError:
        print('ERROR')
        breakpoint()
    return diff_eqs

if __name__ == '__main__':
    # m_dot = km - gam_m*m
    km = 50
    gam_m = 0.5
    m0 = [50, 75]
    p_bounds = [35, 165]
    t_bounds = [0, 10]
    p_vals = np.arange(p_bounds[0], p_bounds[1])
    m0_idx = np.where(np.isin(p_vals, m0))[0]
    y0 = np.zeros(shape=(p_bounds[1]-p_bounds[0]))
    y0[m0_idx] = 1/len(m0)
    soln = solve_ivp(fun=master_equation,
                     t_span=t_bounds,
                     y0=y0,
                     t_eval=np.linspace(t_bounds[0], t_bounds[1], 10),
                     args=(km, gam_m, p_bounds))
    t = soln.t

    for i in range(len(t)):
        plt.plot(p_vals, soln.y[:,i], label=f't={t[i]}')
    plt.ylim([0,0.08])
    plt.legend()
    plt.show()
    plt.close()