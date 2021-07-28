import numpy as np
from numpy.core.arrayprint import format_float_scientific 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from response_model import get_prob_CI, get_k_CI
from savepdf_tex import savepdf_tex

def viral_load_in_room_with_time(t, y, Nv_gen, kappa, ACH, lam, inhale_q, room_V, t_stop_Nv_gen):
    Nv_gen = Nv_gen if t < t_stop_Nv_gen else 0
    Nv_s = y[0]
    Nv = y[1]
    Nv_s_dt = Nv_gen - (lam + kappa + ACH)*Nv_s
    Nv_dt = Nv_s*inhale_q/room_V
    return [Nv_s_dt, Nv_dt]

def get_Nv_direct(p, lam, kappa, ACH, V, Nv_gen, t):
    k= lam + kappa + ACH
    return t, p* Nv_gen/(k*V) * (t + (np.exp(-k*t)-1)/k)
# def viral_load_inhaled_with_time(t, y , Nvs)

def P_wells_riley(p, Q, q, t, t_stop):
    I = np.where(teval < t_stop, 1, 0)
    return 1 - np.exp(-I*p*q*t/Q)


kappa_speaking = 0.39 / 60**2 #  steady state gravitational settling rate [sec^-1] (de-oliviera et al)
kappa_coughing = 0.13 / 60**2# steady state gravitational settling rate [sec^-1] (de-oliviera et al)
virus_decay_rate = 0.636 / 60**2 # [sec^-1]
ACH = np.array([1, 5, 15]) / 60**2 # [sec -1]
kp = 4.1e2 # response constant for exponential model
k_lb, k_ub = get_k_CI()
occupant_inhaling_flow_rate = 0.521e-3 # [m3/sec]

room_volume = 200 # [m^3]
room_height = 2.5 # [m]
room_floor_area = room_volume / room_height

Nv_gen = 0.5  # PFU/sec
background_velocity = ACH * room_volume / (room_floor_area * 60**2) # [m s-1]


# wells-riley
quanta_constant = 0.001
t_0 = 0
t_stop_generation = 1 * 60**2 # 1 hour
t_end = 2 * 60**2 #  2 hours
teval = np.linspace(t_0,t_end,100)
fig, ax = plt.subplots(2,1, figsize=(6,12), sharex=True)
for i, ACH_i in enumerate(ACH):
    ventilation_flow_rate = ACH_i * room_volume
    P_WR = P_wells_riley(p=occupant_inhaling_flow_rate,
                         Q=ACH_i * room_volume,
                         q=quanta_constant,
                         t=teval,
                         t_stop=t_stop_generation)

    # t_direct, Nv_direct = get_Nv_direct(p=occupant_inhaling_flow_rate,
    #           lam=virus_decay_rate,
    #           kappa=kappa_speaking,
    #           ACH=ACH_i,
    #           V=room_volume,
    #           Nv_gen=Nv_gen,
    #           t=teval)
    # P_direct = 1 - np.exp(-Nv_direct/kp)
    # breakpoint()
    
    # plt.plot(t_direct, P_direct, ls=':', color=f'C{i}')
    

    Nv_s_solution = solve_ivp(fun=viral_load_in_room_with_time,
                              t_span=[t_0,t_end],
                              t_eval=teval,
                              y0=[0,0],
                              method='RK45',
                              args=(Nv_gen, kappa_speaking, ACH_i, virus_decay_rate, occupant_inhaling_flow_rate, room_volume, t_stop_generation))
    if Nv_s_solution.success:
        t = Nv_s_solution.t # in seconds
        Nv_s = Nv_s_solution.y[0] # PFUs per volume
        Nv = Nv_s_solution.y[1] # PFUs inhaled
        P = 1 - np.exp(-Nv/kp)
        P_lb, P_ub = get_prob_CI(Nv=Nv)

        quanta = ventilation_flow_rate* Nv/ (occupant_inhaling_flow_rate *  kp * teval)
        quanta_lb = ventilation_flow_rate* Nv/ (occupant_inhaling_flow_rate *  k_lb * teval)
        quanta_ub = ventilation_flow_rate* Nv/ (occupant_inhaling_flow_rate *  k_ub * teval)
        ax[0].plot(t, P, label=f'ACH: {ACH_i*60**2}', color=f'C{i}')
        ax[0].fill_between(x=t, y1=P_lb, y2=P_ub, color=f'C{i}', alpha=0.1)
        # ax[0].plot(t, P_lb, color=f'C{i}', ls=':')
        # ax[0].plot(t, P_ub, color=f'C{i}', ls=':')

        ax[1].plot(t, quanta, label=f'ACH: {ACH_i*60**2}', color=f'C{i}')
        ax[1].fill_between(x=t, y1=quanta_lb, y2=quanta_ub, color=f'C{i}', alpha=0.1)

        # ax[1].plot(t, quanta_lb, ls=':', color=f'C{i}')
        # ax[1].plot(t, quanta_ub, ls=':', color=f'C{i}')

ax[0].set_xlabel('time [s]')
ax[1].set_xlabel('time [s]')
ax[0].set_ylabel('Risk of infection P (%)')
ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%'))
ax[1].set_ylabel(r'\$Iq\$ quanta generation rate')

ax[0].legend(loc='upper left', frameon=False)
ax[1].legend(frameon=False)
# plt.show()
f_loc = '/Users/Tom/Box/NCS Project/models/figures/'
fname = 'simple_dose_response_model'
savepdf_tex(fig=fig, fig_loc=f_loc, name=fname)
plt.close()