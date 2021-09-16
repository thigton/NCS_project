from datetime import time, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm, inv
from classes.contam_model import ContamModel
from classes.weather import Weather
from savepdf_tex import savepdf_tex

def solve(t, C, *args):
    Q , quanta, I, volume = args
    dC = quanta*I + np.matmul(Q, C)
    return dC/volume

if __name__ == '__main__':

    save=False
    init_room_idx = 6
    second_room_idx = 2
    event_time = 1.5
    end_time = 4
    wind_dir = 0.0
    wind_speed = 5.0
    number_of_rooms = 11
    quanta = 5
    p = 0.54
    contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                            'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                            'name': 'school_corridor'}

    # init contam model
    contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
            contam_dir=contam_model_details['prj_dir'],
            project_name=contam_model_details['name'])


    weather_params = Weather(wind_speed=wind_speed, wind_direction=wind_dir, ambient_temp=10.0)
    contam_model.set_initial_settings(weather_params, window_height=1.0)
    # run contam simulation
    contam_model.run_simulation(verbose=True)
    ventilation_rates_open = contam_model.vent_mat

    zone_volumes = contam_model.zones.df['Vol'].astype(float)
    C0 = np.array([0]*len(zone_volumes))
    infected_people = [0]*(number_of_rooms-1)
    infected_people.insert(init_room_idx, 1)
    I = np.array(infected_people)
    S = np.array([30, 30, 30, 30, 30, 0, 30, 30, 30, 30, 30])
    S[init_room_idx] = 29
    soln_door_1 = solve_ivp(fun=solve,
                     y0=C0,
                     t_span=[0,event_time],
                    #  t_eval=np.linspace(0,event_time, 20),
                    args=(ventilation_rates_open, quanta, I, zone_volumes)
                    )
    R_k_1 = np.sum(soln_door_1.y* S[:,None] * p, axis=0)
    P_ni_1 = soln_door_1.y* S[:,None] * p / R_k_1
    steady_state_1 = np.linalg.solve(ventilation_rates_open, -quanta*I)
    R_k_ss_1 = np.sum(steady_state_1*S*p)
    P_ni_ss_1 = steady_state_1*S*p / R_k_ss_1

    t_exact = np.linspace(0, event_time, 20)
    V = np.diag(zone_volumes.values)
    Q = ventilation_rates_open
    A = Q * inv(V)
    B = inv(V)
    u = quanta * I
    Ci=C0
    exact_1 = []
    for t in t_exact:
        Ci = expm(A*t).dot(Ci) + (expm(A*t) - np.identity(number_of_rooms)).dot(inv(A)).dot(B).dot(u)
        exact_1.append(Ci)
    breakpoint()

    C0 = soln_door_1.y[:,-1]
    infected_people = [0]*(number_of_rooms-1)
    infected_people.insert(second_room_idx, 1)
    print(I)

    I = np.array(infected_people)
    print(I)
    S = np.array([30, 30, 30, 30, 30, 0, 30, 30, 30, 30, 30])
    S[second_room_idx] = 29

    soln_door_2 = solve_ivp(fun=solve,
                     y0=C0,
                     t_span=[event_time, end_time],
                    #  t_eval=np.linspace(event_time, end_time, 50),
                    args=(ventilation_rates_open, quanta, I, zone_volumes)
                    )
    R_k_2 = np.sum(soln_door_2.y* S[:,None] * p, axis=0)
    P_ni_2 = soln_door_2.y* S[:,None] * p / R_k_2
    steady_state_2 = np.linalg.solve(ventilation_rates_open, -quanta*I)

    R_k_ss_2 = np.sum(steady_state_2*S*p)
    P_ni_ss_2 = steady_state_2*S*p / R_k_ss_2

    
    full_solution = np.concatenate((soln_door_1.y[:,:-1], soln_door_2.y), axis=1)
    full_time = np.concatenate((soln_door_1.t[:-1], soln_door_2.t), axis=0)
    
    t_ss = [0, event_time, end_time]
    
    # for j , y in enumerate(soln.y):
    fig, ax = plt.subplots(2,1,figsize=(10,7))
    ax[0].plot(full_time, full_solution[init_room_idx,:],
                color='C0', label=contam_model.zones.df.loc[init_room_idx, 'name'])
    ax[0].plot(full_time, full_solution[second_room_idx,:],
                color='C1', label=contam_model.zones.df.loc[second_room_idx, 'name'])

    ax[0].step(t_ss, [0, steady_state_1[init_room_idx], steady_state_2[init_room_idx]],
                where='pre', color='C0', ls='--',)
    ax[0].step(t_ss, [0, steady_state_1[second_room_idx], steady_state_2[second_room_idx]],
                where='pre', color='C1', ls='--',)

    
    ax[1].plot(full_time, full_solution[5,:], color='C1', label=contam_model.zones.df.loc[5, 'name'])
    ax[1].plot(full_time, full_solution[1,:], color='C2', label=contam_model.zones.df.loc[1, 'name'])
    ax[1].plot(full_time, full_solution[10,:],  color='C3', label=contam_model.zones.df.loc[10, 'name'])


    ax[1].step(t_ss, [0, steady_state_1[5], steady_state_2[5]],
                where='pre', color='C1', ls='--',)
    ax[1].step(t_ss, [0, steady_state_1[1], steady_state_2[1]],
                where='pre', color='C2', ls='--',)
    ax[1].step(t_ss, [0, steady_state_1[10], steady_state_2[10]],
                where='pre', color='C3', ls='--',)
    plt.suptitle(f'Initial infection in {contam_model.zones.df.loc[init_room_idx, "name"]}')
    for i in range(2):
        ax[i].ticklabel_format(axis="both", style="sci", scilimits=(0,0))
        ax[i].set_xlabel('Time [hrs]')
        ax[i].set_ylabel(r'\$C_i\$')
        ax[i].legend(loc='upper right')
        ax[i].axvspan(0, event_time, color='k', alpha=0.1)
        midpoint_y_axis = np.mean(ax[i].get_ylim())
        ax[i].text(x=event_time/2, y=midpoint_y_axis, s=f'infection {contam_model.zones.df.loc[init_room_idx, "name"]}', ha='center')
        ax[i].text(x=(event_time+end_time)/2, y=midpoint_y_axis, s=f'infection {contam_model.zones.df.loc[second_room_idx, "name"]}', ha='center')
    plt.tight_layout()


    if save:
        save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
        savepdf_tex(fig=fig, fig_loc=save_loc,
                    name=f'steady_state_assumption_redistribute')
    else:
        plt.show()
    plt.close()

    ###### fig 2 
    fig, ax = plt.subplots(3,1,figsize=(10,10.5))
    ax[0].plot(full_time, np.concatenate((R_k_1[:-1], R_k_2), axis=0),
                color='C0', label='R_k trans')
    ax[0].step(t_ss, [0, R_k_ss_1, R_k_ss_2],
                where='pre', color='C0', ls='--', label='ss')

    ax[0].legend()
    ax[0].set_xlabel('time [hrs]')
    ax[0].set_ylabel('R_k')

    full = np.concatenate((P_ni_1[:,:-1], P_ni_2), axis=1)

    ax[1].plot(full_time, full[init_room_idx,:],
                color='C0', label=contam_model.zones.df.loc[init_room_idx, 'name'])
    ax[1].plot(full_time, full[second_room_idx,:],
                color='C1', label=contam_model.zones.df.loc[second_room_idx, 'name'])

    ax[1].step(t_ss, [0, P_ni_ss_1[init_room_idx], P_ni_ss_2[init_room_idx]],
                where='pre', color='C0', ls='--',)
    ax[1].step(t_ss, [0, P_ni_ss_1[second_room_idx], P_ni_ss_2[second_room_idx]],
                where='pre', color='C1', ls='--',)

    
    ax[2].plot(full_time, full[5,:], color='C1', label=contam_model.zones.df.loc[5, 'name'])
    ax[2].plot(full_time, full[1,:], color='C2', label=contam_model.zones.df.loc[1, 'name'])
    ax[2].plot(full_time, full[10,:],  color='C3', label=contam_model.zones.df.loc[10, 'name'])


    ax[2].step(t_ss, [0, P_ni_ss_1[5], P_ni_ss_2[5]],
                where='pre', color='C1', ls='--',)
    ax[2].step(t_ss, [0, P_ni_ss_1[1], P_ni_ss_2[1]],
                where='pre', color='C2', ls='--',)
    ax[2].step(t_ss, [0, P_ni_ss_1[10], P_ni_ss_2[10]],
                where='pre', color='C3', ls='--',)

    ax[1].set_ylabel(r'\$P_{n,i}\$')
    ax[2].set_ylabel(r'\$P_{n,i}\$')
    
    for i in range(3):
        ax[i].ticklabel_format(axis="both", style="sci", scilimits=(0,0))
        ax[i].set_xlabel('Time [hrs]')
        ax[i].legend(loc='upper right')
        ax[i].axvspan(0, event_time, color='k', alpha=0.1)
        midpoint_y_axis = np.mean(ax[i].get_ylim())
        ax[i].text(x=event_time/2, y=midpoint_y_axis, s=f'infection {contam_model.zones.df.loc[init_room_idx, "name"]}', ha='center')
        ax[i].text(x=(event_time+end_time)/2, y=midpoint_y_axis, s=f'infection {contam_model.zones.df.loc[second_room_idx, "name"]}', ha='center')
    plt.show()
    plt.close()

