"""PLot comparing the steady state quanta concentration to the transient behaviour.

    """
from datetime import time, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from classes.contam_model import ContamModel
from classes.weather import Weather
from savepdf_tex import savepdf_tex

def solve(t, C, *args):
    Q , quanta, I, volume = args
    dC = quanta*I + np.matmul(Q, C)
    return dC/volume

if __name__ == '__main__':

    save=True
    init_room_idx = 0
    event_time = 1.5
    end_time = 4
    wind_dir = 0.0
    wind_speed = 5.0
    time_to_get_results = timedelta(days=5)
    contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                            'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                            'name': 'school_corridor'}

    # init contam model
    contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
            contam_dir=contam_model_details['prj_dir'],
            project_name=contam_model_details['name'])






    simulation_constants = {'duration': timedelta(days=7),
                            'mask_efficiency': 0,
                            'lambda_home': 0/898.2e4 / 24, # [hr^-1] should be quanta hr^-1? h
                            # 'lambda_home': 1, # [hr^-1] should be quanta hr^-1
                            'pulmonary_vent_rate': 0.54, # [m3.hr^-1]
                            'quanta_gen_rate': 5, # [quanta.hr^-1]
                            'recover_rate': (8.2*24)**(-1), # [hr^-1]
                            'school_start': time(hour=9),
                            'school_end': time(hour=16),
                            'no_of_simulations': 1000,
                            'init_students_per_class': 30,
                            'plotting_sample_rate': '5min',
                            'door_open_fraction': 1.0,
                            'window_open_fraction': 1.0,
                            }

    weather_params = Weather(wind_speed=wind_speed,
                             wind_direction=wind_dir,
                             ambient_temp=10.0)
    contam_model.set_initial_settings(weather_params, window_height=1.0)
    # run contam simulation
    contam_model.run_simulation(verbose=True)
    ventilation_rates_open = contam_model.vent_mat



    number_of_rooms = 11
    quanta = 5

    contam_model.set_all_flow_paths_of_type_to(search_type_term='oor',
                                                param_dict={'type': 3},
                                                rerun=True,
                                                )
    ventilation_rates_closed = contam_model.vent_mat


    contam_model.set_all_flow_paths_of_type_to(search_type_term='oor',
                                                param_dict={'type': 4},
                                                rerun=True,
                                                )

    zone_volumes = contam_model.zones.df['Vol'].astype(float)
    C0 = np.array([0]*len(zone_volumes))
    # for i in range(number_of_rooms):
    infected_people = [0]*(number_of_rooms-1)
    infected_people.insert(init_room_idx, 1)
    I = np.array(infected_people)
    soln_door_open = solve_ivp(fun=solve,
                     y0=C0,
                     t_span=[0,event_time],
                    #  t_eval=np.linspace(0,event_time, 10),
                    args=(ventilation_rates_open, quanta, I, zone_volumes)
                    )
    C0 = soln_door_open.y[:,-1]
    soln_door_closed = solve_ivp(fun=solve,
                     y0=C0,
                     t_span=[event_time, end_time],
                     t_eval=np.linspace(event_time, end_time, 25),
                    args=(ventilation_rates_closed, quanta, I, zone_volumes)
                    )
    full_solution = np.concatenate((soln_door_open.y[:,:-1], soln_door_closed.y), axis=1)
    full_time = np.concatenate((soln_door_open.t[:-1], soln_door_closed.t), axis=0)

    open_steady_state = np.linalg.solve(ventilation_rates_open, -quanta*I)
    closed_steady_state = np.linalg.solve(ventilation_rates_closed, -quanta*I)
    t_ss = [0, event_time, end_time]
    
    # for j , y in enumerate(soln.y):
    fig, ax = plt.subplots(1,1,figsize=(10,4), sharex=True)
    ax.plot(full_time, full_solution[init_room_idx,:],
                color='C0', label=contam_model.zones.df.loc[init_room_idx, "name"])
    ax.step(t_ss, [0, open_steady_state[init_room_idx], closed_steady_state[init_room_idx]],
                where='pre', color='C0', ls='--',)
    # ax[1].plot(full_time, full_solution[5,:], color='C1', label=contam_model.zones.df.loc[5, 'name'])
    # ax[1].plot(full_time, full_solution[1,:], color='C2', label=contam_model.zones.df.loc[1, 'name'])
    # ax[1].plot(full_time, full_solution[10,:],  color='C3', label=contam_model.zones.df.loc[10, 'name'])
    # ax[1].step(t_ss, [0, open_steady_state[5], closed_steady_state[5]],
                # where='pre', color='C1', ls='--',)
    # ax[1].step(t_ss, [0, open_steady_state[1], closed_steady_state[1]],
                # where='pre', color='C2', ls='--',)
    # ax[1].step(t_ss, [0, open_steady_state[10], closed_steady_state[10]],
                # where='pre', color='C3', ls='--',)
    plt.suptitle(f'Initial infection in {contam_model.zones.df.loc[init_room_idx, "name"]}')
    # for i in range(2):
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    ax.set_ylabel(r'\$C_i\$', labelpad=20)
    ax.set_xlim([0, end_time])
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right', frameon=False, bbox_to_anchor=(0.92,0.025))
    ax.axvspan(0, event_time, color='k', alpha=0.1)
    ax.set_xlabel('Time [hrs]', labelpad=20)
        
    midpoint_y_axis = np.mean(ax.get_ylim())*0.66
    ax.text(x=event_time*0.99, y=midpoint_y_axis, s='Doors open', ha='right')
    ax.text(x=event_time*1.01, y=midpoint_y_axis, s='Doors closed', ha='left')
    # midpoint_y_axis = np.mean(ax[1].get_ylim())*1.5
    # ax[1].text(x=event_time*0.99, y=midpoint_y_axis, s='Doors open', ha='right')
    # ax[1].text(x=event_time*1.01, y=midpoint_y_axis, s='Doors closed', ha='left')

    plt.tight_layout()





    
   



    if save:
        save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
        savepdf_tex(fig=fig, fig_loc=save_loc,
                    name=f'steady_state_assumption_door_closing')


    else:
        plt.show()
    plt.close()


