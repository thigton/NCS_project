"""Run script: basic

    """
from datetime import time, timedelta

import matplotlib.pyplot as plt

from classes.contam_model import ContamModel
from classes.stocastic_model import StocasticModel
from classes.weather import Weather
time_to_get_results = timedelta(days=5)

contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                        'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                        'name': 'school_corridor'}

# init contam model
contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
        contam_dir=contam_model_details['prj_dir'],
        project_name=contam_model_details['name'])
door_open_fraction = [0.25, 0.5, 0.75, 1.0]
fig, ax = plt.subplots(1,1)

for door in door_open_fraction:
    time_step = timedelta(minutes=10)
    simulation_constants = {'duration': timedelta(days=5),
                            'mask_efficiency': 0,
                            'lambda_home': 0/898.2e4 / 24, # [hr^-1] should be quanta hr^-1? h
                            # 'lambda_home': 1, # [hr^-1] should be quanta hr^-1
                            'pulmonary_vent_rate': 0.54, # [m3.hr^-1]
                            'quanta_gen_rate': 5, # [quanta.hr^-1]
                            'recover_rate': (6.0*24)**(-1), # [hr^-1]
                            'school_start': time(hour=9),
                            'school_end': time(hour=16),
                            'no_of_simulations': 200,
                            'init_students_per_class': 30,
                            'plotting_sample_rate': f'{int(time_step.seconds/60)}min',
                            'time_step': time_step,
                            'door_open_fraction': door,
                            'window_open_fraction': 1.0,
                            }
    weather_params = Weather(wind_speed=5.0, wind_direction=0.0, ambient_temp=10.0)
    contam_model.set_initial_settings(weather_params, window_height=1.0)
    contam_model.load_all_vent_matrices()
    # run contam simulation
    contam_model.run_simulation()
    model = StocasticModel(weather=weather_params,
                           contam_details=contam_model_details,
                           simulation_constants=simulation_constants,
                           contam_model=contam_model,
                           opening_method='internal_doors_only_random',# internal_doors_only_random, all_doors_only_random
                           movement_method=None, # change_rooms_in_group
                           method='DTMC',
                           )
    model.run(results_to_track=['S_df','I_df','R_df', 'risk', 'first_infection_group'],
              parallel=True,
              )
    I_in_init_room = model.percentage_of_infection_in_initial_room()
    model.get_ventilation_rates_from_door_open_df_retropectively()
    model.get_risk_at_time_x_in_classrooms(time=time_to_get_results)

    model.plot_risk_vs_ave_fresh_ventilation(ax=ax, compare='door_open_fraction', incl_init_group=False)


plt.legend()
plt.show()
plt.close()
