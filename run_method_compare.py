"""Run script: to compare the methods.
    """
from datetime import time, timedelta

import matplotlib.pyplot as plt
import os
from classes.contam_model import ContamModel
from classes.stocastic_model import StocasticModel
from classes.weather import Weather

contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                        'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                        'name': 'school_corridor'}
# contam_model_details = {'exe_dir': f'{os.path.dirname(os.path.realpath(__file__))}/contam/',
#                         'prj_dir': f'{os.path.dirname(os.path.realpath(__file__))}/contam_files/',
#                         'name': 'school_corridor'}

# init contam model
contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
        contam_dir=contam_model_details['prj_dir'],
        project_name=contam_model_details['name'])

# fig, ax = plt.subplots(1,1)


simulation_constants = {'duration': timedelta(days=7),
                        'mask_efficiency': 0,
                        'lambda_home': 0/898.2e4 / 24, # [hr^-1] should be quanta hr^-1? h
                        # 'lambda_home': 1, # [hr^-1] should be quanta hr^-1
                        'pulmonary_vent_rate': 0.54, # [m3.hr^-1]
                        'quanta_gen_rate': 5, # [quanta.hr^-1]
                        'recover_rate': (6.0*24)**(-1), # [hr^-1]
                        'school_start': time(hour=9),
                        'school_end': time(hour=16),
                        'no_of_simulations': 2000,
                        'init_students_per_class': 30,
                        'plotting_sample_rate': '5min',
                        'time_step': timedelta(minutes=5),
                        'door_open_fraction': 1.0,
                        'window_open_fraction': 1.0,
                        }
weather_params = Weather(wind_speed=10.0, wind_direction=0.0, ambient_temp=10.0)

fig, ax  = plt.subplots(1,1)
for method in ['CTMC', 'DTMC']:
        contam_model.set_initial_settings(weather_params, window_height=1.0)
        # run contam simulation
        contam_model.run_simulation()
        model = StocasticModel(weather=weather_params,
                               contam_details=contam_model_details,
                               simulation_constants=simulation_constants,
                               contam_model=contam_model,
                               opening_method='all_doors_only_random',
                               method=method,
                               )
        model.run(results_to_track=['S_df','I_df','R_df'])
        model.plot_SIR(ax=ax,ls='-', comparison='method', value=method)

plt.legend()
plt.show()
plt.close()
