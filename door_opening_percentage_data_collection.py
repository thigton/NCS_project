"""Run script: looking into door opening percentage 
"""

# pylint: disable=no-member
from datetime import time, timedelta, datetime

import pandas as pd
import numpy as np

from classes.contam_model import ContamModel
from classes.stocastic_model import StocasticModel
from classes.weather import Weather
import pickle
import os
import util.util_funcs as uf

corridor_temp = 19.0
model_log = uf.load_model_log(fname=f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv')
save=True
wind_dir = 90.0
door_opening_fraction = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
wind_speeds = [5.0, 15.0,]
amb_temps = [5.0, 10.0]
time_to_get_results = timedelta(days=5)
# contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
#                         'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
#                         'name': 'school_corridor'}
contam_model_details = {'exe_dir': f'{os.path.dirname(os.path.realpath(__file__))}/contam/',
                        'prj_dir': f'{os.path.dirname(os.path.realpath(__file__))}/contam_files/',
                        'name': 'school_corridor'}

# init contam model
contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
        contam_dir=contam_model_details['prj_dir'],
        project_name=contam_model_details['name'])

time_step = timedelta(minutes=10)
for amb_temp in amb_temps:
    for i, wind_speed in enumerate(wind_speeds):

        for ii, door in enumerate(door_opening_fraction):
            print(f'Model starting: \nAmbient Temp: {amb_temp} C \nWind speed {wind_speed} kph \nWind direction {wind_dir}')
            print(f'Door fraction: {door}')
            simulation_constants = {'duration': timedelta(days=5),
                                    'mask_efficiency': 0,
                                    'lambda_home': 0/898.2e4 / 24, # [hr^-1] should be quanta hr^-1? h
                                    'pulmonary_vent_rate': 0.54, # [m3.hr^-1]
                                    'quanta_gen_rate': 25, # [quanta.hr^-1]
                                    'recover_rate': (6*24)**(-1), # [hr^-1]
                                    'school_start': time(hour=9),
                                    'school_end': time(hour=16),
                                    'no_of_simulations': 2000,
                                    'init_students_per_class': 30,
                                    'plotting_sample_rate': f'{int(time_step.seconds/60)}min',
                                    'time_step': time_step,
                                    'door_open_fraction': door,
                                    'window_open_fraction': 1.0,
                                    }
            weather_params = Weather(wind_speed=wind_speed, wind_direction=wind_dir, ambient_temp=amb_temp)

            contam_model.set_initial_settings(weather_params, window_height=1.0)
            # corridor = contam_model.get_all_zones_of_type(
            # search_type_term='corridor')
            # temp = corridor_temp - door*(corridor_temp - amb_temp)
            # for zone in corridor['Z#'].values:
            #     contam_model.set_zone_temperature(zone, temp)
            # contam_model.generate_all_ventilation_matrices_for_all_door_open_close_combination()
            contam_model.load_all_vent_matrices()
            # if not contam_model.all_door_matrices:
                # run contam simulation
            # contam_model.run_simulation(verbose=True)
            model = StocasticModel(weather=weather_params,
                                   contam_details=contam_model_details,
                                   simulation_constants=simulation_constants,
                                   contam_model=contam_model,
                                   opening_method='internal_doors_only_random',# internal_doors_only_random, all_doors_only_random
                                   movement_method=None,
                                   method='DTMC',
                                   )
            model.run(results_to_track=['S_df','I_df','R_df','risk', 'first_infection_group'],
                        parallel=True,
                        )

            if save:
                fname = f'model_{datetime.now().strftime("%y%m%d_%H-%M-%S")}'
                model_log = uf.add_model_to_log(model, model_log, fname)
                 # change cwd to file location
                with open(f'{os.path.dirname(os.path.realpath(__file__))}/results/{fname}.pickle', 'wb') as pickle_out:
                    pickle.dump(model, pickle_out)
                model_log.to_csv(f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv')
            del model

