""" run scipt for timestep comparison for different DTMC simulations
"""
from datetime import time, timedelta, datetime
import os
import matplotlib.pyplot as plt
import pickle
from classes.contam_model import ContamModel
from classes.stocastic_model import StocasticModel
from classes.weather import Weather
# from savepdf_tex import savepdf_tex
import util.util_funcs as uf
import gc
import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


save=True
model_log = uf.load_model_log(fname=f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv')

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



simulation_constants = {'duration': timedelta(days=5),
                        'mask_efficiency': 0,
                        'lambda_home': 0/898.2e4 / 24, # [hr^-1] should be quanta hr^-1? h
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
weather_params = Weather(wind_speed=5.0, wind_direction=0.0, ambient_temp=10.0)
# fig, ax  = plt.subplots(1,1, figsize=(12,7))

contam_model.set_initial_settings(weather_params, window_height=1.0)
contam_model.load_all_vent_matrices()

# run contam simulation
# contam_model.run_simulation()
model = StocasticModel(weather=weather_params,
                       contam_details=contam_model_details,
                       simulation_constants=simulation_constants,
                       contam_model=contam_model,
                       opening_method='all_doors_only_random',
                       method='CTMC',
                       )
model.run(results_to_track=['S_df','I_df','R_df','risk', 'first_infection_group'],
            parallel=True,
            )

if save:
    fname = f'model_{datetime.now().strftime("%y%m%d_%H-%M-%S")}_CTMC'
    model_log = uf.add_model_to_log(model, model_log, fname)
     # change cwd to file location
    with open(f'{os.path.dirname(os.path.realpath(__file__))}/results/{fname}.pickle', 'wb') as pickle_out:
        pickle.dump(model, pickle_out)
    model_log.to_csv(f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv')
    del model

time_steps = [
              timedelta(minutes=30),
              timedelta(minutes=15),
              timedelta(minutes=10),
              timedelta(minutes=5),
              timedelta(minutes=2),
              ]

for time_step in time_steps:
    print(time_step)
    simulation_constants['time_step'] = time_step
    # contam_model.set_initial_settings(weather_params, window_height=1.0)
    # contam_model.run_simulation()
    model = StocasticModel(weather=weather_params,
                       contam_details=contam_model_details,
                       simulation_constants=simulation_constants,
                       contam_model=contam_model,
                       opening_method='all_doors_only_random',
                       method='DTMC',
                       )
    model.run(results_to_track=['S_df','I_df','R_df','risk', 'first_infection_group'],
            parallel=True,
            )
    
    if save:
        fname = f'model_{datetime.now().strftime("%y%m%d_%H-%M-%S")}_DTMC_ts_{time_step.total_seconds()/60}'
        model_log = uf.add_model_to_log(model, model_log, fname)
         # change cwd to file location
        with open(f'{os.path.dirname(os.path.realpath(__file__))}/results/{fname}.pickle', 'wb') as pickle_out:
            pickle.dump(model, pickle_out)
        model_log.to_csv(f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv')
    del model
    gc.collect()
    # model.plot_risk(ax=ax,lw=2, comparison=r'\$\Delta t\$',
    #                 value=f'{time_step.total_seconds()/60}mins')

# fig.autofmt_xdate(rotation=45)
# ax.set_ylabel('Infection Risk')
# ax.legend(loc='upper left', bbox_to_anchor=(1,1))
# plt.tight_layout()
# if save:
#     save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
#     savepdf_tex(fig=fig, fig_loc=save_loc,
#                 name=f'compare_stochastic_methods_and_time_steps_2')
# else:

#     plt.show()
#     plt.close()
    
    