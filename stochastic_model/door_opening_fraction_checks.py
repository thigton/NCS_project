import itertools
from datetime import time, timedelta

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

sns.set_theme(style="whitegrid")
import pandas as pd

from classes.contam_model import ContamModel
from classes.stocastic_model import StocasticModel
from classes.weather import Weather
from savepdf_tex import savepdf_tex

plot_time_series = False
save=True
wind_dir = 0.0
door_opening_fraction = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
wind_speed = 0.0
time_to_get_results = timedelta(days=5)
contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                        'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                        'name': 'school_corridor'}

# init contam model
contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
        contam_dir=contam_model_details['prj_dir'],
        project_name=contam_model_details['name'])


col_names = pd.MultiIndex.from_product([door_opening_fraction,
                                        ['total','first room']],
                                        names=['door open', 'grouping'])



sim_duration = [timedelta(days=7), timedelta(days=182)]
fig3, ax3 = plt.subplots(len(sim_duration),1, figsize=(12,10))
for i, duration in enumerate(sim_duration):
    for ii, door in enumerate(door_opening_fraction):
        simulation_constants = {'duration': duration,
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
                                'time_step': '5min',
                                'door_open_fraction': door,
                                'window_open_fraction': 1.0,
                                }
        weather_params = Weather(wind_speed=wind_speed, wind_direction=wind_dir, ambient_temp=10.0)
        contam_model.set_initial_settings(weather_params, window_height=1.0)
        # run contam simulation
        contam_model.run_simulation(verbose=True)
        model = StocasticModel(weather=weather_params,
                               contam_details=contam_model_details,
                               simulation_constants=simulation_constants,
                               contam_model=contam_model,
                               closing_opening_type='door',
                               )
        model.run(results_to_track=['risk', 'first_infection_group'])

        ax3[i].hist(model.door_open_fraction_actual, bins=50, density=True, color=f'C{ii}', alpha=0.40, label=r'\$\Gamma={}\$'.format(door))
    
    ax3[i].set_title(f'Simulation duration : {duration.days} days')
    ax3[i].set_xlim(0,1)
    ax3[i].set_ylim(0,10)
    ax3[i].legend()



if save:
    save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
    # savepdf_tex(fig=fig1, fig_loc=save_loc,
    #             name=f'door_analysis_full_{time_to_get_results.days}d_{str(wind_dir).replace(".","-")}deg')
    # savepdf_tex(fig=fig2, fig_loc=save_loc,
    #             name=f'door_analysis_init_room_{time_to_get_results.days}d_{str(wind_dir).replace(".","-")}deg')
    savepdf_tex(fig=fig3, fig_loc=save_loc,
                name=f'door_opening_dist')

else:
    plt.show()
plt.close()


