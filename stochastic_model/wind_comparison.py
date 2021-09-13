from datetime import time, timedelta
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
import pandas as pd

from classes.contam_model import ContamModel
from classes.stocastic_model import StocasticModel
from classes.weather import Weather
plot_time_series = False
wind_speed = [0.0, 5.0, 10.0, 15.0]
wind_direction = [0.0, 45.0, 90.0]
time_to_get_results = timedelta(days=5)
contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                        'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                        'name': 'school_corridor'}

# init contam model
contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
        contam_dir=contam_model_details['prj_dir'],
        project_name=contam_model_details['name'])

if plot_time_series:    
    fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=True)
col_names = pd.MultiIndex.from_product([[f'{x}deg' for x in wind_direction],
                                        [f'{x}kmph' for x in wind_speed]],
                                        names=['wind direction', 'wind speed'])
box_plot_df = pd.DataFrame(columns=col_names)
for speed, direction in itertools.product(wind_speed, wind_direction):
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
                            'door_open_fraction': 0.2,
                            'window_open_fraction': 1.0,
                            }

    weather_params = Weather(wind_speed=speed, wind_direction=direction, ambient_temp=10.0)

    contam_model.set_initial_settings(weather_params, window_height=1.0)
    # run contam simulation
    contam_model.run_simulation()

    model = StocasticModel(weather=weather_params,
                           contam_details=contam_model_details,
                           simulation_constants=simulation_constants,
                           contam_model=contam_model,
                           )
    model.run(results_to_track=['risk', 'first_infection_group'])
    if plot_time_series:
        model.plot_risk(ax=ax[0], comparison='wind_speed', value=speed)
        model.plot_risk(ax=ax[1], comparison='wind_speed', value=speed, first_infection_group=True)
    box_plot_df[(f'{direction}deg',f'{speed}kmph')] = model.get_risk_at_time(time_to_get_results).rename()

if plot_time_series:
    fig.autofmt_xdate(rotation=45)
    ax[0].set_ylabel('Infection Risk')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

fig1, ax1 = plt.subplots(1,1)
fig1.autofmt_xdate(rotation=45)
# box_plot_df.boxplot()
sns.boxplot(x="wind speed", y="value", hue='wind direction', data=box_plot_df.melt(), ax=ax1)
plt.show()
plt.close()