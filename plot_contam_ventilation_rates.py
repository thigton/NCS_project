"""Run script to produce plots looking at how the ventilation rate changes in CONTAM
based on different environmental condtions.
"""
import itertools
from datetime import time, timedelta

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from classes.contam_model import ContamModel
from classes.weather import Weather
from savepdf_tex import savepdf_tex

plot_time_series = False
save=True
wind_dir = 0.0
wind_speeds = np.linspace(0.0, 20.0, 21)
amb_temp = np.array([5.0, 10.0])
windward_name = 'Classroom6'
leeward_name = 'Classroom1'
students_per_class = 30

scenarios = [5.0, 15.0]
contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                        'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                        'name': 'school_corridor'}

# init contam model
contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
        contam_dir=contam_model_details['prj_dir'],
        project_name=contam_model_details['name'])


fields = ['temp', 'wind_speed', 'Q_fresh', 'Q_total', 'location']
# df = pd.DataFrame(columns=)
series = []
for speed, temp in itertools.product(wind_speeds, amb_temp):

    print(speed, temp)
    weather_params = Weather(wind_speed=speed, wind_direction=wind_dir, ambient_temp=temp)
    contam_model.set_initial_settings(weather_params, window_height=1.0)
    # run contam simulation
    contam_model.run_simulation(verbose=False)
    Q = contam_model.get_ventilation_rate_for(windward_name)
    series.append(pd.Series(data=[temp, speed, Q[1], Q[0], 'windward'],
                            index=fields))
    Q = contam_model.get_ventilation_rate_for(leeward_name)
    series.append(pd.Series(data=[temp, speed, Q[1], Q[0], 'leeward'],
                            index=fields))




df = pd.concat(series, axis=1)
df = df.T
windward_df = df[df['location'] == 'windward']
leeward_df = df[df['location'] == 'leeward']

f, ax = plt.subplots(1, 1, sharey=True, figsize=(10,6))
temp_colours = {5.0: 'blue', 10.0: 'red'}
# for temp in amb_temp:
#     df = windward_df[windward_df['temp'] == temp]
#     ax[0].plot(df['wind_speed'], df['Q_total']/students_per_class,
#             color=temp_colours[temp], label=r'\${:0.0f}\degree C\$ windward classroom'.format(temp))
#     df = leeward_df[leeward_df['temp'] == temp]
#     ax[0].plot(df['wind_speed'], df['Q_total']/students_per_class,
#             color=temp_colours[temp], ls ='--',
#             label=r'\${:0.0f}\degree C\$ leeward classroom'.format(temp))
# ax[0].axhline(10, color='black', ls='--', lw=2)
# ax[0].set_ylim([0, 40])
# ax[0].set_xlabel('wind speed [kph]')
# ax[0].set_ylabel('ventilation rate [l/s/pp]')
# if save:
#     save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
#     savepdf_tex(fig=f, fig_loc=save_loc,
#                 name=f'total_ventilation_wd_{str(wind_dir).replace(".","_")}')
# else:
#     plt.show()
# plt.close()
# breakpoint()
ax.scatter(windward_df.loc[windward_df['wind_speed'].isin(scenarios), 'wind_speed'],
           windward_df.loc[windward_df['wind_speed'].isin(scenarios), 'Q_fresh']/students_per_class,
           color=[temp_colours[x] for x in windward_df.loc[windward_df['wind_speed'].isin(scenarios), 'temp'].values],
           marker='D', s=50)
ax.scatter(leeward_df.loc[leeward_df['wind_speed'].isin(scenarios), 'wind_speed'],
           leeward_df.loc[leeward_df['wind_speed'].isin(scenarios), 'Q_fresh']/students_per_class,
           color=[temp_colours[x] for x in leeward_df.loc[leeward_df['wind_speed'].isin(scenarios), 'temp'].values],
           marker='D', s=50)

for temp in amb_temp:
    df = windward_df[windward_df['temp'] == temp]
    ax.plot(df['wind_speed'], df['Q_fresh']/students_per_class,
            color=temp_colours[temp], label=r'\${:0.1f}\degree C\$ windward classroom'.format(temp))
    
    df = leeward_df[leeward_df['temp'] == temp]
    ax.plot(df['wind_speed'], df['Q_fresh']/students_per_class,
            color=temp_colours[temp], ls ='--',
            label=r'\${:0.1f}\degree C\$ leeward classroom'.format(temp))
# plt.axhline(10, color='black', ls='--', lw=2)
ax.set_ylim(bottom=0)
ax.legend()
ax.set_xlabel('wind speed [kph]')
ax.set_ylabel('ventilation rate [l/s/pp]')
plt.grid(True)
plt.xlim([0,20])
if save:
    save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
    savepdf_tex(fig=f, fig_loc=save_loc,
                name=f'ventilation_wd_{str(wind_dir).replace(".","_")}')
else:
    plt.show()
plt.close()



