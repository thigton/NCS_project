import itertools
from datetime import time, timedelta

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from classes.contam_model import ContamModel
from classes.stocastic_model import StocasticModel
from classes.weather import Weather
from savepdf_tex import savepdf_tex

plot_time_series = False
save=True
wind_dir = 0.0
wind_speeds = np.linspace(0.0, 20.0, 21)
amb_temp = np.linspace(5.0, 19.0, 15)
windward_name = 'Classroom6'
leeward_name = 'Classroom1'
students_per_class = 30


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
f, ax = plt.subplots()
points = ax.scatter(windward_df['wind_speed'], windward_df['Q_total']/students_per_class, c=windward_df['temp'],
                    s=25, cmap='coolwarm' , label= 'windward classroom')
points = ax.scatter(leeward_df['wind_speed'], leeward_df['Q_total']/students_per_class, c=leeward_df['temp'],
                    marker='D',s=25, cmap='coolwarm', label='leeward classroom')
plt.axhline(10, color='red', ls='--', lw=2)
f.colorbar(points)
plt.legend()
plt.xlabel('wind speed [kph]')
plt.ylabel('ventilation rate [l/s/pp]')
if save:
    save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
    savepdf_tex(fig=f, fig_loc=save_loc,
                name=f'total_ventilation')
else:
    plt.show()
plt.close()

f, ax = plt.subplots()
points = ax.scatter(windward_df['wind_speed'], windward_df['Q_fresh']/students_per_class, c=windward_df['temp'],
                    s=25, cmap='coolwarm', label='windward classroom')
points = ax.scatter(leeward_df['wind_speed'], leeward_df['Q_fresh']/students_per_class, c=leeward_df['temp'],
                    marker='D',s=25, cmap='coolwarm', label='leeward classroom')
plt.axhline(10, color='red', ls='--', lw=2)
f.colorbar(points)
plt.legend()
plt.xlabel('wind speed [kph]')
plt.ylabel('ventilation rate [l/s/pp]')
if save:
    savepdf_tex(fig=f, fig_loc=save_loc,
                name=f'fresh_ventilation')
else:
    plt.show()
plt.close()



