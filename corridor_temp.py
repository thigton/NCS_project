from savepdf_tex import savepdf_tex
from classes.weather import Weather
from classes.stocastic_model import StocasticModel
from classes.contam_model import ContamModel
import pandas as pd
import seaborn as sns
import itertools
from datetime import time, timedelta

import matplotlib.pyplot as plt
plt.style.use('ggplot')

sns.set_theme(style="whitegrid")


plot_time_series = False
save = False
time_step = timedelta(minutes=15)
wind_dir = 0.0
corridor_temps = [5.0, 10.0, 15.0]
wind_speeds = [0.0, 5.0, 10.0, 15.0, 20.0]
time_to_get_results = timedelta(days=5)
contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                        'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                        'name': 'school_corridor'}

# init contam model
contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
                           contam_dir=contam_model_details['prj_dir'],
                           project_name=contam_model_details['name'])

if plot_time_series:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

col_names = pd.MultiIndex.from_product([corridor_temps,
                                        ['total', 'first room']],
                                       names=['corridor temps', 'grouping'])

fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))


for i, wind_speed in enumerate(wind_speeds):
    box_plot_df = pd.DataFrame(columns=col_names)
    corridor_temps_df = pd.DataFrame(columns=corridor_temps)
    for corridor_temp in corridor_temps:
        simulation_constants = {'duration': timedelta(days=7),
                                'mask_efficiency': 0,
                                # [hr^-1] should be quanta hr^-1? h
                                'lambda_home': 0/898.2e4 / 24,
                                # 'lambda_home': 1, # [hr^-1] should be quanta hr^-1
                                'pulmonary_vent_rate': 0.54,  # [m3.hr^-1]
                                'quanta_gen_rate': 5,  # [quanta.hr^-1]
                                'recover_rate': (8.2*24)**(-1),  # [hr^-1]
                                'school_start': time(hour=9),
                                'school_end': time(hour=16),
                                'no_of_simulations': 500,
                                'init_students_per_class': 30,
                                'plotting_sample_rate': f'{int(time_step.seconds/60)}min',
                                'time_step': time_step,
                                'door_open_fraction': 1.0,
                                'window_open_fraction': 1.0,
                                }

        weather_params = Weather(
            wind_speed=wind_speed, wind_direction=wind_dir, ambient_temp=corridor_temp)

        contam_model.set_initial_settings(weather_params, window_height=1.0)
        corridor = contam_model.get_all_zones_of_type(
            search_type_term='corridor')
        contam_model.get_zone_temp_of_room_type(search_type_term='classroom')
        for zone in corridor['Z#'].values:
            contam_model.set_zone_temperature(zone, corridor_temp)

        # run contam simulation
        contam_model.run_simulation(verbose=True)

        model = StocasticModel(weather=weather_params,
                               contam_details=contam_model_details,
                               simulation_constants=simulation_constants,
                               contam_model=contam_model,
                               closing_opening_type=None,
                               method='DTMC',
                               )
        model.run(results_to_track=['risk', 'first_infection_group'],
                #   parallel=True,
                  )

        if plot_time_series:
            model.plot_risk(
                ax=ax[0], comparison='corridor temp', value=corridor_temp)
            model.plot_risk(ax=ax[1], comparison='corridor temp',
                            value=corridor_temp, first_infection_group=True)
        box_plot_df[(corridor_temp, 'total')] = model.get_risk_at_time(
            time_to_get_results)
        box_plot_df[(corridor_temp, 'first room')] = model.get_risk_at_time(
            time_to_get_results, first_infection_group=True)

    if plot_time_series:
        fig.autofmt_xdate(rotation=45)
        ax[0].set_ylabel('Infection Risk')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    ax1.boxplot(x=box_plot_df.xs(key='total', level=1, axis=1),
                positions=corridor_temps,
                manage_ticks=False, widths=0.03,
                boxprops=dict(color=f'C{i}',), whiskerprops=dict(color=f'C{i}',),
                capprops=dict(color=f'C{i}'), flierprops=dict(marker='x'),
                )
    ax1.plot(box_plot_df.xs(key='total', level=1, axis=1).columns,
             box_plot_df.xs(key='total', level=1, axis=1).mean(axis=0), color=f'C{i}', label=f'{wind_speed}kmph')

    ax2.boxplot(x=box_plot_df.xs(key='first room', level=1, axis=1),
                positions=corridor_temps,
                manage_ticks=False, widths=0.03,
                boxprops=dict(color=f'C{i}',), whiskerprops=dict(color=f'C{i}',),
                capprops=dict(color=f'C{i}'), flierprops=dict(marker='x'),
                )
    ax2.plot(box_plot_df.xs(key='first room', level=1, axis=1).columns,
             box_plot_df.xs(key='first room', level=1, axis=1).mean(axis=0), color=f'C{i}', label=f'{wind_speed}kmph')

ax1.legend()
ax2.legend()
ax1.set_xlabel('window height')
ax1.set_ylabel(r'Infection risk [x100\%]')
ax2.set_xlabel('window height')
ax2.set_ylabel(r'Infection risk [x100\%]')


if save:
    save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
    savepdf_tex(fig=fig1, fig_loc=save_loc,
                name=f'window_height_full_{time_to_get_results.days}d_{str(wind_dir).replace(".","-")}deg')
    savepdf_tex(fig=fig2, fig_loc=save_loc,
                name=f'window_height_init_room_{time_to_get_results.days}d_{str(wind_dir).replace(".","-")}deg')

else:
    plt.show()
plt.close()
