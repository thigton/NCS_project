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
wind_speeds = [0.0]#, 5.0, 10.0, 15.0, 20.0]
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

col_names = pd.MultiIndex.from_product([door_opening_fraction,
                                        ['total','first room']],
                                        names=['door open', 'grouping'])
# col_names = pd.Index(data=door_opening_fraction, name='door_open')

fig1, ax1 = plt.subplots(1,1, figsize=(12,6))
fig2, ax2 = plt.subplots(1,1, figsize=(12,6))
fig3, ax3 = plt.subplots(1,1, figsize=(12,6))
fig1.autofmt_xdate(rotation=45)
fig2.autofmt_xdate(rotation=45)

for i, wind_speed in enumerate(wind_speeds):
    box_plot_df = pd.DataFrame(columns=col_names)
    door_open_df = pd.DataFrame(columns=door_opening_fraction)
    for door in door_opening_fraction:
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

        if plot_time_series:
            model.plot_risk(ax=ax[0], comparison='door_open_fraction')
            model.plot_risk(ax=ax[1], comparison='door_open_fraction', first_infection_group=True)
        box_plot_df[(door, 'total')] = model.get_risk_at_time(time_to_get_results)
        box_plot_df[(door, 'first room')] = model.get_risk_at_time(time_to_get_results, first_infection_group=True)
        door_open_df[door]= model.door_open_fraction_actual


    if plot_time_series:
        fig.autofmt_xdate(rotation=45)
        ax[0].set_ylabel('Infection Risk')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()



    ax1.boxplot(x=box_plot_df.xs(key='total', level=1, axis=1),
                positions=door_opening_fraction,
                manage_ticks=False, widths=0.03,
                boxprops=dict(color=f'C{i}',), whiskerprops=dict(color=f'C{i}',),
                capprops=dict(color=f'C{i}'), flierprops=dict(marker='x'),
    )
    ax1.plot(box_plot_df.xs(key='total', level=1, axis=1).columns,
             box_plot_df.xs(key='total', level=1, axis=1).mean(axis=0), color=f'C{i}', label=f'{wind_speed}kmph')


    ax2.boxplot(x=box_plot_df.xs(key='first room', level=1, axis=1),
                positions=door_opening_fraction,
                manage_ticks=False, widths=0.03,
                boxprops=dict(color=f'C{i}',), whiskerprops=dict(color=f'C{i}',),
                capprops=dict(color=f'C{i}'), flierprops=dict(marker='x'),
                )
    ax2.plot(box_plot_df.xs(key='first room', level=1, axis=1).columns,
             box_plot_df.xs(key='first room', level=1, axis=1).mean(axis=0), color=f'C{i}', label=f'{wind_speed}kmph')

    door_open_df.loc[:,~door_open_df.columns.isin([0.0,1.0])].plot.kde(ax=ax3)
    # box_plot_df.boxplot()
    # breakpoint()
    # sns.boxplot(x="door open", y="value", hue='grouping', data=box_plot_df.melt(), ax=ax1)
ax1.legend()
ax2.legend()
ax3.legend()
ax1.set_xlabel('door open ratio')
ax1.set_ylabel(r'Infection risk [x100\%]')
ax2.set_xlabel('door open ratio')
ax2.set_ylabel(r'Infection risk [x100\%]')

ax3.set_xlabel('door opening fraction')
ax3.set_xlim([0,1])

if save:
    save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
    savepdf_tex(fig=fig1, fig_loc=save_loc,
                name=f'door_analysis_full_{time_to_get_results.days}d_{str(wind_dir).replace(".","-")}deg')
    savepdf_tex(fig=fig2, fig_loc=save_loc,
                name=f'door_analysis_init_room_{time_to_get_results.days}d_{str(wind_dir).replace(".","-")}deg')
    # savepdf_tex(fig=fig3, fig_loc=save_loc,
                # name=f'door_opening_dist_{time_to_get_results.days}d_{str(wind_dir).replace(".","-")}deg')

else:
    plt.show()
plt.close()


