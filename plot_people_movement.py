"""To plot door open percentage comparison

    """
# pylint: disable=no-member
from datetime import time, timedelta, datetime
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import pandas as pd
from classes.weather import Weather
from savepdf_tex import savepdf_tex
import pickle
import os
import numpy as np
np.set_printoptions(precision=3)
import util.util_funcs as uf



if __name__ == '__main__':

    opening_method=['internal_doors_only_random']
    method=['DTMC']
    movement_method = ['change_rooms_in_group', None]
    contam_model_names = ['school_corridor']
    plot_time_series = False
    save=True
    wind_dir = [0.0, 90.0]
    wind_speeds = [5.0,15.0]
    door_opening_fraction = np.round(np.linspace(0.0, 1.0, 21), decimals=2)
    amb_temps = [5.0]
    corridor_temp = [19.0]
    classroom_temp = [21.0]
    window_height = [1.0]
    time_to_get_results = timedelta(days=4, hours=23, minutes=50)
    time_steps = [timedelta(minutes=10)]


    dtype_converters = {
        'duration': lambda x: timedelta(days=datetime.strptime(x,"%d days").day),
        'school_start': lambda x: datetime.strptime(x, '%H:%M:%S').time(),
        'school_end': lambda x: datetime.strptime(x, '%H:%M:%S').time(),
        'time_step': lambda x: timedelta(minutes=datetime.strptime(x,"0 days %H:%M:%S").minute),
    }

    model_log = pd.read_csv(f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv',
                            index_col=0, converters=dtype_converters)
    runs_to_plot = ['movement_001_q5']#, 'movement_002_q10', 'movement_003_q25']
    models_to_plot = model_log[(model_log['run name'].isin(runs_to_plot))]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    col_names = pd.MultiIndex.from_product([models_to_plot['quanta_gen_rate'].unique(),
                                            models_to_plot['wind_speed'].unique(),
                                            models_to_plot['ambient_temp'].unique(),
                                            models_to_plot['movement_method'].unique()],
                                    names=['quanta', 'wind speed', 'ambient temp.', 'movement_method'])
    df_plotting = pd.DataFrame(columns=col_names)
    print(f'Number of models to plot: {len(models_to_plot)}')
    for i, file in enumerate(models_to_plot.index):
        print(f'extracting data from model {i} of {len(models_to_plot)}', end='\r')
        model = uf.load_model(
                file, loc=f'{os.path.dirname(os.path.realpath(__file__))}/results')
    
        speed = model.weather.wind_speed
        temp = model.weather.ambient_temp
        quanta = model.consts['quanta_gen_rate']
        movement_meth = model.movement_method
        df_plotting[(quanta, speed, temp, movement_meth)] = model.get_risk_at_time(time_to_get_results)

    df_plotting = df_plotting.melt(value_name='risk')
    df_plotting['movement_method'].fillna(value='No movement', inplace=True)
    flds = ['ambient temp.', 'wind speed']
    df_plotting[', '.join(flds)] = pd.Series(df_plotting.reindex(flds, axis='columns')
                                              .astype('str')
                                              .values.tolist()
                                                        ).str.join(', ')
    sns.boxplot(x=', '.join(flds), y='risk', hue='movement_method', data=df_plotting)#, fliersize=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol=r'\%', is_latex=True))
    ax.set_xlabel('(ambient temp., wind speed)')
    ax.set_ylabel('Infection Risk')
    if save:
        save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
        savepdf_tex(fig=fig, fig_loc=save_loc,
                    name='people_movement')
    else:
        plt.show()
        plt.close()
    # mean_df = df_plotting.mean().reset_index(level=[0,2,3,4,5])
    # for i, (wind_d, temp, speed, quanta) in enumerate(itertools.product(mean_df['wind_direction'].unique(),
    #                                                     mean_df['ambient temp.'].unique(),
    #                                                     mean_df['wind speed'].unique(),
    #                                                     mean_df['quanta'].unique())):
    #     wind_label = 'low' if speed  == 5.0 else 'high'
    #     temp_label = 'Winter' if temp  == 5.0 else f'Autumn'
    #     quanta_color= {5:'blue', 10:'red',25:'green'}
    #     wind_ls = {5.0: '-', 15.0: '--'}
    #     wind_dir_marker = {0.0: '*', 90.0: 'D'}
    #     df = mean_df[(mean_df['wind speed'] == speed) & (mean_df['ambient temp.'] == temp) & (mean_df['quanta'] == quanta) & (mean_df['wind_direction'] == wind_d)].dropna()
    #     baseline = df.loc[1.0,:]
    #     ax1.plot(df[df['group'] == 'total'].index,
    #              df[df['group'] == 'total'][0]/baseline[baseline['group']== 'total'][0].values[0] - 1,
    #              color=quanta_color[quanta], ls=wind_ls[speed], marker=wind_dir_marker[wind_d],
    #              label=f'{temp_label} - {wind_label} wind speed, quanta = {quanta}, wind_direction = {wind_d}')
    #     ax2.plot(df[df['group'] == 'total'].index,
    #              df[df['group'] == 'first room'][0],#/baseline[baseline['group']== 'first room'][0].values[0],
    #              color=f'C{i}', label=f'{temp_label} - {wind_label} wind speed, quanta = {quanta}, wind_direction = {wind_d}')
    #     # ax1.axhline(1, color='k', ls='--')
    #     # # ax1.plot(box_plot_df.xs(key='total', level=1, axis=1).columns,
    #     #             #  box_plot_df.xs(key='total', level=1, axis=1).quantile(q=0.95, axis=0), color=f'C{i}', ls='--')
    #     # ax2.plot(box_plot_df.xs(key='first room', level=1, axis=1).columns,
    #                 #  box_plot_df.xs(key='first room', level=1, axis=1).quantile(q=0.95, axis=0), color=f'C{i}', ls='--')
    #     # ax1.boxplot(x=box_plot_df.xs(key='total', level=1, axis=1),
    #     #             positions=door_opening_fraction,
    #     #             manage_ticks=False, widths=0.03, whis=[0,100],
    #     #             boxprops=dict(color=f'C{i}',), whiskerprops=dict(color=f'C{i}',),
    #     #             capprops=dict(color=f'C{i}'), flierprops=dict(marker='x'),
    #     # )
    #     # ax2.boxplot(x=box_plot_df.xs(key='first room', level=1, axis=1),
    #     #             positions=door_opening_fraction,
    #     #             manage_ticks=False, widths=0.03, whis=[0,100],
    #     #             boxprops=dict(color=f'C{i}',), whiskerprops=dict(color=f'C{i}',),
    #     #             capprops=dict(color=f'C{i}'), flierprops=dict(marker='x'),
    #     #             )
    #         # box_plot_df.boxplot()
    #         # breakpoint()
    #         # sns.boxplot(x="door open", y="value", hue='grouping', data=box_plot_df.melt(), ax=ax1)

    # for ax in [ax1, ax2]:
    #     ax.legend(frameon=False)
    #     ax.set_xlabel(r'Door open percentage')
    #     ax.set_xlim([0,1])
    #     # ax.set_ylim(bottom=0)
    #     # ax.set_ylabel(r'Infection risk', labelpad=15)
    #     ax.set_ylabel(r'\$RR = IR/IR_{\Gamma = 1} - 1\$', labelpad=15)
    #     ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol=r'\%', is_latex=True))
    #     ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol=r'\%', is_latex=True))
    #     ax.spines['bottom'].set_position(('data', 0))
        

        
    # for ax, axis in itertools.product([ax1,ax2],['bottom','left']):
    #     ax.spines[axis].set_linewidth(2)
    #     # ax.spines[axis].set_color('black')
    # for ax, axis in itertools.product([ax1,ax2],['top','right']):
    #     ax.spines[axis].set_visible(False)
    



    # # ax3.set_xlabel('door opening fraction')
    # # ax3.set_xlim([0,1])

    # if save:
    #     save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
    #     savepdf_tex(fig=fig1, fig_loc=save_loc,
    #                 name=f'winter_q_compare_door_analysis_full_{time_to_get_results.days}d_{str(wind_dir).replace(".","-")}deg')
        
    # else:
    #     plt.show()
    # plt.close()


