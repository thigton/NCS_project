"""To plot door open percentage comparison

    """
# pylint: disable=no-member
from datetime import time, timedelta, datetime
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from classes.weather import Weather
from savepdf_tex import savepdf_tex
import pickle
import os
import numpy as np
np.set_printoptions(precision=3)

def df_filter(df, filter_dic):
    filter_lst = []
    for k, v in filter_dic.items():
        if k in ['door_open_fraction']:
            filter_lst.append(df[k].round(decimals=2).isin(v))
        elif isinstance(v, list):
            filter_lst.append(df[k].isin(v))
        elif k in ['recover_rate']:
            filter_lst.append(df[k].round(decimals=3) == round(v,ndigits=3))
        else:
            filter_lst.append(df[k] == v)

    filter_lst = pd.concat(filter_lst, axis=1)
    return filter_lst.all(axis=1)

def load_data(fname):
    try:
        with open(f'{os.path.dirname(os.path.realpath(__file__))}/results/{fname}.pickle', 'rb') as pickle_in:
            model = pickle.load(pickle_in)
    except FileNotFoundError:
        with open(f'/mnt/usb-WD_Elements_2620_575832314441394E37445032-0:0-part1/NCS Project/stochastic_model/results/{fname}.pickle', 'rb') as pickle_in:
            model = pickle.load(pickle_in)
    return model

if __name__ == '__main__':

    opening_method=['internal_doors_only_random']
    method=['DTMC']
    contam_model_names = ['school_corridor']
    plot_time_series = False
    save=False
    
    wind_dir = [0.0, 90.0]
    wind_speeds = [5.0,15.0]
    door_opening_fraction = np.round(np.linspace(0.0, 1.0, 21), decimals=2)
    amb_temps = [5.0]
    corridor_temp = [19.0]
    classroom_temp = [21.0]
    window_height = [1.0]
    time_to_get_results = timedelta(days=4, hours=23, minutes=50)
    time_steps = [timedelta(minutes=10)]
    filters = {'duration': timedelta(days=5),
                'mask_efficiency': 0,
                'lambda_home': 0/898.2e4 / 24, # [hr^-1] should be quanta hr^-1? h
                'pulmonary_vent_rate': 0.54, # [m3.hr^-1]
                'quanta_gen_rate': [5],#,10,25], # [quanta.hr^-1]
                'recover_rate': (6*24)**(-1), # [hr^-1]
                'school_start': time(hour=9),
                'school_end': time(hour=16),
                'no_of_simulations': 2000,
                'init_students_per_class': 30,
                'plotting_sample_rate': [f'{int(x.seconds/60)}min' for x in time_steps],
                'time_step': time_steps,
                'door_open_fraction': door_opening_fraction,
                'window_open_fraction': 1.0,
                'ambient_temp': amb_temps,
                'wind_speed': wind_speeds,
                'wind_direction': wind_dir,
                'window_height': window_height,
                'corridor_temp': corridor_temp,
                'classroom_temp': classroom_temp,
                'method': method,
                'opening_method': opening_method,
                'contam_model_name': contam_model_names,
                                   }


    dtype_converters = {
        'duration': lambda x: timedelta(days=datetime.strptime(x,"%d days").day),
        'school_start': lambda x: datetime.strptime(x, '%H:%M:%S').time(),
        'school_end': lambda x: datetime.strptime(x, '%H:%M:%S').time(),
        'time_step': lambda x: timedelta(minutes=datetime.strptime(x,"0 days %H:%M:%S").minute),
    }

    model_log = pd.read_csv(f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv',
                            index_col=0, converters=dtype_converters)
    runs_to_plot = ['door_023_4_rooms_0deg_q5', 'door_024_4_rooms_90deg_q5']
    models_to_plot = model_log[(model_log['run name'].isin(runs_to_plot) )]
    # models_to_plot = model_log[df_filter(model_log, filters)]
    fig1, ax1 = plt.subplots(1,1, figsize=(12,6))
    fig2, ax2 = plt.subplots(1,1, figsize=(12,6))
    col_names = pd.MultiIndex.from_product([models_to_plot['quanta_gen_rate'].unique(),
                                            models_to_plot['door_open_fraction'].unique(),
                                            models_to_plot['wind_speed'].unique(),
                                            models_to_plot['ambient_temp'].unique(),
                                            models_to_plot['wind_direction'].unique(),
                                            ],
                                    names=['quanta', 'gamma', 'wind_speed', 'amb_temp', 'wind_direction'])
    df_plotting = pd.DataFrame(columns=col_names)
    print(f'Number of models to plot: {len(models_to_plot)}')
    for i, file in enumerate(models_to_plot.index):
        print(f'extracting data from model {i} of {len(models_to_plot)}', end='\r')
        model = load_data(file)
        door = round(model.consts['door_open_fraction'], 2)
        speed = model.weather.wind_speed
        temp = model.weather.ambient_temp
        quanta = model.consts['quanta_gen_rate']
        wind_dirx = model.weather.wind_direction
        df_plotting[(quanta, door, speed, temp,wind_dirx)] = model.get_risk_at_time(time_to_get_results)


    mean_df = df_plotting.mean().reset_index(level=[0,2,3,4])
    for i, (wind_d,temp, speed, quanta) in enumerate(itertools.product(mean_df['wind_direction'].unique(),
                                                        mean_df['amb_temp'].unique(),
                                                        mean_df['wind_speed'].unique(),
                                                        mean_df['quanta'].unique())):
        wind_label = 'low' if speed  == 5.0 else 'high'
        temp_label = 'Winter' if temp  == 5.0 else f'Autumn'
        quanta_color= {5:'blue', 10:'red',25:'green'}
        wind_ls = {5.0: '-', 15.0: '--'}
        wind_dir_marker = {0.0: '*', 90.0: 'D'}
        df = mean_df[(mean_df['wind_speed'] == speed) & (mean_df['amb_temp'] == temp) & (mean_df['quanta'] == quanta) & (mean_df['wind_direction'] == wind_d)].dropna()
        baseline = df.loc[1.0,:]
        ax1.plot(df.index,
                 df[0]/baseline[0] - 1,
                 color=quanta_color[quanta], ls=wind_ls[speed], marker=wind_dir_marker[wind_d],
                 label=f'{temp_label} - {wind_label} wind speed, quanta = {quanta}, wind_direction = {wind_d}')
        ax2.plot(df.index,
                 df[0],#/baseline[baseline['group']== 'first room'][0].values[0],
                 color=f'C{i}', label=f'{temp_label} - {wind_label} wind speed, quanta = {quanta}, wind_direction = {wind_d}')
        # ax1.axhline(1, color='k', ls='--')
        # # ax1.plot(box_plot_df.xs(key='total', level=1, axis=1).columns,
        #             #  box_plot_df.xs(key='total', level=1, axis=1).quantile(q=0.95, axis=0), color=f'C{i}', ls='--')
        # ax2.plot(box_plot_df.xs(key='first room', level=1, axis=1).columns,
                    #  box_plot_df.xs(key='first room', level=1, axis=1).quantile(q=0.95, axis=0), color=f'C{i}', ls='--')
        # ax1.boxplot(x=box_plot_df.xs(key='total', level=1, axis=1),
        #             positions=door_opening_fraction,
        #             manage_ticks=False, widths=0.03, whis=[0,100],
        #             boxprops=dict(color=f'C{i}',), whiskerprops=dict(color=f'C{i}',),
        #             capprops=dict(color=f'C{i}'), flierprops=dict(marker='x'),
        # )
        # ax2.boxplot(x=box_plot_df.xs(key='first room', level=1, axis=1),
        #             positions=door_opening_fraction,
        #             manage_ticks=False, widths=0.03, whis=[0,100],
        #             boxprops=dict(color=f'C{i}',), whiskerprops=dict(color=f'C{i}',),
        #             capprops=dict(color=f'C{i}'), flierprops=dict(marker='x'),
        #             )
            # box_plot_df.boxplot()
            # breakpoint()
            # sns.boxplot(x="door open", y="value", hue='grouping', data=box_plot_df.melt(), ax=ax1)

    for ax in [ax1, ax2]:
        ax.legend(frameon=False)
        ax.set_xlabel(r'Door open percentage')
        ax.set_xlim([0,1])
        # ax.set_ylim(bottom=0)
        # ax.set_ylabel(r'Infection risk', labelpad=15)
        ax.set_ylabel(r'\$RR = IR/IR_{\Gamma = 1} - 1\$', labelpad=15)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol=r'\%', is_latex=True))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol=r'\%', is_latex=True))
        ax.spines['bottom'].set_position(('data', 0))
        

        
    for ax, axis in itertools.product([ax1,ax2],['bottom','left']):
        ax.spines[axis].set_linewidth(2)
        # ax.spines[axis].set_color('black')
    for ax, axis in itertools.product([ax1,ax2],['top','right']):
        ax.spines[axis].set_visible(False)
    



    # ax3.set_xlabel('door opening fraction')
    # ax3.set_xlim([0,1])

    if save:
        save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
        savepdf_tex(fig=fig1, fig_loc=save_loc,
                    name=f'winter_q_compare_door_analysis_full_{time_to_get_results.days}d_{str(wind_dir).replace(".","-")}deg')
        # savepdf_tex(fig=fig2, fig_loc=save_loc,
        #             name=f'DTMCdoor_analysis_init_room_{time_to_get_results.days}d_{str(wind_dir).replace(".","-")}deg')
        # savepdf_tex(fig=fig3, fig_loc=save_loc,
                    # name=f'door_opening_dist_{time_to_get_results.days}d_{str(wind_dir).replace(".","-")}deg')

    else:
        plt.show()
    plt.close()


