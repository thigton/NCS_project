import itertools
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import pandas as pd
import seaborn as sns

import util.util_funcs as uf
from savepdf_tex import savepdf_tex

if __name__ == '__main__':
    save = True

    dtype_converters = {
        'duration': lambda x: timedelta(days=datetime.strptime(x, "%d days").day),
        'school_start': lambda x: datetime.strptime(x, '%H:%M:%S').time(),
        'school_end': lambda x: datetime.strptime(x, '%H:%M:%S').time(),
        'time_step': lambda x: timedelta(minutes=datetime.strptime(x, "0 days %H:%M:%S").minute),
    }
    model_log = pd.read_csv(f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv',
                            index_col=0, converters=dtype_converters)

    runs_to_plot = ['door_011_0deg_q5', 'door_014_90deg_q5']

    df = pd.DataFrame()
    windward_rooms = ['1', '2', '3', '4', '5']
    leeward_rooms = ['7', '8', '9', '10', '11']
    models_to_plot = model_log[(model_log['run name'].isin(runs_to_plot))]
    print(f'Number of models to plot: {len(models_to_plot)}')
    df_lst = []
    for i, file in enumerate(models_to_plot.index):
        print(
            f'extracting data from model {i+1} of {len(models_to_plot)}', end='\r')
        model = uf.load_model(
            file, loc=f'{os.path.dirname(os.path.realpath(__file__))}/results')
        if model.weather.wind_speed != 5.0 and model.weather.ambient_temp != 5.0:
            continue
        model.get_ventilation_rates_from_door_open_df_retropectively()
        vent_df = model.fresh_vent_t_series.loc[(model.fresh_vent_t_series.index.weekday.isin([0, 1, 2, 3, 4])) &
                                                (model.fresh_vent_t_series.index.time >= model.consts['school_start']) &
                                                (model.fresh_vent_t_series.index.time < model.consts['school_end'])]
        cols = pd.MultiIndex.from_product([['wind', 'lee'], ['Q_mean', 'Q_std']], names=['wind/lee', 'Q'])
        df_tmp = pd.DataFrame(columns=cols, index=[file])
        df_tmp.loc[:, 'wind'] = [vent_df.loc[:, pd.IndexSlice[:, windward_rooms]].values.mean(),
                                        vent_df.loc[:, pd.IndexSlice[:, windward_rooms]].values.std()]
        df_tmp.loc[:, 'lee'] = [vent_df.loc[:, pd.IndexSlice[:, leeward_rooms]].values.mean(),
                                        vent_df.loc[:, pd.IndexSlice[:, leeward_rooms]].values.std()]
        df_tmp = df_tmp / 60**2  * 1e3 / 30

        for name in models_to_plot.columns:
            if name in ['wind_speed', 'ambient_temp', 'door_open_fraction', 'wind_direction']:
                df_tmp.loc[file, name] = models_to_plot.loc[file, name]

        df_lst.append(df_tmp)

        del model

    df = pd.concat([df] + df_lst, axis=0)
    fig, ax = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    for temp, mark in zip(df['ambient_temp'].unique(), ['x', '.']):

        plot = df.loc[(df['ambient_temp'] == temp) & (df['wind_speed'] == 5.0) & (df['wind_direction'] == 0.0)]
        ax[0,0].errorbar(x=plot['door_open_fraction'],
                     y=plot[('wind', 'Q_mean')],
                     yerr=plot[('wind', 'Q_std')],
                     elinewidth=1, capsize=5,
                     marker=mark, color='C0',
                     label=r'windward classrooms: {:0.1f}\degree C'.format(temp))
        ax[0,0].errorbar(x=plot['door_open_fraction'],
                 y=plot[('lee', 'Q_mean')],
                 yerr=plot[('lee', 'Q_std')],
                 elinewidth=1, capsize=5,
                 marker=mark, color='C1',
                 label=r'leeward classrooms: {:0.1f}\degree C'.format(temp))
        
        plot = df.loc[(df['ambient_temp'] == temp) & (df['wind_speed'] == 15.0) & (df['wind_direction'] == 0.0)]
        ax[0,1].errorbar(x=plot['door_open_fraction'],
                     y=plot[('wind', 'Q_mean')],
                     yerr=plot[('wind', 'Q_std')],
                     elinewidth=1, capsize=5,
                     marker=mark, color='C0',
                     label=r'windward classrooms: {:0.1f}\degree C'.format(temp))
        ax[0,1].errorbar(x=plot['door_open_fraction'],
                 y=plot[('lee', 'Q_mean')],
                 yerr=plot[('lee', 'Q_std')],
                 elinewidth=1, capsize=5,
                 marker=mark, color='C1',
                 label=r'leeward classrooms: {:0.1f}\degree C'.format(temp))
    
        
        plot = df.loc[(df['ambient_temp'] == temp) & (df['wind_speed'] == 5.0) & (df['wind_direction'] == 90.0)]
        ax[1,0].errorbar(x=plot['door_open_fraction'],
                     y=plot[('wind', 'Q_mean')],
                     yerr=plot[('wind', 'Q_std')],
                     elinewidth=1, capsize=5,
                     marker=mark, color='C0',
                     label=r'windward classrooms: {:0.1f}\degree C'.format(temp))
        ax[1,0].errorbar(x=plot['door_open_fraction'],
                 y=plot[('lee', 'Q_mean')],
                 yerr=plot[('lee', 'Q_std')],
                 elinewidth=1, capsize=5,
                 marker=mark, color='C1',
                 label=r'leeward classrooms: {:0.1f}\degree C'.format(temp))
        
        plot = df.loc[(df['ambient_temp'] == temp) & (df['wind_speed'] == 15.0) & (df['wind_direction'] == 90.0)]
        ax[1,1].errorbar(x=plot['door_open_fraction'],
                     y=plot[('wind', 'Q_mean')],
                     yerr=plot[('wind', 'Q_std')],
                     elinewidth=1, capsize=5,
                     marker=mark, color='C0',
                     label=r'windward classrooms: {:0.1f}\degree C'.format(temp))
        ax[1,1].errorbar(x=plot['door_open_fraction'],
                 y=plot[('lee', 'Q_mean')],
                 yerr=plot[('lee', 'Q_std')],
                 elinewidth=1, capsize=5,
                 marker=mark, color='C1',
                 label=r'leeward classrooms: {:0.1f}\degree C'.format(temp))
    ax[0,0].set_title(r'5.0kph, 0.0\degree')
    ax[0,1].set_title(r'15.0kph, 0.0\degree')
    ax[1,0].set_title(r'5.0kph, 90.0\degree')
    ax[1,1].set_title(r'15.0kph, 90.0\degree')
    for coords in [(0,0), (0,1), (1,0), (1,1)]:    
        # ax[coords].legend()
        ax[coords].set_xlabel('Door open percentage')
        ax[coords].set_ylabel('Ventilation rate [l/s/pp]')
        ax[coords].grid(True)
        ax[coords].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol=r'\%', is_latex=True))
        
    ax[0,1].legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    if save:
        save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
        savepdf_tex(fig=fig, fig_loc=save_loc,
                    name='flow_rates')
    else:
        plt.show()
        plt.close()
