"""To plot door open percentage comparison

    """
# pylint: disable=no-member
from datetime import time, timedelta, datetime
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import mplcursors
import seaborn as sns
import itertools
import pandas as pd
from savepdf_tex import savepdf_tex
import os
import pickle
import numpy as np
import util.util_funcs as uf
import concurrent.futures
from classes.contam_model import celciusToKelvin, kilometresPerHourToMetresPerSecond, PerHourToPerSecond


np.set_printoptions(precision=3)


def df_filter(df, filter_dic):
    filter_lst = []
    for k, v in filter_dic.items():
        if k in ['door_open_fraction']:
            filter_lst.append(df[k].round(decimals=2).isin(v))
        elif isinstance(v, list):
            filter_lst.append(df[k].isin(v))
        elif k in ['recover_rate']:
            filter_lst.append(df[k].round(decimals=3) == round(v, ndigits=3))
        else:
            filter_lst.append(df[k] == v)

    filter_lst = pd.concat(filter_lst, axis=1)
    return filter_lst.all(axis=1)

def get_flow_rates_df(args):
    model_no, file = args

    print(f'extracting data from model {model_no} of {len(models_to_plot)}', end='\r')
    model = uf.load_model(
        file, loc=f'{os.path.dirname(os.path.realpath(__file__))}/results')
    model.get_ventilation_rates_from_door_open_df_retropectively(school_time_only=True)
    df_tmp = model.ave_risk_and_timescales_df(TIME_TO_GET_RESULT,
                                              incl_init_group=True,
                                              windward_grouping=False)
    df_tmp = df_tmp.merge(pd.Series(model.first_infection_group, name='first_infection_group'),
                          how='left',
                          left_on=df_tmp.index.names[0],
                          right_index=True)
    for name in models_to_plot.columns:
        df_tmp[name] = models_to_plot.loc[models_to_plot['model'] == file, name].values[0]
    del model
    return df_tmp


if __name__ == '__main__':

    SAVE = False
    PARALLEL = True
    CROSS_TRANSMISSION_ONLY = True
    REMOVE_SCALING = False

    TIME_TO_GET_RESULT = timedelta(days=4, hours=23, minutes=50)

    dtype_converters = {
        'duration': lambda x: timedelta(days=datetime.strptime(x, "%d days").day),
        'school_start': lambda x: datetime.strptime(x, '%H:%M:%S').time(),
        'school_end': lambda x: datetime.strptime(x, '%H:%M:%S').time(),
        'time_step': lambda x: timedelta(minutes=datetime.strptime(x, "0 days %H:%M:%S").minute),
    }
    MODEL_LOG = pd.read_csv(f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv',
                            index_col=0,
                            converters=dtype_converters)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax2 = fig.add_axes([0.625, 0.575, 0.35, 0.35])
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 6))
    ax5 = fig4.add_axes([0.625, 0.575, 0.35, 0.35])

    RUNS_TO_PLOT = [
        'door_027_0deg_q5_wmul_1', 'door_028_0deg_q5_wmul_05',
        'door_029_0deg_q5_wmul_1', 'door_030_0deg_q5_wmul_05',
        'door_012_0deg_q10', 'door_011_0deg_q5', 'door_013_0deg_q25',
        'door_014_90deg_q5', 'door_015_90deg_q10', 'door_016_90deg_q25',
        'door_023_4_rooms_0deg_q5', 'door_024_4_rooms_90deg_q5',
        'door_025_6_rooms_0deg_q5', 'door_026_6_rooms_90deg_q5',
        ]
    groups = []
    for run_name in RUNS_TO_PLOT:
        print(f'run name: {run_name}')
        run = False
        try:
            with open(f'{os.path.dirname(os.path.realpath(__file__))}/results/{run_name}_flow_rate_risk_df.pickle', 'rb') as pickle_in:
                df = pickle.load(pickle_in)
            print('Loaded successfully')
            if isinstance(df, list):
                df = df[-1]
        except FileNotFoundError:
            print(f'No df found for {run_name}')
            run = True
            df = pd.DataFrame(columns=MODEL_LOG.columns
                              )
        if run:
            models_to_plot = MODEL_LOG[(MODEL_LOG['run name'] == run_name)]
            print(f'Number of models to plot: {len(models_to_plot)}')
            df_lst = []
            if PARALLEL:
                with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
                    df_lst = executor.map(get_flow_rates_df,
                                       [(x, file) for x, file in enumerate(models_to_plot['model'])])
            else:
                def run_generator(model_list):
                    for count, file in enumerate(model_list):
                        yield get_flow_rates_df((count, file))
                df_lst = run_generator(models_to_plot['model'])

            df = pd.concat(list(df_lst), axis=0)

            with open(f'{os.path.dirname(os.path.realpath(__file__))}/results/{run_name}_flow_rate_risk_df.pickle', 'wb') as pickle_out:
                pickle.dump(df, pickle_out)

        
        df.index = pd.MultiIndex.from_tuples(df.index, names=('sim_id', 'room_id'))
        g = 9.81 # ms^-2
        window_extent = 1 # window height is currently 1 m will need to change if the window height changes
        T_class = celciusToKelvin(df['classroom_temp'])
        T_amb = celciusToKelvin(df['ambient_temp'])
        u = kilometresPerHourToMetresPerSecond(df['wind_speed'])
        df['Froude'] = ((u / (g*(T_class-T_amb)/T_amb * window_extent)**0.5)
                        .astype(float)
                        .round(decimals=1)
                        )


        if REMOVE_SCALING:
            df['Q_mean'] = df['Q_mean'] * df['quanta_gen_rate']
            df['Q_mean_ex'] = df['Q_mean_ex'] * df['quanta_gen_rate']
        if CROSS_TRANSMISSION_ONLY:
            df = df.loc[df.index.get_level_values(1) != df['first_infection_group']]
        def grouping_func(group):
            if group.name == 'contam_model_name':
                return group.value_counts().index[0]
            # elif group.name in ['risk']:
                # return group.quantile(0.9)
            elif group.name not in ['Q_std', 'Q_std_ex']:
                return group.mean()
            else:
                var = group.apply(lambda x: x**2).mean()
                return np.sqrt(var)
        groups.append(df.groupby(['wind_direction', 'wind_speed',
        'ambient_temp', 'quanta_gen_rate', 'door_open_fraction', 'contam_model_name']).agg(grouping_func))
    grouped_df = pd.concat(groups, axis=0)


    grouped_df.reset_index(inplace=True)
    grouped_df.rename({'quanta_gen_rate': 'quanta generation rate',
                       'wind_direction': 'wind direction',
                       'contam_model_name' : 'ventilation model'},
                      axis=1, inplace=True)
    grouped_df['ventilation model'].replace({'school_corridor': '10 classrooms',
                                             '6_classrooms': '6 classrooms',
                                             '4_classrooms': '4 classrooms'},
                                            inplace=True)
    if CROSS_TRANSMISSION_ONLY:
        grouped_df = grouped_df.loc[grouped_df['risk'] != 0.0].copy()
    for axs, x_axis in zip ([[ax, ax2], [ax4, ax5]],
                                     ['Q_mean', 'Q_mean_ex']):
        s1 = sns.scatterplot(x=x_axis, y='risk', data=grouped_df, ax=axs[0],
                             hue='Froude', style='ventilation model', palette='viridis')
        s2 = sns.scatterplot(x=x_axis, y='risk', data=grouped_df, ax=axs[1],
                             hue='Froude', style='ventilation model', palette='viridis', legend=False)
        axs[0].set_ylim([0, 1])
        axs[1].set_ylim([0, 0.06])

        axs[0].set_ylabel('Infection risk')
        axs[1].set_ylabel('Infection risk')
        axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol=r'\%', is_latex=True))
        axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol=r'\%', is_latex=True))
        axs[0].legend(bbox_to_anchor=(0.90, 0.05), loc='lower right', borderaxespad=0.)
    ax.set_xlabel(r'\$\frac{Q_{fresh}/V}{q}\$')
    ax2.set_xlabel(r'\$\frac{Q_{fresh}/V}{q}\$')
    ax4.set_xlabel(r'\$\frac{Q_{int}/V}{q}\$')
    ax5.set_xlabel(r'\$\frac{Q_{int}/V}{q}\$')
    
    # norm = Normalize(vmin=0, vmax=0.1)
    norm = LogNorm(vmin=1e-4, vmax=1e-1)
    sns.scatterplot(x='Q_mean', y='Q_mean_ex', data=grouped_df, ax=ax3,
                         hue='risk', palette='viridis', hue_norm=norm,
                         style='ventilation model',)
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_ylim(bottom=0)
    ax3.set_xlim(left=0)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    # sm.set_array([])
    ax3.get_legend().remove()
    # breakpoint()
    fig3.colorbar(sm, format=mtick.PercentFormatter(xmax=1, decimals=None, symbol=r'\%', is_latex=True))
    ax3.set_ylabel(r'\$\frac{Q_{int}/V}{q}\$')
    ax3.set_xlabel(r'\$\frac{Q_{fresh}/V}{q}\$')
    
    fig.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    if SAVE:
        save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
        savepdf_tex(fig=fig, fig_loc=save_loc,
                    name='scaling_attempt_all_rooms')
        savepdf_tex(fig=fig, fig_loc=save_loc,
                    name='Q_int vs infection risk')
        savepdf_tex(fig=fig3, fig_loc=save_loc,
                    name='Q_fresh_vs_Q_int')
    else:
        cursor = mplcursors.cursor(ax3, hover=True)

        cursor.connect("add", lambda sel: sel.annotation.set_text(f'Risk: {grouped_df["risk"][sel.target.index]:0.3f}\nWind speed: {grouped_df["wind_speed"][sel.target.index]:0.3f}\nWind direction: {grouped_df["wind direction"][sel.target.index]:0.3f}\nAmbient Temp: {grouped_df["ambient_temp"][sel.target.index]:0.3f}'))
        plt.show()
        plt.close()
    