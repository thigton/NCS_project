import os
from datetime import datetime, timedelta
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from classes.simulation import Simulation
import random
from savepdf_tex import savepdf_tex

class StocasticModel():
    def __init__(self, weather, contam_details,
                simulation_constants,
                contam_model, closing_opening_type):

        self.weather = weather
        self.contam_details = contam_details
        self.consts = simulation_constants
        self.contam_model = contam_model
        self.closing_opening_type = closing_opening_type
        self.vent_mat_closed = self.get_closed_matrix(opening=self.closing_opening_type)
        self.start_time = datetime(year=2021, month=3, day=1, hour=0, minute=0, second=0)
        self.end_time = self.start_time + self.consts['duration']
        self.S_df = pd.DataFrame()
        self.I_df = pd.DataFrame()
        self.R_df = pd.DataFrame()
        self.risk = pd.DataFrame()
        self.inter_event_time = pd.DataFrame()
        self.first_infection_group = pd.DataFrame()



    def run(self, results_to_track):
        """[summary]

        Args:
            results_to_track (list<str>): options: S_df, I_df, R_df, risk, inter_event_time, first_infection_group
        """
        results = defaultdict(list)
        for sim_number in range(self.consts['no_of_simulations']):
            if sim_number % 10 == 0:
                print(f'{sim_number/self.consts["no_of_simulations"]:0.0%} complete', end='\r')
            sim = Simulation(sim_id=sim_number,
                             start_time=self.start_time,
                             simulation_constants = self.consts,
                             contam_model=self.contam_model,
                             vent_mat_closed={'matrix': self.vent_mat_closed,
                                                'type': self.closing_opening_type},
                            )
            sim.run()
            sim.generate_dataframes()
            for val in results_to_track:
                results[val].append(getattr(sim, val))
            results['door_open_fraction_actual'].append(sim.door_open_percentage)
            results['window_open_fraction_actual'].append(sim.window_open_percentage)
            # self.simulations.append(sim)
            # if sim.S_df.loc[self.end_time].sum() / sim.S_df.loc[self.start_time].sum() < 0.6:
            #     # sample some simulations to make sure they are running ok
            #     sim.plot_lambda()
            #     sim.plot_SIR()
            #     sim.plot_SIR_total()
            #     exit()
        for k,v in results.items():
            if k in ['door_open_fraction_actual', 'window_open_fraction_actual']:
                setattr(self, k, v)
                continue
            axis = 0 if k == 'inter_event_time' else 1
            setattr(self, k, pd.concat(results[k], axis=axis))

        if 'inter_event_time' in results.keys():
            self.plot_inter_event_time_distribution()

        del results

    def plot_SIR(self, ax, ls, comparison, **kwargs):
        value = kwargs['value'] if 'value' in kwargs else self.consts[comparison]
        for i, X in enumerate(['S','I','R']):
            time = getattr(self, f'{X}_sum').index
            mean = getattr(self, f'{X}_sum').mean(axis=1)
            q_05 = getattr(self, f'{X}_sum').quantile(q=0.05, axis=1)
            q_95 = getattr(self, f'{X}_sum').quantile(q=0.95, axis=1)

            ax.plot(time, mean, label=f'{X} - {comparison} -- {value}', color=f'C{i}', ls=ls)
            ax.fill_between(time, q_05, q_95, color=f'C{i}', alpha=0.2)


    def plot_inter_event_time_distribution(self):
        fig, ax  = plt.subplots(1,1, figsize=(10,5))
        (pd.to_timedelta(self.inter_event_time) / pd.Timedelta(hours=1)).hist(bins=range(0,120, 1), ax=ax, density=True)
        ax.set_xlabel('Inter event time [hr]')
        ax.set_ylabel('density')
        ax.set_xlim(left=0)
        save = input('Do you want to save the inter_event_time_distribution? [Y/N]')
        if 'y' in save.lower():
            save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'

            savepdf_tex(fig=fig, fig_loc=save_loc,
                    name=f'inter_event_time_distribution')
        else:
            plt.show()
        plt.close()

    def plot_time_step_distribution(self):
        pass

    def plot_risk(self, ax, comparison, **kwargs):
        df = self.first_infection_group if 'first_infection_group' in kwargs else self.risk
        value = kwargs['value'] if 'value' in kwargs else self.consts[comparison]
        time = df.index
        mean = df.mean(axis=1)
        q_05 = df.quantile(q=0.05, axis=1)
        q_95 = df.quantile(q=0.95, axis=1)

        ax.plot(time, mean, label=f'{comparison} -- {value}',)
        ax.fill_between(time, q_05, q_95, alpha=0.2)

    @property
    def S_sum(self):
        return self.S_df.groupby(level='sim id', axis=1).sum()

    @property
    def I_sum(self):
        return self.I_df.groupby(level='sim id', axis=1).sum()

    @property
    def R_sum(self):
        return self.R_df.groupby(level='sim id', axis=1).sum()

    def get_risk_at_time(self, time, **kwargs):
        df = self.first_infection_group if 'first_infection_group' in kwargs else self.risk
        date_time = self.start_time + time
        return df.loc[date_time].values

    def get_closed_matrix(self, opening):
        dic = {'door': {'fraction' : self.consts['door_open_fraction'],
                        'open_type': 4,
                        'closed_type': 3} ,
               'window':{'fraction' : self.consts['window_open_fraction'],
                        'open_type': 1,
                        'closed_type': 6}
                        }
        self.contam_model.set_all_flow_paths_of_type_to(search_type_term=opening[1:],
                                    param_dict={'type': dic[opening]['closed_type']},
                                    rerun=True,
                                            )
        
        #save matrix
        vent_mat_closed = self.contam_model.vent_mat
        #reopen openings so self.contam_model.vent_mat is the open matrix
        self.contam_model.set_all_flow_paths_of_type_to(search_type_term=opening[1:],
                                                        param_dict={'type': dic[opening]['open_type']},
                                                        rerun=True,
                                                                )
        return vent_mat_closed


if __name__ == '__main__':
    pass
