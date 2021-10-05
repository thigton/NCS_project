from datetime import datetime
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from classes.CTMC_simulation import Simulation
from classes.DTMC_simulation import DTMC_simulation
from savepdf_tex import savepdf_tex
import concurrent.futures
import gc

class StocasticModel():
    def __init__(self, weather, contam_details,
                simulation_constants,
                contam_model, opening_method=None, movement_method=None,
                method='CTMC',
                ):

        self.weather = weather
        self.contam_details = contam_details
        self.consts = simulation_constants
        self.contam_model = contam_model
        self.opening_method = opening_method
        self.movement_method = movement_method
        self.method = method
        if self.method == 'CTMC':
            self.sim_method = Simulation
        elif self.method == 'DTMC':
            self.sim_method = DTMC_simulation
        else:
            raise ValueError('method should be either DTMC or CTMC [default]')
        self.start_time = datetime(year=2021, month=3, day=1,
                                   hour=self.consts['school_start'].hour,
                                   minute=self.consts['school_start'].minute,
                                   second=self.consts['school_start'].second,
                                   microsecond=self.consts['school_start'].microsecond,)
        self.end_time = self.start_time + self.consts['duration']
        self.S_df = pd.DataFrame()
        self.I_df = pd.DataFrame()
        self.R_df = pd.DataFrame()
        self.risk = pd.DataFrame()
        self.inter_event_time = pd.DataFrame()
        self.first_infection_group = pd.DataFrame()


    def run(self, results_to_track, parallel=False):
        """main run function which is actually called from run script."""
        if parallel:
            self.run_in_parallel(results_to_track)
        else:
            self.run_for_loop(results_to_track)
            
    def run_in_parallel(self, results_to_track):
        """Runs the different simulations in parallel (across multiple cpus)"""
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            results = executor.map(self.run_base_routine,
                                   [(x, results_to_track) for x in range(self.consts['no_of_simulations'])])
            # for r in executor.map(
        #         func, [x[i:i + 1] for i in range(n)], chunksize=100):
        results_dic = self.assign_results_as_attrs(results)
        if 'inter_event_time' in results_dic.keys():
            self.plot_inter_event_time_distribution()
        del results
        gc.collect()

    def run_for_loop(self,results_to_track):
        """Run script if onnly using a single cpu."""
        def run_generator(no_of_simulations, results_to_track):
            for sim_number in range(no_of_simulations):
                yield self.run_base_routine((sim_number, results_to_track))
        results = run_generator(self.consts['no_of_simulations'], results_to_track)
        results_dic = self.assign_results_as_attrs(results)
        if 'inter_event_time' in results_dic.keys():
            self.plot_inter_event_time_distribution()
        del results
        gc.collect()

    def run_base_routine(self, args_tuple):
        """actual run routine of each simulation, including some results extraction

        Args:
            results_to_track (list<str>): options: S_df, I_df, R_df, risk, inter_event_time, first_infection_group
        """
        sim_number, results_to_track = args_tuple
        print(f'Starting simulation {sim_number} of {self.consts["no_of_simulations"]}', end='\r')
        
        results = defaultdict()
        sim = self.sim_method(sim_id=sim_number,
                         start_time=self.start_time,
                         simulation_constants = self.consts,
                         contam_model=self.contam_model,
                         opening_method= self.opening_method,
                         movement_method = self.movement_method
                        )
        sim.run()
        sim.generate_dataframes()

        # sim.plot_quanta_conc()
        # sim.plot_lambda()
        # sim.plot_SIR()
        # sim.plot_SIR_total()
        if 'door' in self.opening_method:
            results['door_open_fraction_actual'] = sim.door_open_percentage
        for val in results_to_track:
            results[val] = getattr(sim, val)
        del sim
        return results

    def assign_results_as_attrs(self, results):
        """Takes the results generator created when running the 
        simulations and assigning the important data to 
        atrributes of the stochastic model"""
        results_dic = defaultdict(list)
        for result in results:
            for k, v in result.items():
                results_dic[k].append(v)
        for k,v in results_dic.items():
            if k in ['door_open_fraction_actual', 'window_open_fraction_actual']:
                setattr(self, k, v)
                continue
            axis = 0 if k == 'inter_event_time' else 1
            setattr(self, k, pd.concat(results_dic[k], axis=axis))
            getattr(self, k).fillna(axis=0, inplace=True, method='ffill')
        del results
        return results_dic

    def plot_SIR(self, ax, ls, comparison=None, **kwargs):
        """Plot mean SIR of all simulations"""
        if comparison:
            value = kwargs['value'] if 'value' in kwargs else self.consts[comparison]
        else:
            value = ''

        lw = 1 if 'lw' not in kwargs else kwargs['lw']
        for i, X in enumerate(['S','I','R']):
            time = getattr(self, f'{X}_sum').index
            mean = getattr(self, f'{X}_sum').mean(axis=1)
            q_05 = getattr(self, f'{X}_sum').quantile(q=0.05, axis=1)
            q_95 = getattr(self, f'{X}_sum').quantile(q=0.95, axis=1)

            ax.plot(time, mean, label=f'{X} - {comparison} -- {value}', color=f'C{i}', ls=ls, lw=lw)
            ax.fill_between(time, q_05, q_95, color=f'C{i}', alpha=0.2)


    def plot_inter_event_time_distribution(self):
        """Plot distibution of the time to the next event (CTMC method only) Redundant"""
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


    def plot_risk(self, ax, comparison, **kwargs):
        """Plot the mean infection risk with time."""
        df = self.first_infection_group if 'first_infection_group' in kwargs else self.risk
        value = kwargs['value'] if 'value' in kwargs else self.consts[comparison]
        lw = 1 if 'lw' not in kwargs else kwargs['lw']
        ls = '-' if 'ls' not in kwargs else kwargs['ls']
        color = next(ax._get_lines.prop_cycler)['color'] if 'color' not in kwargs else kwargs['color']
        time = df.index
        mean = df.mean(axis=1)
        # q_05 = df.quantile(q=0.05, axis=1)
        # q_95 = df.quantile(q=0.95, axis=1)

        ax.plot(time, mean, label=f'{comparison} -- {value}', ls=ls, lw=lw, color=color)
        # ax.plot(time, q_95, ls='--', color=color)
        # ax.plot(time, q_05, ls='--', color=color)
        # ax.fill_between(time, q_05, q_95, alpha=0.2, color=color)

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
        """Get the risk at a certain time through the simulation."""
        df = self.first_infection_group if 'first_infection_group' in kwargs else self.risk
        date_time = self.start_time + time
        return df.loc[date_time].values


if __name__ == '__main__':
    pass
