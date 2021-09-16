# pylint: disable=no-member
from hashlib import shake_128
from classes.CTMC_simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import random
from operator import attrgetter

class DTMC_simulation(Simulation):

    def __init__(self, sim_id, start_time, simulation_constants, contam_model, vent_mat_closed):
        super().__init__(sim_id, start_time, simulation_constants, contam_model, vent_mat_closed)
        self.room_volumes = self.contam_model.zones.df['Vol'].astype(float).values
        self.dt = self.time_step.total_seconds()/60**2

    def run(self):
        while self.current_time < self.end_time:
            self.quanta_conc.append(self.calculate_quanta_conc())
            self.Lambda.append(self.infectivity_rate())
            self.assign_infectivity_rate_to_rooms_and_groups()
            self.time.append(self.current_time + self.time_step)

            # calculate R_k
            self.R_k = sum(np.concatenate(
                (self.current_lambda*self.S_arr, self.recover_rate*self.I_arr), axis=0))

            self.checkForInfectionRecover()

            # assign location of next student group
            group_ids = [x.group_id for x in self.rooms]
            for i, id in enumerate(group_ids):
                index = [ x.group_id for x in self.students].index(id)
                getattr(self.students[index], self.next_event_func[i])()

            self.open_close_opening(opening=self.vent_mat_closed_type)
            self.in_school_track.append(self.in_school())


    # def open_close_opening(self, opening):
    #     pass

    def calculate_quanta_conc(self):
        # if the list doesn't exist this is the first entry
        if not hasattr(self, 'quanta_conc'):
            return np.zeros(len(self.rooms))
        elif self.in_school():
            C0 = self.current_quanta_conc

            qI = 0 - self.infection_rates
            return C0 + self.dt/self.room_volumes *(qI + self.vent_matrix.dot(C0))
        elif not self.in_school():
            C0 = self.current_quanta_conc
            return C0 + self.dt/self.room_volumes *(self.vent_matrix.dot(C0))
            

    def checkForInfectionRecover(self):
        self.prob_infection = (self.current_lambda*self.S_arr) * self.dt

        self.prob_recover = (self.recover_rate*self.I_arr) * self.dt
        self.prob_no_event = 1 - self.prob_infection - self.prob_recover
        full_array = np.vstack((self.prob_infection,
                                self.prob_recover,
                                self.prob_no_event))


        if np.all(np.round(np.sum(full_array, axis=0), 3)) != 1.0:
            breakpoint()
            raise ValueError(f'{sum(full_array)} ---> all probabilities should equal 1')
        prob_zones = np.cumsum(full_array, axis=0)
        X = np.random.rand(len(self.rooms))
        event_numbers = (prob_zones >=X).sum(axis=0)
        event_types = ['no_event', 'recovery', 'infection']
        self.next_event_func = [event_types[i-1] for i in event_numbers]

    def plot_quanta_conc(self):
        for i in range(len(self.quanta_conc[0])):
            plt.plot(self.time, [q[i] for q in self.quanta_conc])
        self.shade_school_time(ax=plt.gca())
        plt.show()
        plt.close()
