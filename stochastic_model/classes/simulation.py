import random
from datetime import date, datetime, time, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classes.room import Room
from classes.student_group import Students
from savepdf_tex import savepdf_tex

class Simulation():

    def __init__(self, sim_id, start_time, simulation_constants, contam_model, vent_mat_closed):
        self.sim_id = sim_id
        self.duration = simulation_constants['duration']
        self.recover_rate = simulation_constants['recover_rate']
        self.quanta_gen_rate = simulation_constants['quanta_gen_rate']
        self.school_start = simulation_constants['school_start']
        self.school_end = simulation_constants['school_end']
        self.init_students_per_class = simulation_constants['init_students_per_class']
        self.pulmonary_vent_rate = simulation_constants['pulmonary_vent_rate']
        self.lambda_home = simulation_constants['lambda_home']
        self.plotting_sample_rate = simulation_constants['plotting_sample_rate']
        self.door_open_fraction = simulation_constants['door_open_fraction']
        self.window_open_fraction = simulation_constants['window_open_fraction']
        self.start_time = start_time
        self.end_time = self.start_time + self.duration
        self.time = [self.start_time]
        self.inter_event_time = [0]
        self.contam_model = contam_model
        self.vent_mat_closed = vent_mat_closed['matrix']
        self.vent_mat_closed_type = vent_mat_closed['type']
        self.to_open = True

        self.Lambda = [
            np.ones(shape=len(self.contam_model.zones.df))*self.lambda_home]

        # Init the classroom and student group classes
        self.rooms = [Room(matrix_id=idx, zone_info=data) for idx, data in
                      self.contam_model.zones.df.iterrows()]
        self.students = [Students(init_room=room,
                                  init_students_per_class=self.init_students_per_class)
                         for room in self.rooms]
        self.total_S = sum([students.latest_S for students in self.students])

        self.apply_on_all(seq='rooms',
                          method='get_current_student_group',
                          students=self.students)
        self.first_infection()

    def run(self):
        while self.current_time < self.end_time:
            #check if in school
            # calculate lambda
            self.Lambda.append(self.infectivity_rate())
            # assign infectivity rate to the room and student group
            self.apply_on_all(seq='rooms',
                              method='update_infectivity_rate',
                              infectivity_rates=self.current_lambda)
            self.apply_on_all(seq='students',
                              method='update_infectivity_rate',
                              rooms=self.rooms,
                              infectivity_rates=self.current_lambda)
            # calculate R_k
            self.R_k = sum(np.concatenate(
                (self.current_lambda*self.S_arr, self.recover_rate*self.I_arr), axis=0))
            # inter event time
            tmp_timedelta = self.time_to_next_event()
            self.inter_event_time.append(tmp_timedelta)
            next_school_start_time = self.next_school_start_or_end_time(
                which='start')
            next_school_end_time = self.next_school_start_or_end_time(
                which='end')


            if (not self.in_school() and
                    (self.current_time + tmp_timedelta >= next_school_start_time or
                     self.R_k == 0.0)):
                self.time.append(min([next_school_start_time, self.end_time]))
                self.apply_on_all(seq='students', method='no_event')
            elif (self.in_school() and
                    (self.current_time + tmp_timedelta >= next_school_end_time or
                     self.R_k == 0.0)):

                self.time.append(min([next_school_end_time, self.end_time]))
                self.apply_on_all(seq='students', method='no_event')
            else:
                ## Need to know what type of event is next and location
                # probability of infection in each room
                self.prob_infection = (self.current_lambda*self.S_arr)/self.R_k
                self.prob_recover = (self.recover_rate*self.I_arr)/self.R_k
                self.locationOfNextEvent()
                # assign location of next student group
                room_idx = self.get_room_idx_by_attr(
                    attr='matrix_id', value=self.next_event_matrix_idx)  # index of the room in the list
                group_idx = self.get_student_group_idx_by_attr(
                    attr='room_id', value=self.rooms[room_idx[0]].room_id)  # index of the students in the list
                for idx, student in enumerate(self.students):
                    if idx in group_idx:
                        getattr(student, self.next_event_func)()
                    else:
                        student.no_event()
                self.time.append(self.current_time +
                                 self.current_inter_event_time)
            self.open_close_opening(opening=self.vent_mat_closed_type)
            # self.open_close_opening(opening='window')

    def open_close_opening(self, opening):
        X = random.random()
        dic = {'door': {'fraction': self.door_open_fraction,
                        'open_type': 4,
                        'closed_type': 3},
               'window': {'fraction': self.window_open_fraction,
                          'open_type': 1,
                          'closed_type': 6}
               }
        for k, v in dic.items():
            if k == opening:
                to_open = True if X < v['fraction'] else False
                self.to_open = to_open
                self.apply_on_all(seq='rooms', method='open_close_opening',
                          bool=to_open, opening=opening)
            else:
                self.apply_on_all(seq='rooms', method='open_close_opening',
                          bool=True, opening=k)

        # check if open or not, needs to be more advanced if we are varying this
        # if opening == 'door':
        #     currently_open = self.rooms[0].door_open[-1]
        # elif opening == 'window':
        #     currently_open = self.rooms[0].window_open[-1]

        # if to_open and not currently_open:
        #     # open all the openings
        #     self.contam_model.set_all_flow_paths_of_type_to(search_type_term=opening[1:],
        #                                                 param_dict={'type': dic[opening]['open_type']},
        #                                                 rerun=True,
        #                                                         )
        # elif not to_open and currently_open:
        #     # close all the openings
        #     self.contam_model.set_all_flow_paths_of_type_to(search_type_term=opening[1:],
        #                                         param_dict={'type': dic[opening]['closed_type']},
        #                                         rerun=True,
        #                                                 )


    def first_infection(self):
        infected_group_id = random.choice(
            [room.group_id for room in self.rooms if room.room_type == 'classroom'])
        group_idx = self.get_student_group_idx_by_attr(
            attr='group_id', value=infected_group_id)

        if len(group_idx) != 1:
            raise ValueError(
                'More than one group of students found with the same id!')
        else:
            self.first_infection_group = self.students[group_idx[0]].group_id
            # infect 1 student
            self.students[group_idx[0]].infection(first=True)

    def get_student_group_idx_by_attr(self, attr, value):
        """returns idx of student group whos attribute
        matches the value

        Args:
            attr (string): an attribute of the student class
            value (any): value to check (needs some type checking)

        Returns:
            list: list of matching students
        """
        return [idx for idx, student in enumerate(self.students) if getattr(student, attr) == value]

    def get_room_idx_by_attr(self, attr, value):
        return [idx for idx, room in enumerate(self.rooms) if getattr(room, attr) == value]

    @property
    def current_time(self):
        return self.time[-1]

    @property
    def current_inter_event_time(self):
        return self.inter_event_time[-1]

    @property
    def vent_matrix(self):
        if self.to_open:
            return self.contam_model.vent_mat
        else:
            return self.vent_mat_closed

    def in_school(self, t_delta=timedelta(hours=0)):
        t = self.current_time + t_delta
        return (t.weekday() in [0, 1, 2, 3, 4] and
                t.time() >= self.school_start and
                t.time() < self.school_end)

    def infectivity_rate(self):
        if self.in_school():
            self.quanta_conc = np.linalg.solve(
                self.vent_matrix, self.infection_rates)
            return self.quanta_conc * self.pulmonary_vent_rate
        else:
            return np.ones(len(self.rooms))*self.lambda_home

    @property
    def infection_rates(self):
        return np.array([-(x.latest_I*self.quanta_gen_rate) for x in self.students])

    @property
    def current_lambda(self):
        return self.Lambda[-1]

    def time_to_next_event(self):
        if self.R_k != 0:
            return timedelta(hours=-np.log(random.random())/self.R_k)
        else:
            return timedelta(seconds=0)

    def next_school_start_or_end_time(self, which):
        """[summary]

        Args:
            which (string): options [start, end]

        Returns:
            [type]: [description]
        """
        # week day before start/end
        if (self.current_time.time() < getattr(self, f'school_{which}') and
                self.current_time.weekday() in [0, 1, 2, 3, 4]):
            return self.current_time.replace(hour=getattr(self, f'school_{which}').hour,
                                             minute=getattr(self, f'school_{which}').minute)
        # week day after start/end monday to thursday
        elif (self.current_time.time() >= getattr(self, f'school_{which}') and
                self.current_time.weekday() in [0, 1, 2, 3]):
            return self.current_time.replace(day=self.current_time.day+1,
                                             hour=getattr(
                                                 self, f'school_{which}').hour,
                                             minute=getattr(self, f'school_{which}').minute)
        # its the weekend find the monday times
        else:
            days_ahead = 7 - self.current_time.weekday()
            return self.current_time.replace(day=self.current_time.day+days_ahead,
                                             hour=self.school_start.hour,
                                             minute=0)

    def locationOfNextEvent(self):
        # concat all probabilities
        full_array = np.concatenate(
            (self.prob_infection, self.prob_recover), axis=0)
        # check all probabilities add up to 1.
        if round(sum(full_array), 3) != 1.0:
            breakpoint()
            raise ValueError(
                f'{sum(full_array)} ---> all probabilities should equal 1')
        # reshape into 2d array with infection and recovery on seperate rows
        prob_zones = np.cumsum(full_array).reshape(2, len(self.rooms))
        X = random.random()
        gt_rand = np.where(prob_zones >= X)  # returns 2d index where the
        self.next_event_matrix_idx = gt_rand[1][0]
        if self.next_event_matrix_idx == 5:
            print('Infection in the corridor?')
            breakpoint()
        self.next_event_func = 'infection' if gt_rand[0][0] == 0 else 'recovery'

    def apply_on_all(self, seq, method, *args, **kwargs):
        for obj in getattr(self, seq):
            getattr(obj, method)(*args, **kwargs)

    # this might need to be improved to allow students to move between classrooms
    @property
    def S_arr(self):
        return np.array([student.latest_S for student in self.students])

    @property
    def I_arr(self):
        return np.array([student.latest_I for student in self.students])

    @property
    def R_arr(self):
        return np.array([student.latest_R for student in self.students])

    def generate_dataframes(self):
        self.S_df = self.get_df(param='S')
        self.I_df = self.get_df(param='I')
        self.R_df = self.get_df(param='R')
        self.risk = (self.total_S - self.S_df.sum(axis=1)) / self.total_S
        self.inter_event_time = pd.Series(self.inter_event_time)
        S_init_group = self.S_df[(self.sim_id, self.first_infection_group)]
        self.first_infection_group = (
            self.init_students_per_class - 1 - S_init_group) / (self.init_students_per_class - 1)
        self.door_open_percentage = self.get_opening_open_df(opening='door')
        self.window_open_percentage = self.get_opening_open_df(opening='window')

    def get_df(self, param):
        try:
            df_idx = pd.MultiIndex.from_product([[self.sim_id], [
                                                student.group_id for student in self.students]], names=['sim id', 'student group id'])
            df_tmp = pd.DataFrame(data=[getattr(student, param) for student in self.students],
                                  index=df_idx,
                                  columns=self.time)
            df_tmp = df_tmp.T.resample(self.plotting_sample_rate).pad()
        except ValueError:
            breakpoint()
        return df_tmp.loc[df_tmp.index <= self.end_time]

    def get_opening_open_df(self, opening):
        try:
            df_idx = pd.MultiIndex.from_product(
                [[self.sim_id], [room.room_id for room in self.rooms]], names=['sim id', 'room id'])
            df_tmp = pd.DataFrame(data=[getattr(room, f'{opening}_open') for room in self.rooms],
                                  index=df_idx,
                                  columns=self.time)
            df_tmp = df_tmp.T.resample(self.plotting_sample_rate).pad()
            df_tmp = df_tmp.loc[df_tmp.index <= self.end_time]
        except ValueError:
            breakpoint()
        return df_tmp.mean().mean()

    @property
    def current_infection_risk(self):
        return (self.total_S - self.S_arr.sum()) / self.total_S


####################### plot methods ##############################


    def plot_lambda(self):
        fig = plt.figure()
        fig.autofmt_xdate()
        init_infection_idx = [x.I[0] for x in self.students].index(1)
        t_min = self.start_time.replace(
            hour=self.school_start.hour, minute=self.school_start.minute)
        t_max = self.start_time.replace(
            hour=self.school_end.hour, minute=self.school_end.minute)
        while t_min < self.end_time:
            if t_min.weekday() in [0, 1, 2, 3, 4]:
                plt.axvspan(xmin=t_min, xmax=t_max, color='k', alpha=0.2)
            t_min = t_min + timedelta(days=1)
            t_max = t_max + timedelta(days=1)

        for i in range(len(self.Lambda[0])):
            lw = 1 if i != init_infection_idx else 2
            color = 'k' if i != init_infection_idx else 'r'
            data = [x[i] for x in self.Lambda]
            plt.step(self.time, data, lw=lw, color=color)
        # plt.axhline(self.lambda_home, color='k', lw=3)
        plt.ylabel(r'\$ \lambda \in \{C_ip, \lambda_{home} \} \$')
        # plt.legend()
        plt.tight_layout()
        save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
        savepdf_tex(fig=fig, fig_loc=save_loc,
                name=f'infectivity_rate_example')
        # plt.show()
        plt.close()
        del fig

    def plot_SIR(self):
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        fig.autofmt_xdate()
        init_infection_idx = [x.I[0] for x in self.students].index(1)
        for i, student in enumerate(self.students):
            lw = 1 if i != init_infection_idx else 2
            color = 'k' if i != init_infection_idx else 'r'
            ax[0].step(self.time, student.S, where='post', lw=lw, ls='-',
                     color=color,)
            ax[1].step(self.time, student.I, where='post',
                     lw=lw, ls='--', color=color)
            ax[2].step(self.time, student.R, where='post',
                     lw=lw, ls=':', color=color)
        for ii in range(3):
            ax[ii].set_ylim([0, self.init_students_per_class])

        ax[0].set_ylabel('S')
        ax[1].set_ylabel('I')
        ax[2].set_ylabel('R')
        # plt.legend()
        plt.tight_layout()
        save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
        savepdf_tex(fig=fig, fig_loc=save_loc,
                name=f'SIR_example_per_room')
        # plt.show()
        plt.close()
        del fig

    def plot_SIR_total(self):
        fig = plt.figure()
        for i, X in enumerate(['S','I','R']):
            plt.step(getattr(self, f'{X}_df').index, getattr(self, f'{X}_df').sum(axis=1), where='post',
                     color=f'C{i}', label=X)
        # plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
        savepdf_tex(fig=fig, fig_loc=save_loc,
                name=f'SIR_example_total')
        # plt.show()
        plt.close()
        del fig
