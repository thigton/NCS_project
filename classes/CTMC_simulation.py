# pylint: disable=no-member
import random
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classes.room import Room
from classes.student_group import Students
from classes.contam_model import binary_ref_to_int


class Simulation():

    def __init__(self, sim_id, start_time, simulation_constants,
                 contam_model, opening_method, movement_method):
        self.sim_id = sim_id
        for attr, value in simulation_constants.items():
            setattr(self, attr, value)
        self.start_time = start_time
        self.end_time = self.start_time + self.duration
        self.time = [self.start_time]
        self.inter_event_time = [0]
        self.contam_model = contam_model
        self.opening_method = opening_method
        self.movement_method = movement_method
        self.to_open = True
        self.in_school_track = [self.in_school()]

        # Init the classroom and student group classes
        self.rooms = [Room(matrix_id=idx, zone_info=data) for idx, data in
                      self.contam_model.zones.df.iterrows()]
        if self.opening_method:
            self.open_close_opening( opening_method=self.opening_method)
        self.students = [Students(init_room=room,
                                  init_students_per_class=self.init_students_per_class)
                         for room in self.rooms]
        self.total_S = sum([students.latest_S for students in self.students])

        self.apply_on_all(seq='rooms',
                          method='update_group_ids',
                          students=self.students)
        self.first_infection()
        self.quanta_conc = [self.calculate_quanta_conc()]
        self.Lambda = [self.infectivity_rate()]
        self.assign_infectivity_rate_to_rooms_and_groups()


    def run(self):
        while self.current_time < self.end_time:
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
            if self.R_k == 0:
                # premature end either everyone is infected or recovered
                self.time.append(self.end_time)
                self.apply_on_all(seq='students', method='no_event')
            elif (not self.in_school() and
                    self.current_time + tmp_timedelta >= next_school_start_time):
                self.time.append(min([next_school_start_time, self.end_time]))
                self.apply_on_all(seq='students', method='no_event')
                all_open_override = False

            elif (self.in_school() and
                    self.current_time + tmp_timedelta >= next_school_end_time):
                self.time.append(min([next_school_end_time, self.end_time]))
                self.apply_on_all(seq='students', method='no_event')
                all_open_override = True
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
                all_open_override = False

            if self.opening_method:
                self.open_close_opening(opening_method=self.opening_method,
                                        all_open_override=all_open_override)
            # calculate lambda for the next check
            self.quanta_conc.append(self.calculate_quanta_conc())
            self.Lambda.append(self.infectivity_rate())
            self.assign_infectivity_rate_to_rooms_and_groups()
            self.in_school_track.append(self.in_school())



    def open_close_opening(self, opening_method, all_open_override=False):

        if 'door' in opening_method:
            if not hasattr(self.contam_model, 'all_door_matrices'):
                print('Large ventilation matrix not found...... will run Contam directly')
                self.contam_model.generate_all_ventilation_matrices_for_all_door_open_close_combination()
                self.contam_model.load_all_vent_matrices()


            all_doors = self.contam_model.get_all_flow_paths_of_type(search_type_term='oor')
            if all_open_override:
                doors_open = np.array([True]*len(all_doors))
            elif opening_method == 'all_doors_only_random':
                X = np.random.rand(len(all_doors))
                doors_open = X < self.door_open_fraction
            elif opening_method == 'internal_doors_only_random':
                internal_doors_paths = all_doors[(all_doors['n#'] !='-1') & (all_doors['m#'] !='-1')]['P#'].values
                # X = np.random.rand(len(internal_doors))
                # doors_open = X < self.door_open_fraction
                # idx = self.contam_model.external_door_matrix_idx
                def insert_external_doors(row, internal_doors_paths):
                    X = random.random()
                    return X < self.door_open_fraction if row['P#'] in internal_doors_paths else True
                doors_open = all_doors.apply(lambda x: insert_external_doors(x, internal_doors_paths),
                                             axis=1).values


            binary_ref = ''.join((doors_open).astype(int).astype(str))
            self.contam_model.set_big_vent_matrix_idx(idx=binary_ref_to_int(binary_ref))
            self.assign_door_position_to_room_cls(doors_open, all_doors)
            del all_doors, binary_ref, doors_open





    def assign_door_position_to_room_cls(self, open_bool, opening_df):
        classrooms = [x for x in self.rooms if x.room_type == 'classroom']
        corridors = [x for x in self.rooms if x.room_type == 'corridor']
        for classroom in classrooms:
            boolean = open_bool[(opening_df['n#'] == classroom.room_id)|(opening_df['m#'] == classroom.room_id)][0]
            classroom.open_close_opening(boolean, 'door')
        for corridor in corridors:
            boolean = open_bool[((opening_df['n#'] == corridor.room_id) & (opening_df['m#'] == '-1')) |
                                 ((opening_df['m#'] == corridor.room_id) & (opening_df['n#'] == '-1'))]
            corridor.open_close_opening(boolean, 'door')


    @property
    def vent_matrix(self):
        if hasattr(self.contam_model, 'all_door_matrices'):
            return self.contam_model.all_door_matrices[self.contam_model.big_vent_matrix_idx]
        else:
            return self.contam_model.vent_mat

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
            [room.current_group_id for room in self.rooms if room.room_type == 'classroom'])
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



    def in_school(self, t_delta=timedelta(hours=0)):
        t = self.current_time + t_delta
        return (t.weekday() in [0, 1, 2, 3, 4] and
                t.time() >= self.school_start and
                t.time() < self.school_end)

    @property
    def current_quanta_conc(self):
        return self.quanta_conc[-1]

    def calculate_quanta_conc(self):
        if self.in_school():
            return np.linalg.solve(
                    self.vent_matrix, self.infection_rates)
        else:
            return np.zeros(len(self.rooms))

    def infectivity_rate(self):
        if self.in_school():
            return self.current_quanta_conc * self.pulmonary_vent_rate
        else:
            return np.ones(len(self.rooms))*self.lambda_home

    @property
    def infection_rates(self):
        return np.array([-(x.latest_I*self.quanta_gen_rate) for x in self.students])

    @property
    def current_lambda(self):
        return self.Lambda[-1]

    def assign_infectivity_rate_to_rooms_and_groups(self):
        self.apply_on_all(seq='rooms',
                              method='update_infectivity_rate',
                              infectivity_rates=self.current_lambda)
        self.apply_on_all(seq='students',
                              method='update_infectivity_rate',
                              rooms=self.rooms,
                              infectivity_rates=self.current_lambda)

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
        # week day before start/end i.e. the next time is the same day
        if (self.current_time.time() < getattr(self, f'school_{which}') and
                self.current_time.weekday() in [0, 1, 2, 3, 4]):
            tmp_time = self.current_time
        # week day after start/end monday to thursday i.e. the next time is tomorrow
        elif (self.current_time.time() >= getattr(self, f'school_{which}') and
                self.current_time.weekday() in [0, 1, 2, 3]):
            tmp_time = self.current_time + timedelta(days=1)

        # its the weekend find the monday times
        else:
            days_ahead = 7 - self.current_time.weekday()
            tmp_time = self.current_time + timedelta(days=days_ahead)
        # time = tmp_time.replace(hour=getattr(self, f'school_{which}').hour,
        #                          minute=getattr(self, f'school_{which}').minute,
        #                          second=getattr(self, f'school_{which}').second,
        #                          microsecond=getattr(self, f'school_{which}').microsecond,)
        # breakpoint()

        return tmp_time.replace(hour=getattr(self, f'school_{which}').hour,
                                 minute=getattr(self, f'school_{which}').minute,
                                 second=getattr(self, f'school_{which}').second,
                                 microsecond=getattr(self, f'school_{which}').microsecond,)


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
        if  'door' in self.opening_method:
            self.door_open_df, self.door_open_percentage = self.get_opening_open_df(opening='door')

        # self.window_open_df, self.window_open_percentage = self.get_opening_open_df(opening='window')

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
        return df_tmp.loc[df_tmp.index < self.end_time]

    def get_opening_open_df(self, opening, in_school_only=True):
        try:
            df_idx = pd.MultiIndex.from_product(
                [[self.sim_id], [room.room_id for room in self.rooms if room.room_type == 'classroom']], names=['sim id', 'room id'])
            df_tmp = pd.DataFrame(data=[getattr(room, f'{opening}_open') for room in self.rooms if room.room_type == 'classroom'],
                                  index=df_idx,
                                  columns=self.time)
            df_tmp = df_tmp.T.resample(self.plotting_sample_rate).pad()
            df_tmp = df_tmp.loc[df_tmp.index < self.end_time]
            if in_school_only:
                in_school_series = pd.Series(data=self.in_school_track, index=self.time)
                in_school_series = in_school_series.resample(self.plotting_sample_rate).pad()
                in_school_series = in_school_series.loc[in_school_series.index < self.end_time]

        except ValueError:
            breakpoint()
        return df_tmp, df_tmp[in_school_series.values].mean().mean()

    @property
    def current_infection_risk(self):
        return (self.total_S - self.S_arr.sum()) / self.total_S


####################### plot methods ##############################


    def plot_lambda(self):
        fig = plt.figure()
        fig.autofmt_xdate()
        init_infection_idx = [x.I[0] for x in self.students].index(1)
        self.shade_school_time(ax=plt.gca())

        for i in range(len(self.Lambda[0])):
            lw = 1 if i != init_infection_idx else 2
            color = 'k' if i != init_infection_idx else 'r'
            data = [x[i] for x in self.Lambda]
            try:
                plt.step(self.time, data, where='post', lw=lw, color=color)
            except ValueError:
                breakpoint()
        # plt.axhline(self.lambda_home, color='k', lw=3)
        plt.ylabel(r'\$ \lambda \in \{C_ip, \lambda_{home} \} \$')
        # plt.legend()
        plt.tight_layout()
        # save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
        # savepdf_tex(fig=fig, fig_loc=save_loc,
        #         name=f'infectivity_rate_example')
        plt.show()
        plt.close()
        del fig

    def plot_SIR(self):
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        plt.suptitle(f'door fraction : {self.door_open_percentage:0.1%}')
        fig.autofmt_xdate()
        init_infection_idx = [x.I[0] for x in self.students].index(1)
        for i, student in enumerate(self.students):
            lw = 1 if i != init_infection_idx else 2
            color = 'k' if i != init_infection_idx else 'r'
            ax[0].step(self.time, student.S, where='post', lw=lw, ls='-',
                     color=color,)
            ax[1].step(self.time, student.I, where='post',
                     lw=lw, ls='-', color=color)
            ax[2].step(self.time, student.R, where='post',
                     lw=lw, ls='-', color=color)
        for ii in range(3):
            ax[ii].set_ylim([0, self.init_students_per_class])
            ax[ii].fill_between(x=self.door_open_df.index, y1 = self.door_open_df.mean(axis=1),
                           step="post", alpha=0.5, color='r')
            self.shade_school_time(ax[ii])


        ax[0].set_ylabel('S')
        ax[1].set_ylabel('I')
        ax[2].set_ylabel('R')
        # plt.legend()
        plt.tight_layout()
        save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
        # savepdf_tex(fig=fig, fig_loc=save_loc,
        #         name=f'SIR_example_per_room')
        plt.show()
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
        # savepdf_tex(fig=fig, fig_loc=save_loc,
                # name=f'SIR_example_total')
        plt.show()
        plt.close()
        del fig

    def shade_school_time(self, ax):
        t_min = self.start_time.replace(
            hour=self.school_start.hour, minute=self.school_start.minute)
        t_max = self.start_time.replace(
            hour=self.school_end.hour, minute=self.school_end.minute)
        while t_min < self.end_time:
            if t_min.weekday() in [0, 1, 2, 3, 4]:
                ax.axvspan(xmin=t_min, xmax=t_max, color='k', alpha=0.2)
            t_min = t_min + timedelta(days=1)
            t_max = t_max + timedelta(days=1)


    def plot_quanta_conc(self):
        for i in range(len(self.quanta_conc[0])):
            plt.step(self.time, [q[i] for q in self.quanta_conc], where='post')
        self.shade_school_time(ax=plt.gca())
        plt.show()
        plt.close()