import random
from datetime import date, datetime, time, timedelta

import matplotlib.pyplot as plt
import numpy as np

from room_cls import Room
from student_group_cls import Students


class Simulation():

    def __init__(self, sim_id, simulation_constants, contam_model):
        self.sim_id = sim_id
        self.duration = simulation_constants['duration']
        self.recover_rate = simulation_constants['recover_rate']
        self.quanta_gen_rate = simulation_constants['quanta_gen_rate']
        self.school_start = simulation_constants['school_start']
        self.school_end = simulation_constants['school_end']
        self.init_students_per_class = simulation_constants['init_students_per_class']
        self.pulmonary_vent_rate = simulation_constants['pulmonary_vent_rate']
        self.lambda_home = simulation_constants['lambda_home']
        self.start_time = datetime(year=2021, month=3, day=1, hour=0, minute=0, second=0)
        self.end_time = self.start_time + self.duration
        self.time = [self.start_time]
        self.inter_event_time = [0]
        self.contam_model = contam_model
        self.Lambda = [np.ones(shape=len(self.contam_model.zones.df))*self.lambda_home]

        # Init the classroom and student group classes
        self.rooms = [Room(matrix_id=idx, zone_info=data) for idx, data in \
                             self.contam_model.zones.df.iterrows()]
        self.students = [Students(init_room=room,
                         init_students_per_class=self.init_students_per_class)
                            for room in self.rooms]
        self.apply_on_all(seq='rooms',
                          method='get_current_student_group',
                          students=self.students)
        self.first_infection()

    def first_infection(self):
        infected_group_id = random.choice([room.group_id for room in self.rooms if room.room_type == 'classroom'])
        group_idx = self.get_student_group_idx_by_attr(attr='group_id', value=infected_group_id)
        if len(group_idx) != 1:
            raise ValueError('More than one group of students found with the same id!')
        else:
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
    def current_lambda(self):
        return self.Lambda[-1]

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
            self.R_k = sum(np.concatenate((self.current_lambda*self.S_arr, self.recover_rate*self.I_arr), axis=0))
            # inter event time
            self.inter_event_time.append(self.time_to_next_event())
            if ((not self.in_school() and 
                    self.current_time + self.current_inter_event_time > self.next_school_start_time) or
                    (self.in_school() and 
                    self.current_time + self.current_inter_event_time > self.next_school_end_time)):
                self.time.append(self.next_school_start_time)
                # breakpoint()
                # update S I R so that it hasn't changed in this time
                self.apply_on_all(seq='students', method='no_event')
                continue

            # probability of infection in each room
            self.prob_infection = (self.current_lambda*self.S_arr)/self.R_k
            self.prob_recover = (self.recover_rate*self.I_arr)/self.R_k
            # assign location of next student group
            self.locationOfNextEvent()
            room_idx = self.get_room_idx_by_attr(attr='matrix_id', value=self.next_event_matrix_idx) # index of the room in the list
            group_idx = self.get_student_group_idx_by_attr(attr='room_id', value=self.rooms[room_idx[0]].room_id) # index of the students in the list
            for idx, student in enumerate(self.students):
                if idx in group_idx:
                    getattr(student, self.next_event_func)()
                else:
                    student.no_event()
            self.time.append(self.current_time + self.current_inter_event_time)
            # print(self.current_inter_event_time)
            print(self.I_arr)
            print(self.R_arr)
            # print(self.next_event_func)
            # print(self.time)
            print(self.current_time)
            # breakpoint()

    @property
    def infection_rates(self):
        return np.array([-(x.latest_I*self.quanta_gen_rate) for x in self.students])

    @property
    def vent_matrix(self):
        return self.contam_model.vent_mat

    
    def in_school(self, t_delta=timedelta(hours=0)):
        t = self.current_time + t_delta
        return (t.weekday() in [0, 1, 2, 3, 4] and 
        t.time() >= self.school_start and 
        t.time() < self.school_end)


    def infectivity_rate(self):
        if self.in_school():
            self.quanta_conc = np.linalg.solve(self.vent_matrix, self.infection_rates)
            return self.quanta_conc * self.pulmonary_vent_rate
        else:
            return np.ones(len(self.rooms))*self.lambda_home
    
    
    def time_to_next_event(self):
        return timedelta(hours=-np.log(random.random())/self.R_k)

    @property
    def next_school_start_time(self):
        # before 9am monday to friday
        if (self.current_time.time() < self.school_start and 
                self.current_time.weekday() in [0,1,2,3,4]):
            return self.current_time.replace(hour=self.school_start.hour, minute=0)
        # after 9am monday to thursday
        elif (self.current_time.time() >= self.school_start and 
                self.current_time.weekday() in [0,1,2,3]):
            return self.current_time.replace(day=self.current_time.day+1,
                                             hour=self.school_start.hour,
                                             minute=0)
        # its the weekend find the next monday morning
        else:
            days_ahead = 7 - self.current_time.weekday()
            return self.current_time.replace(day=self.current_time.day+days_ahead,
                                             hour=self.school_start.hour,
                                             minute=0)
    @property
    def next_school_end_time(self):
        return self.next_school_start_time.replace(hour=self.school_end.hour)

    def locationOfNextEvent(self):
        full_array = np.concatenate((self.prob_infection, self.prob_recover), axis=0) # concat all probabilities
        if round(sum(full_array),3) != 1.0: #check all probabilities add up to 1.
            raise ValueError(f'{sum(full_array)} ---> all probabilities should equal 1')
        prob_zones = np.cumsum(full_array).reshape(2,len(self.rooms)) # reshape into 2d array with infection and recovery on seperate rows
        X = random.random()
        gt_rand = np.where(prob_zones >= X) # returns 2d index where the
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


    def plot_lambda(self):
        for i in range(len(self.Lambda[0])):
            data = [x[i] for x in self.Lambda]
            plt.plot(self.time, data, label=f'room {i}')
        print([x.I[0] for x in self.students])
        plt.axhline(self.lambda_home)
        plt.legend()
        plt.show()
        plt.close()

    def plot_SIR(self):
        for i, student in enumerate(self.students):
            plt.plot(self.time, student.S, ls='-', color=f'C{i}', label=f'group {student.group_id}')
            plt.plot(self.time, student.I, ls='--', color=f'C{i}')
            plt.plot(self.time, student.R, ls=':', color=f'C{i}')
        plt.legend()
        plt.show()
        plt.close()