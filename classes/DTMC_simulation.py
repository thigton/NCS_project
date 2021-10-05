# pylint: disable=no-member
from classes.CTMC_simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson

class DTMC_simulation(Simulation):

    def __init__(self, sim_id, start_time, simulation_constants, contam_model, opening_method,  movement_method):
        super().__init__(sim_id, start_time, simulation_constants, contam_model, opening_method, movement_method)
        self.room_volumes = self.contam_model.zones.df['Vol'].astype(float).values
        self.dt = self.time_step.total_seconds()/60**2

    def run(self):
        while self.current_time < self.end_time:
            self.quanta_conc_integration_results = self.quanta_conc_integration()
            self.quanta_conc.append(self.calculate_quanta_conc())
            self.Lambda.append(self.infectivity_rate())
            self.assign_infectivity_rate_to_rooms_and_groups()
            self.checkForInfectionRecover()
            
            self.time.append(self.current_time + self.time_step)


            # assign location of next student group
            for student, func in zip(self.students, self.next_event_func):
                getattr(student, func)()
            # group_ids = [x.current_group_id for x in self.rooms]
            # for i, id in enumerate(group_ids):
            #     index = [ x.group_id for x in self.students].index(id)
            #     getattr(self.students[index], self.next_event_func[i])()
            #     if self.current_time.minute == 10.0:
            #         breakpoint()


            # open/close doors
            if self.opening_method:
                all_open_override = False if self.in_school() else True
                self.open_close_opening(opening_method=self.opening_method, all_open_override=all_open_override)

            # move groups
            if self.movement_method and self.current_time.minute == 0.0 and self.in_school():
                self.move_students()
                
            self.in_school_track.append(self.in_school())
                

    def move_students(self):
        if self.movement_method == 'change_rooms_in_group':
            # Q is in the order of self.contam_model.zones.df['Z#']
            room_ids = [x.room_id for x in self.rooms if x.room_type == 'classroom']
            corridors = [(x.room_id , x.matrix_id) for x in self.rooms if x.room_type == 'corridor']
            np.random.shuffle(room_ids)
            room_ids.insert(corridors[0][1], corridors[0][0])

            # assign new room id to groups
            for new_room, student in zip(room_ids, self.students):
                student.changeClassroom(new_room)

            # re order students in list so room ids match use matrix id.
            new_idx = [self.get_room_idx_by_attr('room_id', x)[0] for x in room_ids]
            self.students = [self.students[i] for i in new_idx]
            
            self.apply_on_all(seq='rooms',
                            method='update_group_ids',
                            students=self.students)


    def checkForInfectionRecover(self):
        self.prob_infection = (self.current_lambda*self.S_arr) * self.dt

        self.prob_recover = (self.recover_rate*self.I_arr) * self.dt
        self.prob_no_event = 1 - self.prob_infection - self.prob_recover
        full_array = np.vstack((self.prob_infection,
                                self.prob_recover,
                                self.prob_no_event))


        if np.all(np.round(np.sum(full_array, axis=0), 3)) != 1.0:
            raise ValueError(f'{sum(full_array)} ---> all probabilities should equal 1')
        prob_zones = np.cumsum(full_array, axis=0)
        X = np.random.rand(len(self.rooms))
        event_numbers = (prob_zones >=X).sum(axis=0)
        event_types = ['no_event', 'recovery', 'infection']
        self.next_event_func = [event_types[i-1] for i in event_numbers]



    def quanta_conc_integration(self):
        def integration(t, C, *args):
            Q , infection_rate, volume = args
            dC = infection_rate + np.matmul(Q, C)
            return dC/volume

        C0 = self.current_quanta_conc
        if self.in_school():
            qI = 0 - self.infection_rates
        elif not self.in_school():
            qI = np.zeros(len(self.rooms))
        return solve_ivp(fun=integration,
                 y0=C0,
                 t_span=[0,self.dt],
                args=(self.vent_matrix, qI, self.room_volumes)
                )


    def calculate_quanta_conc(self,):
        # if the list doesn't exist this is the first entry
        if not hasattr(self, 'quanta_conc'):
            return np.zeros(len(self.rooms))
        else:
            return self.quanta_conc_integration_results.y[:,-1]


    def infectivity_rate(self):
        if not hasattr(self, 'Lambda'):
            return np.zeros(len(self.rooms))
        if self.in_school():
            try:
                total_exposure = simpson(self.quanta_conc_integration_results.y*self.pulmonary_vent_rate,
                                self.quanta_conc_integration_results.t, axis=1)
            except ValueError as e:
                print(e)
                print('simpsons rule : Value error')
            return total_exposure/self.dt
        else:
            return np.ones(len(self.rooms))*self.lambda_home

    def plot_quanta_conc(self):
        for i in range(len(self.quanta_conc[0])):
            plt.plot(self.time, [q[i] for q in self.quanta_conc])
        self.shade_school_time(ax=plt.gca())
        plt.show()
        plt.close()

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
                plt.plot(self.time, data, lw=lw, color=color)
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



