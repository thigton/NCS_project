import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from de_oliveira_droplet_model import DataClass
from de_oliveira_droplet_distribution import get_particle_distribution, get_particle_distribution_parameters

def get_particle_time_series(data, action='speaking',aerosols_only=False, mass=False):
    init_distribution = data.initial_particle_distribution[action]
    if mass:
        init_total = data.droplet_mass.iloc[0,:].mul(init_distribution).sum()
    else:
        init_total = init_distribution.sum()

    displacement_df = data.droplet_displacement
    if aerosols_only:
        displacement_df.mask(data.droplet_diameter > 5e-6, inplace=True)
        # init_distribution = init_distribution[:len(aerosols_dias)]
    still_in_the_air = displacement_df > 1e-6
    
    if mass:
        # breakpoint()
        return still_in_the_air.mul(data.droplet_mass.mul(init_distribution, axis=1)).sum(axis=1)/init_total
    else:
        return still_in_the_air.mul(init_distribution, axis=1).sum(axis=1)/init_total

def get_viable_dose(data, action='speaking'):
    breakpoint()

if __name__ == '__main__':
    mass_bool = True
    aerosol_only_bool = False
    fnames = os.listdir('/Users/Tom/Box/NCS Project/models/epidemic_functions/data_files')
    colour_counter = -1
    for f in fnames:
        with open(f'/Users/Tom/Box/NCS Project/models/epidemic_functions/data_files/{f}', 'rb') as pickle_in:
            data = pickle.load(pickle_in)
        parameters = get_particle_distribution_parameters()
        print(f)
        if data.ventilation_velocity == 0:
            colour_counter += 1
        else:
            continue

        source_params = {'speaking': {'t': 30, 'Q': 0.211}, 'coughing': {
            't': 0.5, 'Q': 1.25}}  # in litres and seconds
        # diameters and particle_numbers are in metres
        diameters, particle_numbers = get_particle_distribution(params=parameters,
                                                                modes=['1', '2', '3'],
                                                                source=source_params,
                                                                plot=False,
                                                                dia_eval=data.droplet_displacement.columns.values*1e6)
        data.initial_particle_distribution = particle_numbers
        # time = data.droplet_displacement.index
        # for i in range(0, data.droplet_displacement.shape[1], 20):
        #     array = data.droplet_displacement.iloc[:,i]
        #     plt.plot(time, array, label=f'd={data.droplet_displacement.columns[i]*1e6:0.0f}$\mu$m')
        # plt.legend()
        # plt.show()
        # plt.close()

        get_viable_dose(data)

        total_number_cough = get_particle_time_series(data, action='coughing', aerosols_only=aerosol_only_bool, mass=mass_bool)
        total_number_speak = get_particle_time_series(data, action='speaking', aerosols_only=aerosol_only_bool, mass=mass_bool)
        plt.plot(total_number_cough,label=f'RH={data.RH} cough', color=f'C{colour_counter}')
        plt.plot(total_number_speak,label=f'RH={data.RH} speak', color=f'C{colour_counter}', ls='--')
    plt.legend()
    plt.xscale('log')
    if mass_bool:
        plt.yscale('log')
        plt.xlim([1,5e3])
        plt.ylim([1e-4,1])
    else:
        plt.xlim([0,3600])
        plt.ylim([0.85,1.01])
    plt.show()
    plt.close()
    # class DataClass():
#     def __init__(self, air_temp, RH, saliva, Lambda,
#                 t_end, vent_u, particle_distribution, results):
#         self.air_temp = air_temp
#         self.RH = RH
#         self.saliva = saliva
#         self.Lambda = Lambda
#         self.simulation_length = t_end
#         self.sim_time_resolution = results[0].shape[0]
#         self.ventilation_velocity = vent_u
#         self.initial_particle_distribution = particle_distribution
#         self.sim_droplet_resolution = len(self.initial_particle_distribution)
#         X_df, v_df, Td_df, md_df, yw_df, Nv_df, D_df = results
#         self.droplet_displacement = X_df
#         self.droplet_velocity = v_df
#         self.droplet_temp = Td_df
#         self.droplet_mass = md_df
#         self.droplet_mass_fraction = yw_df
#         self.droplet_viral_load = Nv_df
#         self.droplet_diameter = D_df

#         self.sim_date = date.today()