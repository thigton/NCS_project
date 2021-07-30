import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from de_oliveira_droplet_model import DataClass
from de_oliveira_droplet_distribution import (
    get_particle_distribution, get_particle_distribution_parameters)

from savepdf_tex import savepdf_tex




if __name__ == '__main__':

    NV_ID10_range = (20, 83)
    NV_ID50_range = (130, 530)
    NV_ID100 = 1000
    action = 'speaking'
    fnames = os.listdir(
        '/Users/Tom/Box/NCS Project/models/epidemic_functions/data_files')
    colour_counter = -1
    fig = plt.figure(figsize=(15, 15))
    for f in fnames:
        with open(f'/Users/Tom/Box/NCS Project/models/epidemic_functions/data_files/{f}', 'rb') as pickle_in:
            data = pickle.load(pickle_in)
        parameters = get_particle_distribution_parameters()
        print(f)
        if data.RH == 0.6:
            colour_counter += 1
        else:
            continue

        source_params = {'speaking': {'t': 30, 'Q': 0.211},
                         'coughing': {'t': 0.5, 'Q': 1.25}}  # in litres and seconds
        # diameters and particle_numbers are in metres
        teval = data.droplet_displacement.index.values
        t_0 = teval[0]               # s ... initial time
        t_end = teval[-1]           # s ... end of simulation time
        delta_t = teval[1]-teval[0]
        # volume of speech in time step
        delta_volume = source_params[action]['Q']*1e-3 * delta_t
        t_stop = 30
        emit_schedule = teval < t_stop
        # delta_numbers = pd.DataFrame(index=data.droplet_displacement.columns)

        delta_numbers = data.initial_particle_distribution[action]*delta_volume
        Nv_in_breathing_zone_by_diameter = data.droplet_viral_load.mask( (data.droplet_displacement < 1.2) | (data.droplet_displacement > 1.8))
        Nv_in_breathing_zone_base = Nv_in_breathing_zone_by_diameter.mul(delta_numbers.T).sum(axis=1).values

        for i, t in enumerate(teval):
            print(f't:{t:0.1f}secs', end='\r')
            if emit_schedule[i]:
                if i == 0:
                    Nv_in_breathing_zone = Nv_in_breathing_zone_base
                else:
                    Nv_in_breathing_zone = Nv_in_breathing_zone + np.concatenate((np.zeros(shape=i), Nv_in_breathing_zone_base[:-i]), axis=0)



        plt.plot(teval, Nv_in_breathing_zone,
                 label=f'u={data.ventilation_velocity:0.3f}', color=f'C{colour_counter}')
    plt.axhspan(ymin=NV_ID10_range[0],
                ymax=NV_ID10_range[1], color='r', alpha=0.5)
    plt.axhspan(ymin=NV_ID50_range[0],
                ymax=NV_ID50_range[1], color='r', alpha=0.5)
    plt.axhline(NV_ID100, color='r')
    plt.xscale('log')
    plt.yticks([1e0,1e2,1e4,1e6,1e8], [r'\$10^0\$', r'\$10^2\$', r'\$10^4\$', r'\$10^6\$', r'\$10^8\$'])
    plt.xlabel('time (secs)')
    plt.yscale('log')
    plt.ylabel('viral load between 1.2 and 1.8 m (PFUs)')
    plt.legend()
    plt.show()
    # savepdf_tex(fig, fig_loc='/home/tdh17/Documents/BOX/NCS Project/models/epidemic_functions/',name='speak_viral_load')
    plt.close()
