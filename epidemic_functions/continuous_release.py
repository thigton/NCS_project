import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from de_oliveira_droplet_model import DataClass
from de_oliveira_droplet_distribution import (
    get_particle_distribution, get_particle_distribution_parameters)


if __name__ == '__main__':
    action = 'speaking'

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
        
        teval = data.droplet_displacement.index.values
        t_0 = teval[0]               # s ... initial time
        t_end = teval[-1]           # s ... end of simulation time
        delta_t = teval[1]-teval[0]
        delta_volume = source_params[action]['Q']*1e-3 * delta_t # volume of speech in time step
        delta_numbers = pd.DataFrame(np.vstack([particle_numbers[action]*delta_volume]*len(teval)), index=teval, columns=diameters) # the number of droplets per diameter each time step
        delta_numbers.loc[delta_numbers.index > 30] = 0
        X_df = pd.DataFrame(index=teval, columns=diameters)
        v_df = pd.DataFrame(index=teval, columns=diameters)
        Td_df = pd.DataFrame(index=teval, columns=diameters)
        md_df = pd.DataFrame(index=teval, columns=diameters)
        yw_df = pd.DataFrame(index=teval, columns=diameters)
        Nv_df = pd.DataFrame(index=teval, columns=diameters)
        # D_df = pd.DataFrame(np.vstack([droplet_sizes]*len(teval)), index=teval, columns=droplet_sizes)
        PFUs = []
        for t in teval:
            print(f't:{t:0.1f}secs', end='\r')
            # when t is equal to 4 and the initial release was 2 
            # you want to grab the 2 front the data
            for df, attr in zip([X_df, v_df, Td_df, md_df, yw_df, Nv_df],
                                ['droplet_displacement', 'droplet_velocity',
                                 'droplet_temp', 'droplet_mass','droplet_mass_fraction',
                                'droplet_viral_load']):
                tmp = getattr(data, attr).set_index(keys=t-getattr(data, attr).index)
                tmp = tmp.loc[tmp.index >=0]
                df.loc[tmp.index] = tmp
            PFUs.append(Nv_df.mul(delta_numbers, axis=1).sum().sum())
        plt.plot(teval[:len(PFUs)], PFUs, label=f)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        plt.close()

 