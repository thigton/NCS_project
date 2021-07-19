import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from de_oliveira_droplet_model import DataClass
from de_oliveira_droplet_distribution import (
    get_particle_distribution, get_particle_distribution_parameters)


if __name__ == '__main__':
    NV_ID10_range = (20, 83)
    NV_ID50_range = (130, 530)
    action = 'speaking'
    fnames = os.listdir('/Users/Tom/Box/NCS Project/models/epidemic_functions/data_files')
    colour_counter = -1
    fig = plt.figure(figsize=(15,15))
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
        t_stop = 30
        emit_schedule = teval < t_stop
        
        delta_numbers = pd.DataFrame(index=data.droplet_displacement.columns)
        # X_df = pd.DataFrame(index=teval, columns=diameters)
        # Nv_df = pd.DataFrame(index=teval, columns=diameters)
        # dia_df = pd.DataFrame(index=teval, columns=diameters)
        # D_df = pd.DataFrame(np.vstack([droplet_sizes]*len(teval)), index=teval, columns=droplet_sizes)
        PFUs = []
        for i, t in enumerate(teval):
            delta_numbers.columns = delta_numbers.columns + delta_t
            if emit_schedule[i]:
                delta_numbers[0] = particle_numbers[action]*delta_volume
            print(f't:{t:0.1f}secs', end='\r')
            X_df = data.droplet_displacement.loc[teval <= t, :]
            Nv_df = data.droplet_viral_load.loc[teval <= t, :]
            dia_df = data.droplet_diameter.loc[teval <= t, :]
            # for df, attr in zip([X_df, Nv_df, dia_df],
            #                     ['droplet_displacement','droplet_viral_load', 'droplet_diameter']):
            #     tmp = getattr(data, attr).set_index(keys=t-getattr(data, attr).index)
            #     tmp = tmp.loc[tmp.index >=0]
            #     df.loc[tmp.index] = tmp


            Nv_in_breathing_zone = Nv_df.mask((X_df < 1.2) | (X_df > 1.8))
            Nv_in_breathing_zone = Nv_in_breathing_zone.loc[Nv_in_breathing_zone.index > t-t_stop]
            PFUs.append(Nv_in_breathing_zone.mul(delta_numbers.T).sum().sum())
            
        plt.plot(teval[:len(PFUs)], PFUs, label=f, color=f'C{colour_counter}')
        plt.axhspan(ymin=NV_ID10_range[0], ymax=NV_ID10_range[1], color='r',alpha=0.5)
        plt.axhspan(ymin=NV_ID50_range[0], ymax=NV_ID50_range[1], color='r',alpha=0.5)
        plt.xscale('log')
        plt.xlabel('time (secs)')
        plt.yscale('log')
        plt.ylabel('viral load between 1.2 and 1.8 m (PFUs)')
        plt.legend()
        plt.show()
        plt.close()

 