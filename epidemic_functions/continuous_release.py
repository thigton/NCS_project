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
        '/home/tdh17/Documents/BOX/NCS Project/models/epidemic_functions/data_files')
    colour_counter = -1
    fig = plt.figure(figsize=(15, 15))
    for f in fnames:
        with open(f'/home/tdh17/Documents/BOX/NCS Project/models/epidemic_functions/data_files/{f}', 'rb') as pickle_in:
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
        diameters, particle_numbers = get_particle_distribution(params=parameters,
                                                                modes=[
                                                                    '1', '2', '3'],
                                                                source=source_params,
                                                                plot=False,
                                                                dia_eval=data.droplet_displacement.columns.values*1e6)

        teval = data.droplet_displacement.index.values
        t_0 = teval[0]               # s ... initial time
        t_end = teval[-1]           # s ... end of simulation time
        delta_t = teval[1]-teval[0]
        # volume of speech in time step
        delta_volume = source_params[action]['Q']*1e-3 * delta_t
        t_stop = 60
        emit_schedule = teval < t_stop
        delta_numbers = pd.DataFrame(index=data.droplet_displacement.columns)
        # X_df = pd.DataFrame(index=teval, columns=diameters)
        # Nv_df = pd.DataFrame(index=teval, columns=diameters)
        # dia_df = pd.DataFrame(index=teval, columns=diameters)
        # D_df = pd.DataFrame(np.vstack([droplet_sizes]*len(teval)), index=teval, columns=droplet_sizes)
        PFUs = []
        pcolor_df = pd.DataFrame()
        for i, t in enumerate(teval):
            print(f't:{t:0.1f}secs', end='\r')
            delta_numbers.columns = delta_numbers.columns + delta_t
            if emit_schedule[i]:
                delta_numbers[0] = particle_numbers[action]*delta_volume

            # X_df = data.droplet_displacement.loc[teval <= t, :]
            # Nv_df = data.droplet_viral_load.loc[teval <= t, :]
            # dia_df = data.droplet_diameter.loc[teval <= t, :]
            # new_df = pd.concat([data.droplet_displacement.loc[delta_numbers.columns, :].stack(),
            #                     data.droplet_viral_load.loc[delta_numbers.columns, :].mul(delta_numbers.T).stack()],
            #                    axis=1,
            #                    )
            # new_df.rename({0: 'x', 1: 'Nv'}, axis=1, inplace=True)

            # # new_df.sort_values('x', axis=0, ascending=False, inplace=True)
            # new_df = new_df.groupby(
            #     pd.cut(new_df["x"], np.arange(1, 2.001, 0.001))).sum()

            # new_df.columns = ['', t]

            # pcolor_df = pd.concat([pcolor_df, new_df[t]], axis=1)

            Nv_in_breathing_zone = data.droplet_viral_load.loc[teval <= t, :].mask(
                (data.droplet_displacement.loc[teval <= t, :] < 1.2) | (data.droplet_displacement.loc[teval <= t, :] > 1.8))
            Nv_in_breathing_zone = Nv_in_breathing_zone.loc[Nv_in_breathing_zone.index > t-t_stop]
            PFUs.append(Nv_in_breathing_zone.mul(delta_numbers.T).sum().sum())
            # if t == 300:
            #     break
        # pcolor_df.index = pd.IntervalIndex(pcolor_df.index).mid
        # time_mat, midpoint_mat = np.meshgrid(
            # teval[np.where(teval==t_stop)[0][0]:len(PFUs)], pcolor_df.index)
        # plt.pcolormesh(time_mat, midpoint_mat, pcolor_df.loc[:,pcolor_df.columns >= t_stop].to_numpy(),
        #                shading='gouraud', vmin=0, vmax=2000)
        # plt.show()
        # plt.close()
        plt.plot(teval[:len(PFUs)], PFUs,
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
    # plt.show()
    savepdf_tex(fig, fig_loc='/home/tdh17/Documents/BOX/NCS Project/models/epidemic_functions/',name='speak_viral_load')
    plt.close()
