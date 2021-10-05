"""Run script to produce all the ventilation matrices for all combinations of door openings.
"""
import itertools
from datetime import time, timedelta

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from classes.contam_model import ContamModel
from classes.stocastic_model import StocasticModel
from classes.weather import Weather
from savepdf_tex import savepdf_tex
import pickle

wind_dir = 90.0
wind_speeds = np.array([ 5.0,15.0, ])
amb_temp = np.array([5.0, 10.0, ])
window_height = 1.0


contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                        'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                        'name': 'school_corridor'}

# init contam model
contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
                           contam_dir=contam_model_details['prj_dir'],
                           project_name=contam_model_details['name'])


fields = ['temp', 'wind_speed', 'Q_fresh']
# df = pd.DataFrame(columns=)
for speed, temp in itertools.product(wind_speeds, amb_temp):

    print(speed, temp)
    weather_params = Weather(
        wind_speed=speed, wind_direction=wind_dir, ambient_temp=temp)
    contam_model.set_initial_settings(
        weather_params, window_height=window_height)
    # run contam simulation
    contam_model.run_simulation(verbose=False)
    contam_model.generate_all_ventilation_matrices_for_all_door_open_close_combination()


