import os
from datetime import datetime

import pandas as pd

from weather_cls import Weather
from contam_model_cls import ContamModel


class StocasticModel():
    def __init__(self, weather, **kwargs):
        self.weather = weather
        self.duration = kwargs['duration'] if 'duration' in kwargs else float(input('Simulation length in days?')) * 24
        self.mask_efficiency = kwargs['mask_efficiency'] if 'mask_efficiency' in kwargs else float(input('What is the assumed mask efficiency? [%]')) / 100
        self.lambda_home = kwargs['lambda_home'] if 'lambda_home' in kwargs else float(input('What is the assumed home infectivity rate? [hr^-1]'))

    def run(self, numberOfSimulations):
        pass

    def storeSIR(self):
        pass

    def getVentilationDataFromCSV(self):
        pass

    def getVentilationDataFromContamTxt(self):
        folder = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_result_txt_files'
        fnames = os.listdir(folder)
        for file in fnames:
            df = pd.read_table(f'{folder}/{file}', encoding="latin1", delimiter = "\t", header=13)
            breakpoint()


    def plot(self):
        pass






if __name__ == '__main__':
    contam_exe_dir = '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/'
    prj_dir = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/'
    name = 'school_corridor'

    weather = Weather(windSpeed=15, ambientTemperature=10)
    contamModel = ContamModel(contam_exe_dir=contam_exe_dir,
                              contam_dir=prj_dir,
                              project_name=name)


    model = StocasticModel(weather=weather, duration=5)
    model.getVentilationDataFromContamTxt()
