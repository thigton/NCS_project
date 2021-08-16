import pandas as pd
from datetime import datetime
import os

class StocasticModel():
    def __init__(self, weather, **kwargs):
        self.weather = weather
        self.duration = kwargs['duration'] if 'duration' in kwargs else float(input('Simulation length in days?')) * 24
        self.mask_efficiency = kwargs['mask_efficiency'] if 'mask_efficiency' in kwargs else float(input('What is the assumed mask efficiency? [%]')) / 100

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


class Weather():
    def __init__(self, windSpeed, ambientTemperature):
        self.windSpeed = windSpeed
        self.ambientTemperature = ambientTemperature

    def get_current_weather(self):
        pass



if __name__ == '__main__':

    weather = Weather(windSpeed=15, ambientTemperature=10)
    model = StocasticModel(weather=weather, duration=5)
    model.getVentilationDataFromContamTxt()