import os
from datetime import datetime, timedelta
import pandas as pd
from contam_model_cls import ContamModel
from datetime import datetime

from simulation_cls import Simulation

class StocasticModel():
    def __init__(self, weather, contam_details, simulation_constants):
        self.weather = weather
        self.contam_details = contam_details
        self.consts = simulation_constants



        # init contam model
        self.contam_model = ContamModel(contam_exe_dir=self.contam_details['exe_dir'],
                contam_dir=self.contam_details['prj_dir'],
                project_name=self.contam_details['name'])
        # set weather params in model
        self.contam_model.set_environment_conditions(condition='wind_direction', value=self.weather.wind_direction,  units='km/hr')
        self.contam_model.set_environment_conditions(condition='wind_speed', value=self.weather.wind_speed,  units='km/hr')
        self.contam_model.set_environment_conditions(condition='ambient_temp', value=self.weather.ambient_temp,  units='C')
        # run contam simulation
        self.contam_model.run_simulation()


    def run(self):
        for run in range(self.consts['no_of_simulations']):
            if run % 10 == 0:
                print(f'{run/self.consts["no_of_simulations"]:0.2%} complete')
            sim = Simulation(sim_id=run,
                             simulation_constants = self.consts,
                             contam_model=self.contam_model,
                            )
            sim.run()
            sim.plot_SIR()
            breakpoint()

    def storeSIR(self):
        pass







if __name__ == '__main__':
    pass
