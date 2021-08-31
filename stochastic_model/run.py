from stocastic_model_cls import StocasticModel
from weather_cls import Weather
from datetime import time, timedelta

contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                        'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                        'name': 'school_corridor'}

simulation_constants = {'duration': timedelta(days=28),
                        'mask_efficiency': 0,
                        'lambda_home': 2000/898.2e4 / 24, # [hr^-1]
                        'pulmonary_vent_rate': 0.54, # [m3.hr^-1]
                        'quanta_gen_rate': 5, # [quanta.hr^-1]
                        'recover_rate': (8.2*24)**(-1), # [hr^-1]
                        'school_start': time(hour=9),
                        'school_end': time(hour=16),
                        'no_of_simulations': 1000,
                        'init_students_per_class': 30,
                        }

weather = Weather(wind_speed=0.0, wind_direction=0.0, ambient_temp=10.0)

model = StocasticModel(weather=weather,
                       contam_details=contam_model_details,
                       simulation_constants=simulation_constants,
                       )

model.run()