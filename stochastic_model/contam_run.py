from weather_cls import Weather
from contam_model_cls import ContamModel


wind_speed = 15
wind_direction = 0
ambient_temp = 10

contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                        'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                        'name': 'school_corridor'}


contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
        contam_dir=contam_model_details['prj_dir'],
        project_name=contam_model_details['name'])
# set weather params in model
contam_model.set_environment_conditions(condition='wind_direction', value=wind_direction,  units='km/hr')
contam_model.set_environment_conditions(condition='wind_speed', value=wind_speed,  units='km/hr')
contam_model.set_environment_conditions(condition='ambient_temp', value=ambient_temp,  units='C')

# run contam simulation
contam_model.run_simulation()

# change one of the classroom window heights
contam_model.set_flow_path(path=3, param='opening_height', value=1.0)
# close a classroom door
contam_model.set_flow_path(path=14, param=)
# run contam simulation
contam_model.run_simulation()