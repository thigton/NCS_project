"""Simple run script highlighting some of the features of the ContamModel class
"""
from classes.contam_model import ContamModel


wind_speed = 15.0
wind_direction = 0.0
ambient_temp = 10.0

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
print(contam_model.flow_rate_df)

# change one of the classroom window heights
contam_model.set_flow_path(path=3, param='opening_height', value=1.0)
# close a classroom door (change the opening type to something that is "closed")
contam_model.set_flow_path(path=14, param='type', value=3)
# run contam simulation
contam_model.run_simulation()
print(contam_model.flow_rate_df)
