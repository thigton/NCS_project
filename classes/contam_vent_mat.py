from datetime import datetime

class ContamVentMatrixStorage():

    def __init__(self, contam_model):
        self.contam_model_name = contam_model.project_name
        Ta, _, Ws, Wd = contam_model.environment_conditions.df.iloc[0,:4].values
        self.ambient_temp = Ta
        self.wind_speed = Ws
        self.wind_direction = Wd
        self.window_height = contam_model.get_window_height()
        self.corridor_temp = contam_model.get_zone_temp_of_room_type('corridor')
        self.classroom_temp = contam_model.get_zone_temp_of_room_type('classroom')
        self.windows_open = True
        self.date_init = datetime.now()
        self.matrices = contam_model.all_door_matrices
        self.external_door_matrix_idx = contam_model.external_door_matrix_idx