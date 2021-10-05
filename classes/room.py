class Room():
    def __init__(self, matrix_id, zone_info):
        self.room_id = zone_info['Z#']
        self.matrix_id = matrix_id
        self.group_id = []
        self.room_name = zone_info['name']
        self.temperature = zone_info['T0']
        self.door_open = []
        self.window_open = []
        self.infectivity_rate = 0

    def __repr__(self):
        return f"""Room id: {self.room_id}
matrix_id: {self.matrix_id}
name: {self.room_name}
door open? {'true' if self.door_open else 'false'}
window open? {'true' if self.window_open else 'false'}
infectivity_rate: {self.infectivity_rate}
        """

    @property
    def room_type(self):
        if 'lassroom' in self.room_name:
            return 'classroom'
        elif 'orridor' in self.room_name:
            return 'corridor'
        else:
            return 'unknown'

    def update_group_ids(self, students):
        for student in students:
            if student.current_room == self.room_id:
                self.group_id.append(student.group_id)

        
    @property
    def current_group_id(self):
        return self.group_id[-1]

    def update_infectivity_rate(self, infectivity_rates):
        self.infectivity_rate = infectivity_rates[self.matrix_id]


    def open_close_opening(self, bool, opening):
        """Assign latest status of opening (door or window) / (open or closed)"""
        getattr(self, f'{opening}_open').append(bool)


