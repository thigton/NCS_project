class Students():

    def __init__(self, init_room, init_students_per_class):
        self.group_id = init_room.room_id
        self.room_id = [init_room.room_id]
        self.S = [init_students_per_class] if init_room.room_type == 'classroom' else [0]
        self.I = [0]
        self.R = [0]
        self.infectivity_rate = 0

    @property
    def current_room(self):
        return self.room_id[-1]

    def changeClassroom(self, new_classroom):
        self.room_id.append(new_classroom)


    def infection(self, first=False):
        if self.S[-1] == 0:
            print('Can not have an  infection. There is no one left')
            breakpoint()
            exit()
        if first:
            self.S[0] -= 1
            self.I[0] += 1
        else:
            self.S.append(self.latest_S-1)
            self.I.append(self.latest_I+1)
            self.R.append(self.latest_R)

    def recovery(self):
        if self.I[-1] == 0:
            print('Can not have a recovery. There is no one infected')
            exit()
        self.S.append(self.latest_S)
        self.I.append(self.latest_I-1)
        self.R.append(self.latest_R+1)

    def no_event(self):
        self.S.append(self.latest_S)
        self.I.append(self.latest_I)
        self.R.append(self.latest_R)

    @property
    def latest_S(self):
        return self.S[-1]

    @property
    def latest_I(self):
        return self.I[-1]

    @property
    def latest_R(self):
        return self.R[-1]

    @property
    def total_number(self):
        return self.latest_S + self.latest_I + self.latest_R


    def update_infectivity_rate(self, rooms, infectivity_rates):
        for room in rooms:
            if room.room_id == self.room_id:
                self.infectivity_rate = infectivity_rates[room.matrix_id]

if __name__ == '__main__':
    pass
    # x = Students(init_zone_info=pd.DataFrame([[0,1]], columns=['Z#' ,'P#']), init_students_per_class=30)

    # x.infection()
    # x.infection()
    # x.infection()
    # breakpoint()