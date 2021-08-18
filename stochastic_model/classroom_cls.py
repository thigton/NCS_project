from student_group_cls import StudentGroup
from simulation_cls import Simulation


class Classroom(Simulation):
    def __init__(self, id):
        self.classroom_id = id
        self.init_students()
        self.quanta_concentration = 0

    def init_students(self):
        self.students = StudentGroup()


if __name__ == '__main__':

    class1 = Classroom(id=1)
    breakpoint()