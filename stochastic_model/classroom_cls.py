from student_group_cls import StudentGroup

class Classroom():
    def __init__(self, id):
        self.id = id
        self.init_students()
        self.quanta_concentration = 0

    def init_students(self):
        self.students = StudentGroup()


if __name__ == '__main__':

    class1 = Classroom(id=1)
    breakpoint()