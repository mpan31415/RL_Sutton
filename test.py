
class Person:
    def __init__(self, name, age) -> None:
        self.name = name
        self.age = age
        
    def show_person_info(self):
        print("%s is %d years old" % (self.name, self.age))
        

class Student(Person):
    def __init__(self, name:str, age:int, grade:float) -> None:
        super().__init__(name, age)
        self.grade = grade
        
    def show_person_info(self):
        self.name = self.name.upper()
        return super().show_person_info()
        
    def show_student_info(self):
        print("%s is %d years old with a grade of %.1f" % (self.name, self.age, self.grade))
        

bob = Person("bob", 10)
bob.show_person_info()

michael = Student("michael", 21, 100.0)
michael.show_person_info()
michael.show_student_info()