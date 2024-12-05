from enum import Enum

class Title(Enum):
    MR = "Mr."

class Test:
    def __init__(self, name: str, title: Title):
        self.name = name
        self.title = title

    def __repr__(self):
        return f"{self.title.value} {self.name}"
    

t1 = Test("Adhir", Title.MR)

print(t1)