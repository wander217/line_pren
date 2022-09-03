class Averager:
    def __init__(self):
        self.value: float = 0
        self.num: int = 0

    def update(self, value: float, num: int):
        self.value += value
        self.num += num

    def calc(self):
        if self.num == 0:
            return 0
        return self.value / self.num

    def clear(self):
        self.value = 0
        self.num = 0
