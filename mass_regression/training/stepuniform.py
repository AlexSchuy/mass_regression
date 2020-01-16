import numpy as np


class StepUniform:
    def __init__(self, start, num, step):
        self.values = np.linspace(start, start + (num - 1)*step, num)

    def rvs(self):
        return np.random.choice(self.values)
