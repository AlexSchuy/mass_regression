import numpy as np


class StepLogUniform:
    def __init__(self, start, num, step, base=10):
        self.values = np.logspace(start=start, stop=start + (num - 1)*step, num=num, base=base)

    def rvs(self):
        return int(np.random.choice(self.values))
