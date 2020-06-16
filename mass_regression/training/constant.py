

class Constant:
  def __init__(self, value):
    self.value = value

  def rvs(self, loc=None, scale=None, size=None, random_state=None):
    return self.value
