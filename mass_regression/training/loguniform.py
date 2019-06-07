from scipy.stats import uniform


class LogUniform:
  def __init__(self, loc, scale, base=10, discrete=False):
    self.loc = loc
    self.scale = scale
    self.base = base
    self.discrete = discrete

  def rvs(self, loc=None, scale=None, size=None, random_state=None):
    if not loc:
      loc = self.loc
    if not scale:
      scale = self.scale
    if size:
      sample = self.base**uniform.rvs(loc=self.loc, scale=self.scale, size=size, random_state=random_state)
    else:
      sample = self.base**uniform.rvs(loc=self.loc, scale=self.scale, random_state=random_state)
    if self.discrete:
      return int(round(sample))
    else:
      return sample
