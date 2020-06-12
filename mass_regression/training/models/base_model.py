
class BaseModel():
    def __init__(self, dataset, seed=0):
        self.model = None
        self.dataset = dataset
        self.seed = seed

    def build(self, hparams):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)