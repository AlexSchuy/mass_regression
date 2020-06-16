
class BaseModelFactory():
    def __init__(self, dataset, seed=0, name='base_model'):
        self.name = name
        self.dataset = dataset
        self.seed = seed

    @property
    def concatenated_input(self):
        return NotImplementedError

    def build(self, hparams):
        raise NotImplementedError