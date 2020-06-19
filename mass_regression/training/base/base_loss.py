class BaseLoss():
    def __init__(self, x, x_pad, dataset, name='BaseLoss'):
        self.name = name
        self.dataset = dataset
        self.x = x
        self.x_pad = x_pad
        self.num_targets = dataset.num_targets

    def __call__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

    def loss_fn(self, y_true, y_pred):
        raise NotImplementedError()