class CustomLoss():
    def __init__(self, x, x_pad, dataset, loss_fn, name='CustomLoss'):
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.x = x
        self.x_pad = x_pad
        self.num_targets = dataset.num_targets()

    def __call__(self, y_true, y_pred):
        return self.loss_fn(self.x, self.x_pad, y_true, y_pred, self.dataset)