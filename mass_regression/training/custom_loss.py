class CustomLoss():
    def __init__(self, x, x_pad, dataset, loss_fn, name='CustomLoss'):
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.x = x
        self.num_targets = data.get_num_targets(dataset, target)
        self.scale_y_pad = scale_y_pad
        self.unscale_y = unscale_y
        self.unscale_x = unscale_x
        self.mix_weight = mix_weight
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mixed = mixed  # testing parameter

    def __call__(self, y_true, y_pred):
        return self.loss_fn(x, x_pad, y_true, y_pred, dataset)