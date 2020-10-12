from torch.optim import SGD, AdamW


def sgd_factory(lr: float, weight_decay: float, momentum: float, nesterov: bool):
    def fact(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov):
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    return fact


def adamw_factory(lr: float, weight_decay: float, amsgrad: bool):
    def fact(params, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad):
        return AdamW(params, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    return fact
