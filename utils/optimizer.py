from torch import nn
from torch.optim import SGD, Adam


def build_optimizer(args, model, lr, momentum, weight_decay):
    g = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g[2].append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g[0].append(v.weight)

    if args.adam:
        optimizer = Adam(params=g[2], lr=lr, betas=(momentum, 0.999))
    else:
        optimizer = SGD(params=g[2], lr=lr, momentum=momentum, nesterov=True)

    optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer