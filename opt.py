from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
def get_opt(args, model):
    optimizers = {'adam': Adam, 'sgd': SGD}
    optimizer = args.opt

    opt_params = {"lr": args.lr, 'weight_decay': args.decay}

    if optimizer=="sgd":
        opt_params['momentum'] = 0.9

    opt = optimizers[optimizer](filter(lambda p: p.requires_grad, model.parameters()), **opt_params)
    return opt

def get_criterion(args):
    return CrossEntropyLoss()
    