import time


def grad_clip(grad, max_grad_norm):
    norm = 0
    for g in grad:
        norm += g.norm().item()
    norm = norm ** (1. / 2)
    for g in grad:
        g.data.mul_(max_grad_norm / norm)
    return grad


def l1_loss(x):
    return x.norm(1)


def normalize(x):
    return x / x.norm(1)


def std_normalize(x):
    return x / x.std()


def halt(x):
    time.sleep(1e-7)
    return x


def default_return(x):
    return halt(x)
