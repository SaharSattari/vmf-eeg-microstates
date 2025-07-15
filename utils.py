import os
import torch


realmin = 1e-12


def norm(input, p=2, dim=0, eps=1e-12):
    return input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_params(mixture_model, X):
    samples = torch.tensor(X, dtype=torch.float64)
    params = mixture_model.get_params()

    mus = params[1].detach().numpy()
    kappas = params[2].detach().numpy()
    logliks, logpcs = mixture_model(samples)
    logalpha = params[0].detach()
    jill = logalpha.unsqueeze(0) + logpcs
    posterior_probs = jill.log_softmax(dim=1).exp()

    probabilities = posterior_probs

    return probabilities, kappas, mus
