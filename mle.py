import torch
import model
import utils
import numpy as np


def mle_vmf(X, num_of_clusters):
    opts = {}
    opts["max_iters"] = 10000  # maximum number of EM iterations
    opts["rll_tol"] = 1e-5  # tolerance of relative loglik improvement

    # randomly initialized mixture
    mix = model.MixvMF(x_dim=X.shape[1], n_components=num_of_clusters)
    samples = torch.tensor(X, dtype=torch.float64)

    # EM learning
    ll_old = -np.inf

    for steps in range(opts["max_iters"]):

        # E-step
        logalpha, mus, kappas = mix.get_params()
        logliks, logpcs = mix(samples)
        ll = logliks.sum()
        jll = logalpha.unsqueeze(0) + logpcs
        qz = jll.log_softmax(1).exp()

        if steps == 0:
            prn_str = "[Before EM starts] loglik = %.4f\n" % ll.item()
        else:
            prn_str = "[Steps %03d] loglik (before M-step) = %.4f\n" % (
                steps,
                ll.item(),
            )
        print(prn_str)

        # tolerance check
        if steps > 0:
            rll = (ll - ll_old).abs() / (ll_old.abs() + utils.realmin)
            if rll < opts["rll_tol"]:
                prn_str = "Stop EM since the relative improvement "
                prn_str += "(%.6f) < tolerance (%.6f)\n" % (rll.item(), opts["rll_tol"])
                print(prn_str)
                break

        ll_old = ll

        # M-step
        qzx = (qz.unsqueeze(2) * samples.unsqueeze(1)).sum(0)
        qzx_norms = utils.norm(qzx, dim=1)
        mus_new = qzx / qzx_norms
        Rs = qzx_norms[:, 0] / (qz.sum(0) + utils.realmin)
        kappas_new = (mix.x_dim * Rs - Rs**3) / (1 - Rs**2)
        alpha_new = qz.sum(0) / samples.shape[0]

        # assign new params
        mix.set_params(alpha_new, mus_new, kappas_new)

        logliks, logpcs = mix(samples)
        ll = logliks.sum()
        prn_str = "[Training done] loglik = %.4f\n" % ll.item()
        print(prn_str)

    # save model
    mix_em = mix

    return mix_em
