import numpy as np
import time

from src.gibbs import run_gibbs, get_expected_image
from src.potentials import calculate_J, calculate_mu


def run_em(noisy_arr, n_colors=2, n_em_iters=10, n_gibbs_iters=10, burnin=5,
       sample_every=2, beta=1, smoothing=1, mu=None, J=None):
    """

    Args:
        noisy_arr: the original noisy array to include

    Returns:
        samples: a list of `n_em_iters` 2D arrays, where each array represents
                 the image after one iteration of the EM algorithm.
        extra_info: a dictionary you can use for debugging and experimentation.
                 This will be ignored by all test cases.
    """

    start = time.time()
    extra_info = {}
    em_samples = []
    arr = noisy_arr.copy()

    for i in range(n_em_iters):

        gibbs_samples = run_gibbs(arr, n_iter=n_gibbs_iters, n_colors=n_colors, J=J,
                                  mu=mu, beta=beta)

        raise NotImplementedError

    # NOTE: Keep this code here for `tests/test_em.py`
    extra_info["gibbs_samples"] = gibbs_samples
    extra_info["J"] = J
    extra_info["mu"] = mu
    runtime = (time.time() - start) / 60
    print(f"{n_em_iters} EM iters with {n_gibbs_iters} Gibbs iters took {runtime:.2f} min")
    return em_samples, extra_info
