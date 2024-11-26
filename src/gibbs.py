import itertools
import numpy as np
from scipy.stats import mode

from src.potentials import ising_potential, potts_potential


def run_gibbs(orig_arr, n_iter, n_colors=2, beta=1, J=None, mu=None):
    """
    Run the Gibbs Sampling algorithm on the image to get estimates of p(z | x)

    Args:
        orig_arr: the original image array
        n_iter: the number of iterations to run
            This is not the total number of conditional probabilities to
            calculate!  In each iteration, you should sample N new points,
            where N is the number of pixels in the entire image. You can either
            do this deterministically (e.g., with a for loop over each pixel)
            or randomly choose N pixels.
        n_colors: The number of colors. If n == 2, use `ising_potential`.
            Otherwise, using `potts_potential`.
        beta, J, mu: hyperparameters for the potential function. Pass these to the
            potential functions.

    Returns:
        samples: a list of arrays, where each array represents the pixels of an
            image after one iteration of the algorithm.
    """
    # Don't modify `orig_arr`!
    arr = orig_arr.copy()

    raise NotImplementedError


def get_expected_image(samples, burnin=5, sample_every=1):
    """
    Given a list of samples from p(Z | X), for each individual pixel Z_{i, j},
        compute the *marginally* most likely value. Construct that into a
        image of the corresponding size.

    Args:
        samples: a list of 2D arrays representing the outputs of `run_gibbs`
        burnin: how many samples to ignore from the start of the `samples` list
        sample_every: after burnin, go through `samples` and take one sample
                      every so often

    Hint: you can use numpy slicing with `samples[burnin::sample_every]` to get
          the samples you should include in this calculation.
    Hint: you can use scipy.stats.mode to compute the mode.

    Returns:
        arr: a 2D image arrary of the same shape as each image in `samples`
             that takes the mode (e.g., scipy.stats.mode) value of the
             included samples.

    """
    assert 0 <= burnin
    assert burnin < len(samples)
    assert (type(sample_every) == int) and (sample_every >= 1)
    assert np.all([samples[0].size == x.size for x in samples])

    raise NotImplementedError
