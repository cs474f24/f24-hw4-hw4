import numpy as np

from src.utils import get_neighbors


def default_J(n_colors=2):
    """
    Creates a predetermined J matrix that can be used in `ising_potential` or
        `potts_potential`.

    Note: when `n_colors == 2`, you can hardcode this matrix based on the
    definition of the Ising model.

    Note: When `n_colors > 2` passing this matrix as the J argument in
    `potts_potential` should make it such that adjacent pixels are most likely
    to have the same color, and then more likely to have similar colors, and
    least likely to have dramatic shifts. Said differently, this function
    shouldn't return the identity matrix.

    Returns:
        J: a 2D array with shape `(n_colors, n_colors)` where J[i, j] is the
        potential of a pixel taking value i when its neighbor has value j.
    """

    raise NotImplementedError


def default_mu(n_colors=2):
    """
    Creates a predetermined mu matrix that can be used in `ising_potential` or
        `potts_potential`.

    Note: when `n_colors == 2`, you can hardcode this matrix based on the
    definition of the Ising model.

    Note: when `n_colors > 2`, because we're adding noise uniformly at random
    in `add_noise`, this matrix should treat all mismatches as equally likely.

    Returns:
        mu: a 2D array with shape `(n_colors, n_colors)` where mu[i, j] is the
        potential of a pixel taking value i when the observed noisy pixel takes
        value j.
    """

    raise NotImplementedError


def calculate_J(arr, n_colors=2, smoothing=1):
    """
    Using an image arr, calculate a J matrix that can be used in
        `ising_potential` or `potts_potential`.

    Note: the easiest way to do this is likely just to count the number of
    adjacencies that we see between neighboring pixels. Think about how these
    values are used in `ising_potential` or `potts_potential`; how does this
    relate to a distribution over likely pixel configurations?

    Args:
        arr: a 2D image array
        n_colors: the number of colors
        smoothing: optional parameter depending on how you calculate J

    Returns:
        J: a 2D array with shape `(n_colors, n_colors)` where J[i, j] is the
        potential of a pixel taking value i when its neighbor has value j.
    """
    assert len(arr.shape) == 2
    assert np.all((arr >= 0) | (arr < n_colors))

    raise NotImplementedError


def calculate_mu(noisy_arr, orig_arr, n_colors=2, smoothing=1):
    """
    Using an orig_arr as a reference and a noisy_arr as our observed image,
        calculate a mu matrix that can be used in `ising_potential` or
        `potts_potential`.

    Args:
        noisy_arr: a 2D image array
        orig_arr: a 2D image array
        n_colors: the number of colors
        smoothing: optional parameter depending on how you calculate mu

    Returns:
        mu: a 2D array with shape `(n_colors, n_colors)` where mu[i, j] is the
            potential of a true pixel having value i when its noisy observed
            pixel has value j.
    """
    assert noisy_arr.shape == orig_arr.shape
    for arr in [noisy_arr, orig_arr]:
        assert len(arr.shape) == 2
        assert np.all((arr >= 0) | (arr < n_colors))

    raise NotImplementedError


def ising_potential(arr, i, j, orig, beta=1, J=None, mu=None):
    """
    Compute the conditional probability distribution
        over two binary states for `arr[i, j]`.

    To start, read:
    https://en.wikipedia.org/wiki/Ising_model

    However, you will need to make some design decisions about this potential.
    On Wikipedia, it uses J_{i,j}(s_i, s_j) to refer to the energetic cost of
    a neighboring state {s_i, s_j} and h_i(s_i) to refer to the energetic cost
    of pixel i taking value s_i when the observed noisy pixel is `orig`.

    You can assume that J and mu are 2x2 matrices. In the Wikipedia definition,
    mu is a scalar, but we are treating it as a matrix to try to make these more
    consistent across implementations.

    Args:
        arr: 2D pixel array, assumed to be binary
        i, j: coordinates of the pixel to consider flipping
        orig: the observed noisy pixel (e.g., `orig_img[i, j]`)
        beta, J, mu: hyperparameters for the Ising model

    Returns:
        a normalized conditional distribution over two hidden
        states. If this returns array([0.1, 0.9]), then there
        will be a 90% probability that the pixel takes value 1.
    """

    # Make sure we have the right dtype and number of colors
    assert np.issubdtype(arr.dtype, np.integer)
    neighbors = get_neighbors(arr, i, j)
    assert np.all((neighbors == 0) | (neighbors == 1))
    assert orig in [0, 1]

    if J is None:
        J = np.zeros((2, 2))
    else:
        assert isinstance(J, np.ndarray)
        assert J.shape == (2, 2)
    if mu is None:
        mu = np.zeros((2, 2))
    else:
        assert isinstance(mu, np.ndarray)
        assert mu.shape == (2, 2)

    raise NotImplementedError


def potts_potential(arr, i, j, orig, beta=1, J=None, mu=None, n_colors=3):
    """
    Compute the conditional probability distribution
        over `n_colors` states for `arr[i, j]`.

    To start, read:
    https://en.wikipedia.org/wiki/Potts_model#Generalized_Potts_model

    However, you will need to make some design decisions about this potential.
    On Wikipedia, it uses J_{i,j}(s_i, s_j) to refer to the energetic cost of
    a neighboring state {s_i, s_j} and h_i(s_i) to refer to the energetic cost
    of `arr[i, j]` taking value s_i when the observed noisy pixel is `orig`.


    Args:
        arr: 2D pixel array
        i, j: coordinates of the pixel to consider flipping
        orig: the observed noisy pixel (e.g., `orig_img[i, j]`)
        beta, J, mu: hyperparameters for the Potts model

    Returns:
        a normalized conditional distribution over `n_colors` hidden
        states. If this returns array([0.1, 0.5, 0.3, 0.1]), then there
        will be a 50% probability that the pixel takes value 1.

    """
    # Make sure we have the right dtype and number of colors
    assert np.issubdtype(arr.dtype, np.integer)
    neighbors = get_neighbors(arr, i, j)
    assert np.all((neighbors >= 0) | (neighbors < n_colors))

    if J is None:
        J = np.zeros((n_colors, n_colors))
    else:
        assert isinstance(J, np.ndarray)
        assert J.shape == (n_colors, n_colors)
    if mu is None:
        mu = np.zeros((n_colors, n_colors))
    else:
        assert isinstance(mu, np.ndarray)
        assert mu.shape == (n_colors, n_colors)

    raise NotImplementedError
