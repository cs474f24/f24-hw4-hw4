import numpy as np


# Simple 3x3 "images" for tests
bin_arr = np.array([
    [1, 0, 1],
    [0, 0, 1],
    [1, 1, 1]])

nonbin_arr = np.array([
    [0, 1, 0],
    [2, 0, 3],
    [0, 3, 0]])


def test_default_2x2():
    from src.potentials import default_J, default_mu

    J = default_J(n_colors=2)
    mu = default_mu(n_colors=2)

    assert J.shape == (2, 2)
    assert J[0, 0] > J[0, 1]
    assert J[0, 0] > J[1, 0]
    assert J[1, 1] > J[0, 1]
    assert J[1, 1] > J[1, 0]

    assert mu.shape == (2, 2)
    assert mu[0, 0] > mu[0, 1]
    assert mu[0, 0] > mu[1, 0]
    assert mu[1, 1] > mu[0, 1]
    assert mu[1, 1] > mu[1, 0]


def test_calculate_mu():
    from src.potentials import calculate_mu

    X = np.ones([4, 4])
    X[0, 0] = 0
    Y = np.eye(4)

    mu = calculate_mu(Y, X, n_colors=2, smoothing=0)

    msg = "If true pixel is 0, noisy pixel is 1"
    assert mu[0, 1] == np.max(mu), msg
    msg = "If true pixel is 0, noisy pixel never 0"
    assert mu[0, 0] == np.min(mu), msg
    msg = "If true pixel is 1, neighbor most likely 0"
    assert mu[1, 0] > mu[1, 1], msg


def test_calculate_J():
    from src.potentials import calculate_J

    X = np.ones([4, 4], dtype=int)
    X[1, 1] = 0

    J = calculate_J(X, n_colors=2, smoothing=0)

    msg = "If current pixel is 0, neighbor must be 1"
    assert J[0, 1] == np.max(J), msg
    msg = "If current pixel is 0, neighbor is never 0"
    assert J[0, 0] == np.min(J), msg
    msg = "If current pixel is 1, neighbor most likely 1"
    assert J[1, 1] > J[1, 0], msg


def test_ising_potential1():
    from src.utils import get_neighbors
    from src.potentials import ising_potential

    # Starter code should pass these next three cases, but make sure you
    #   understand how `get_neighbors` works.
    # Center pixel has neighbors: 0, 0, 1, 1
    assert np.all(np.equal(
        get_neighbors(bin_arr, 1, 1), np.array([0, 0, 1, 1])))
    # Top-left pixel has neighbors: 0, 0
    assert np.all(np.equal(
        get_neighbors(bin_arr, 0, 0), np.array([0, 0])))
    # Center-left pixel has neighbors: 1, 0, 1
    assert np.all(np.equal(
        get_neighbors(bin_arr, 1, 0), np.array([1, 1, 0])))

    # Your implementation of `ising_potential` should pass this
    #   test in the simple case where J=mu=None.
    lst = []
    for beta in [0.1, 1, 10]:
        for pixel in [(0, 0), (1, 0), (1, 1)]:
            for orig in [0, 1]:
                pot = ising_potential(
                    bin_arr, *pixel, orig=orig, J=None, mu=None, beta=beta)
                lst.append(pot)
    msg = "With J=mu=None, all states should have equal potential."
    assert np.all(np.equal(lst, 0.5)), msg


def test_ising_potential2():
    from src.potentials import ising_potential

    zeros = np.zeros([2, 2])
    ones = np.ones([2, 2])
    epi = np.array([[np.pi, np.e], [np.e, np.pi]])

    # Look at the center pixel bin_arr[1, 1]
    for orig in [0, 1]:
        # J is constant; ignore neighbors
        test = ising_potential(bin_arr, 1, 1, orig, J=zeros, mu=epi)
        assert test[orig] > test[1 - orig]
        test = ising_potential(bin_arr, 1, 1, orig, J=ones, mu=epi)
        assert test[orig] > test[1 - orig]

        # mu is constant; neighbors are tied
        test = ising_potential(bin_arr, 1, 1, orig, J=epi, mu=ones)
        assert np.isclose(test[orig], test[1 - orig])
        test = ising_potential(bin_arr, 1, 1, orig, J=epi, mu=zeros)
        assert np.isclose(test[orig],test[1 - orig])

        # neither constant; orig breaks tie
        test = ising_potential(bin_arr, 1, 1, orig, J=epi, mu=epi)
        assert test[orig] > test[1 - orig]
        test = ising_potential(bin_arr, 1, 1, orig, J=epi, mu=epi)
        assert test[orig] > test[1 - orig]

    # Look at the center-left pixel bin_arr[1, 0]
    # If mu = epi, and orig = 0, the potentials are tied
    test = ising_potential(bin_arr, 1, 0, 0, J=epi, mu=epi)
    assert np.all(np.isclose(test, 0.5))
    # If mu = epi, and orig = 1, then 1 becomes much more likely
    test = ising_potential(bin_arr, 1, 0, 1, J=epi, mu=epi)
    assert test[1] > test[0]


def test_ising_potential3():
    from src.potentials import ising_potential, default_J, default_mu

    J = default_J(n_colors=2)
    mu = default_mu(n_colors=2)

    # Test center pixel
    pot11_orig1 = ising_potential(bin_arr, 1, 1, 1, J=J, mu=mu)
    pot11_orig0 = ising_potential(bin_arr, 1, 1, 0, J=J, mu=mu)
    assert np.isclose(pot11_orig1[1], pot11_orig0[0])
    assert np.isclose(pot11_orig1[0], pot11_orig0[1])

    # Test center-left and center-right
    pot10_orig0 = ising_potential(bin_arr, 1, 0, 0, J=J, mu=mu)
    pot12_orig0 = ising_potential(bin_arr, 1, 2, 0, J=J, mu=mu)
    assert np.all(np.isclose(pot10_orig0, pot12_orig0))
    # Use same matrix for J and mu
    test = ising_potential(bin_arr, 1, 0, 0, J=J, mu=J)
    assert np.all(np.isclose(test, [0.5, 0.5]))
    test = ising_potential(bin_arr, 1, 0, 0, J=mu, mu=mu)
    assert np.all(np.isclose(test, [0.5, 0.5]))

    # Test corners
    pot00_orig1 = ising_potential(bin_arr, 0, 0, 1, J=J, mu=mu)
    pot22_orig0 = ising_potential(bin_arr, 2, 2, 0, J=J, mu=mu)
    assert np.isclose(pot00_orig1[1], pot22_orig0[0])
    assert np.isclose(pot00_orig1[1], pot22_orig0[0])
    assert np.isclose(pot00_orig1[0], pot22_orig0[1])

    # All zeros image; mu should not dominate J
    zeros = np.zeros([3, 3], dtype=int)
    test = ising_potential(zeros, 1, 1, 1, J=J, mu=mu)
    assert test[0] > test[1]
    test = ising_potential(zeros, 1, 0, 1, J=J, mu=mu)
    assert test[0] > test[1]
    test = ising_potential(zeros, 0, 0, 1, J=J, mu=mu)
    assert test[0] > test[1]


def test_potts_potential1():
    from src.potentials import potts_potential, default_J, default_mu

    n_colors = 4
    J = default_J(n_colors=n_colors)
    mu = default_mu(n_colors=n_colors)

    # Test the center pixel
    pot11_orig3 = potts_potential(nonbin_arr, 1, 1, 3, n_colors=8)
    assert pot11_orig3.shape == (8, )
    pot11_orig3 = potts_potential(nonbin_arr, 1, 1, 3, n_colors=n_colors, J=J, mu=mu)
    assert pot11_orig3.shape == (n_colors, )
    assert np.all(np.argsort(pot11_orig3) == np.arange(4))
    pot11_orig2 = potts_potential(nonbin_arr, 1, 1, 2, n_colors=n_colors, J=J, mu=mu)
    assert np.all(np.argsort(pot11_orig2)[:2] == np.arange(2))

    # Test side
    pot10_orig0_J = potts_potential(nonbin_arr, 1, 0, 0, n_colors=n_colors, J=J, mu=mu)
    assert np.all(np.argsort(pot10_orig0_J) == (3, 2, 1, 0))

    # Test corner
    pot20_orig3_J = potts_potential(nonbin_arr, 2, 0, 3, n_colors=n_colors, J=J, mu=mu)
    pot20_orig2_J = potts_potential(nonbin_arr, 2, 0, 2, n_colors=n_colors, J=J, mu=mu)
    assert pot20_orig3_J[0] < pot20_orig3_J[1] and pot20_orig3_J[1] < pot20_orig3_J[2]
    assert pot20_orig2_J[0] < pot20_orig2_J[1] and pot20_orig2_J[1] < pot20_orig2_J[2]
    # Most of potential should be on 2 or 3
    assert pot20_orig3_J[2:].sum() > 0.8
    assert pot20_orig2_J[2:].sum() > 0.8
