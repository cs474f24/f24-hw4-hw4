import numpy as np
from PIL import Image
from src.data import quantize, get_color_idxs


def test_gibbs_sampling():
    from src.potentials import default_J, default_mu
    from src.gibbs import run_gibbs

    tiny_image = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]], dtype=np.uint8)
    tiny_image = Image.fromarray(tiny_image)
    tiny_image = quantize(tiny_image, 2)
    tiny_image_arr = get_color_idxs(np.array(tiny_image), n_colors=2)

    n_colors = 2
    J = default_J(n_colors=n_colors)
    mu = default_mu(n_colors=n_colors)
    samples = run_gibbs(tiny_image_arr, 50, n_colors=n_colors, J=J, mu=mu)

    # Over many samples, less than 1/9 of all pixels should be nonzero
    all_samples = np.concatenate(samples, axis=1).reshape(-1)
    assert np.mean(all_samples) < 1/9, "{np.mean(all_samples):.3f}"


def test_gibbs_expected_image():
    from src.potentials import default_J, default_mu
    from src.gibbs import run_gibbs, get_expected_image

    tiny_image = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]], dtype=np.uint8)
    tiny_image = Image.fromarray(tiny_image)
    tiny_image = quantize(tiny_image, 2)
    tiny_image_arr = get_color_idxs(np.array(tiny_image), n_colors=2)

    n_colors = 2
    J = default_J(n_colors=n_colors)
    mu = default_mu(n_colors=n_colors)
    np.random.seed(1)
    samples = run_gibbs(tiny_image_arr, 50, n_colors=n_colors, J=J, mu=mu)

    image = get_expected_image(samples, burnin=0, sample_every=1)
    assert np.all(image == 0), np.mean(image)

    # J matrix and no mu should push all pixels to 0s
    J = 1 * np.ones([n_colors, n_colors])
    J[0, 0] = 5
    J[1, 1] = 0

    for i in range(5):
        np.random.seed(i)
        random_image = np.random.randint(0, n_colors, size=(10, 10))
        samples = run_gibbs(random_image, 20, n_colors=n_colors, J=J, mu=None)
        image = get_expected_image(samples, burnin=15, sample_every=1)
        assert np.mean(image) < 0.05, (i, np.mean(image))


def test_gibbs_denoises():
    from src.data import load_mnist, quantize, get_color_idxs, add_noise
    from src.gibbs import run_gibbs, get_expected_image
    from src.potentials import default_J, default_mu
    from src.utils import mean_squared_error

    n_colors = 4
    noise = 0.2

    image = load_mnist(0)
    image = quantize(image, n_colors)
    image_arr = get_color_idxs(np.array(image), n_colors)

    noisy_image = add_noise(image, noise, n_colors=n_colors)
    noisy_image_arr = get_color_idxs(np.array(noisy_image), n_colors=n_colors)

    J = default_J(n_colors=n_colors)
    mu = default_mu(n_colors=n_colors)

    np.random.seed(1)
    samples = run_gibbs(noisy_image_arr, 10, n_colors=n_colors, J=J, mu=mu)
    exp_image = get_expected_image(samples, burnin=5, sample_every=1)

    mse1 = mean_squared_error(image_arr, noisy_image_arr)
    mse2 = mean_squared_error(image_arr, exp_image)

    assert mse1 > 2 * mse2, f"{mse2:.3f} should be half of {mse1:.3f}"
