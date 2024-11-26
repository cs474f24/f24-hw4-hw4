import numpy as np
from PIL import Image
from src.data import quantize, get_color_idxs
from src.utils import mean_squared_error


def test_em():
    from src.gibbs import run_gibbs, get_expected_image
    from src.potentials import default_J, default_mu
    from src.em import run_em

    n_colors = 2
    n_gibbs_iters = 10
    n_em_iters = 10
    burnin = 5
    init_J = default_J()
    init_mu = default_mu()

    mses, Js, mus = [], [], []
    for seed in range(5):
        np.random.seed(seed)
        tiny_image_arr = np.random.randint(0, n_colors, size=(10, 10))

        np.random.seed(seed)
        no_em = run_gibbs(tiny_image_arr, n_gibbs_iters, n_colors=n_colors,
                          J=init_J, mu=init_mu)
        no_em_image = get_expected_image(no_em, burnin=burnin, sample_every=1)

        np.random.seed(seed)
        with_em, extra_info = run_em(
            tiny_image_arr, n_colors=2, n_gibbs_iters=n_gibbs_iters,
            n_em_iters=n_em_iters, burnin=burnin)

        Js.append(np.mean(np.square(init_J - extra_info["J"]) / np.sum(np.square(init_J))))
        mus.append(np.mean(np.square(init_mu - extra_info["mu"]) / np.sum(np.square(init_mu))))

        with_em_image = with_em[-1]
        mse = mean_squared_error(no_em_image, with_em_image)
        mses.append(mse)

    assert np.mean(Js) > 0.4, "EM should learn new J matrix"
    assert np.mean(mus) > 1, "EM should learn new mu matrix"
    assert np.mean(mses) > 0.4, "EM should learn different images than Gibbs alone"
