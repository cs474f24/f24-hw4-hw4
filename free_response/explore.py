import argparse
import numpy as np

from src.data import load_cats, load_mnist
from src.data import quantize, get_color_idxs, add_noise
from src.em import run_em
from src.potentials import default_J, default_mu
from src.utils import mean_squared_error, plot_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["cats", "mnist"], default="mnist")
    parser.add_argument("--img", type=int, default=0)
    parser.add_argument("--seed", type=float, default=1)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--n_colors", type=int, default=2)
    parser.add_argument("--n_em_iters", type=int, default=5)
    parser.add_argument("--n_gibbs_iters", type=int, default=5)
    parser.add_argument("--burnin", type=int, default=2)
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--smoothing", type=float, default=1)
    parser.add_argument("--noshow", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.data == "cats":
        image = load_cats(args.img)
    elif args.data == "mnist":
        image = load_mnist(args.img)

    image = quantize(image, args.n_colors)
    image_arr = get_color_idxs(np.array(image), args.n_colors)

    noisy_image = add_noise(image, args.noise, n_colors=args.n_colors)
    noisy_image_arr = get_color_idxs(np.array(noisy_image), n_colors=args.n_colors)

    em_args = {
        "n_colors": args.n_colors,
        "n_em_iters": args.n_em_iters,
        "n_gibbs_iters": args.n_gibbs_iters,
        "burnin": args.burnin,
        "sample_every": args.sample_every,
        "beta": args.beta,
        "J": default_J(args.n_colors),
        "mu": default_mu(args.n_colors),
        "smoothing": args.smoothing,
    }

    samples, extra_info = run_em(noisy_image_arr, **em_args)

    # If you only ran one EM iteration, plot out visualization of Gibbs
    # sampling algorithm
    if args.n_em_iters == 1 and "gibbs_samples" in extra_info:
        print("Plotting Gibbs samples")
        plot_samples(
            extra_info["gibbs_samples"], image_arr, noisy_image_arr,
            n_colors=args.n_colors, show=not args.noshow, save=not args.nosave)

    # Otherwise, plot a summary of denoising per EM iteration
    else:
        plot_samples(
            samples, image_arr, noisy_image_arr,
            n_colors=args.n_colors, show=not args.noshow, save=not args.nosave)


if __name__ == "__main__":
    main()
