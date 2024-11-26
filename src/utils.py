# You should not need to edit these functions.

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

from src.data import color_idxs_to_image


def get_neighbors(arr, i, j):
    """
    Given a 2D array of pixels `arr` and a centerpoint
        (i, j); find all the neighbors of that point.
        If that point is on an edge, there may be
        fewer than four neighbors, but always at least
        two.

    Args:
        arr: 2D pixel array
        i, j: coordinates of the pixel for which to find neighbors

    Returns:
        1-D array of values taken by neighbors of (i, j)
    """

    assert isinstance(arr, np.ndarray)
    assert len(arr.shape) == 2
    assert i < arr.shape[0] and j < arr.shape[1]

    neighbors = np.concatenate([
        arr[max(0, i - 1):i, j],
        arr[i, max(0, j - 1):j],
        arr[i + 1: min(i + 2, arr.shape[0]), j],
        arr[i, j + 1: min(j + 2, arr.shape[0])],
    ], axis=0)

    return neighbors


def mean_squared_error(original_image, reconstructed_image):
    return np.mean(np.square(original_image - reconstructed_image))


def plot_samples(samples, orig_image_arr, noisy_image_arr, n_colors=2, show=True, save=True):
    """
    Plot the progress of Gibbs sampling or the EM algorithm at denoising an image.

    Args:
        samples: a list of 2-D image arrays
        orig_image_arr: the true image that the samples are trying to reproduce
        n_colors: the number of colors
        show: should we show the resulting images on the screen?
        save: should we save the resulting images to file?
    """

    all_mses = [mean_squared_error(orig_image_arr, samples[idx])
                for idx in range(len(samples))]

    if len(samples) == 1:
        midpoint = None
        ncols = 3
    else:
        # Find the lowest MSE from any iteration
        midpoint = np.argmin(all_mses[:-1])
        ncols = 4

    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

    mse = mean_squared_error(orig_image_arr, noisy_image_arr)
    axes[0].imshow(color_idxs_to_image(noisy_image_arr, n_colors), cmap='gray')
    axes[0].set_title(f"Noisy image; MSE={mse:.2f}")
    axes[0].axis('off')

    axes[-1].imshow(color_idxs_to_image(orig_image_arr, n_colors), cmap='gray')
    axes[-1].set_title("Original image")
    axes[-1].axis('off')

    col = 1
    if midpoint is not None:
        image = color_idxs_to_image(samples[midpoint], n_colors)
        mse = all_mses[midpoint]
        title = f"t={midpoint + 1}; MSE={mse:.2f}"

        axes[col].imshow(image, cmap='gray')
        axes[col].set_title(title)
        axes[col].axis('off')

        col += 1

    # Plot the final sample
    image = color_idxs_to_image(samples[-1], n_colors)
    mse = all_mses[-1]
    title = f"t={len(samples)}; MSE={mse:.2f}"

    axes[col].imshow(image, cmap='gray')
    axes[col].set_title(title)
    axes[col].axis('off')

    fig.tight_layout(pad=1.2)
    if save:
        fn = datetime.now().strftime("%Y-%m-%d-%H-%M.png")
        fig.savefig(os.path.join("plots", fn), bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    if len(samples) > 1:
        rnge = np.arange(1, len(samples) + 1)
        fig2, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(rnge, all_mses)
        ax.set_title("MSE per iteration")
        ax.set_xticks([1 + idx for idx in range(len(samples))])
        fig2.tight_layout()
        if save:
            fn = datetime.now().strftime("%Y-%m-%d-%H-%M-mse.png")
            fig2.savefig(os.path.join("plots", fn), bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig2)
