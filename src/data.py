# Dataset loading functions. Don't edit this file.

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def check_data():
    if not os.path.isdir("data"):
        raise IOError("Make sure you run code from the root directory of your repo.")

    msg = "Follow the Data setup instructions in data/README.md: missing '{}'"
    for path in ["cats", "mnist"]:
        if not os.path.isdir(os.path.join("data", path)):
            raise IOError(msg.format(path))
        for i in range(16):
            if not os.path.isfile(os.path.join("data", path, f"{i}.png")):
                raise IOError(msg.format("{path}/{i}.png"))


def load_cats(i=None):
    """
    Load the cat data as Image objects from png files
    """
    check_data()
    if i is not None:
        return Image.open(f"data/cats/{i}.png")
    cats = []
    for i in range(16):
        cats.append(Image.open(f"data/cats/{i}.png"))
    return cats


def load_mnist(i=None):
    """
    Load the MNIST data as Image objects from png files
    """
    check_data()
    if i is not None:
        return Image.open(f"data/mnist/{i}.png")
    mnist = []
    for i in range(16):
        mnist.append(Image.open(f"data/mnist/{i}.png"))
    return mnist


def get_color_idxs(arr, n_colors=2):
    """
    Take a greyscale image with 256 colors and map it down to `n_colors` ids
    """
    k = 256 // (n_colors - 1)
    idxs = np.round(arr / k).astype(int)
    return idxs


def color_idxs_to_image(color_idxs, n_colors=2):
    """
    Given an image with only `n_colors` colors, map those color ids
    back into a 256 greyscale format.
    """
    k = 256 // (n_colors - 1)
    arr = np.clip(k * color_idxs, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def quantize(image, num_colors=5):
    """
    Convert a pillow image into greyscale with only `num_colors` colors
    """
    arr = np.array(image.convert("L"))
    color_idxs = get_color_idxs(arr, num_colors)
    return color_idxs_to_image(color_idxs, num_colors)


def display_image(image, n_colors, size=(128, 128), title=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    new_image = image.resize(size, resample=Image.Resampling.NEAREST)
    new_image = quantize(new_image, n_colors)
    ax.imshow(new_image, cmap='gray')
    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    plt.show()
    plt.close(fig)


def add_noise(image, prob=0.1, n_colors=2):
    """
    Change each pixel in the image independently with probability `prob`.
    You shouldn't need to change this function.
    """
    arr = np.array(image.convert("L"))
    colors = np.unique(arr)
    n_colors = colors.shape[0]

    # each color, including original, has a `1/num_colors` chance
    random_sample = np.random.randint(0, n_colors, size=arr.shape)

    # so we set `swap_prob` > `prob`
    swap_prob = min(1, (n_colors / (n_colors - 1)) * prob)
    swap = np.random.binomial(n=1, p=swap_prob, size=arr.shape)

    arr = (1 - swap) * arr + swap * colors[random_sample]
    return Image.fromarray(arr.astype(np.uint8), mode="L")


if __name__ == "__main__":
    cats = load_cats()
    for n_colors in [2, 4, 8]:
        x = quantize(cats[0], n_colors)
        assert n_colors == np.unique(np.array(x.getdata())).shape[0]
        display_image(x, n_colors, (256, 256), title=f"quantize(cats[0], {n_colors})")
