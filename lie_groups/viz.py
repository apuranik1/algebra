import numpy as np
import matplotlib.pyplot as plt
from skimage import color


def lch2rgb(L, C, h):
    return color.lab2rgb(color.lch2lab(np.stack([L, C, h], axis=-1)))


def complex_colors(z: np.ndarray):
    """Display a complex matrix as a generalizd heatmap.

    The argument is encoded cyclically via the hue, and the norm is
    encoded via lightness (and chroma, a little).

    Uses CIELCH for perceptual uniformity.
    """
    r = np.abs(z)
    arg = np.angle(z)

    h = arg + np.pi
    l = 100 * (1.0 - 1.0 / (1.0 + r**0.5))
    # c = np.full_like(r, 30.)
    c = (l / 100) ** 0.1 * 30.0

    colors = lch2rgb(l, c, h)
    return colors


def label_matrix(matrix, ax=None) -> plt.Axes:
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(
                j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w"
            )
    return ax


def complex_heatmap(matrix, ax=None):
    ax = label_matrix(matrix, ax=ax)
    im = ax.imshow(complex_colors(matrix))


def heatmap(matrix, ax=None):
    ax = label_matrix(matrix, ax=ax)
    im = ax.imshow(matrix)
