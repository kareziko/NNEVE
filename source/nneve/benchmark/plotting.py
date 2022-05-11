from datetime import datetime
from typing import Optional, Sized, Tuple, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from PIL import Image


def datetag(
    date: Optional[datetime] = None, fmt: str = "%d_%m_%Y_%H_%M_%S_%f"
):
    if date is None:
        date = datetime.now()
    return date.strftime(fmt)


def plot_with_stats(  # noqa: CFQ002
    *samples: ArrayLike,
    labels: Optional[Tuple[str, ...]] = None,
    colors: Optional[Tuple[str, ...]] = None,
    linewidth: int = 1,
    linestyle: str = "-",
    figsize: Tuple[int, int] = (14, 7),
    fig: Optional[Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    if fig is None:
        fig = plt.figure(figsize=figsize)

    if ax is None:
        ax = plt.axes()

    if labels is None:
        labels = tuple(f"Sample {i}" for i in range(len(samples)))

    if colors is None:
        colors = (
            "#a83240",
            "#5353f5",
            "#0bb836",
            "#c0d61c",
            "#8339bf",
            "#f56342",
            "#4098e6",
            "#e640a9",
            "#70e630",
        )
    # find max size to ensure lines are long from left to right
    n = max(len(cast(Sized, sample)) for sample in samples)

    for sample, color, label in zip(samples, colors, labels):
        ax.plot(
            sample,
            linewidth=linewidth,
            label=label,
            linestyle=linestyle,
            color=color,
        )
        ax.axhline(np.max(sample), 0, n, color=color[:7] + "40")
        ax.axhline(np.mean(sample), 0, n, color=color[:7] + "c0")
        ax.axhline(np.median(sample), 0, n, color=color[:7] + "80")
        ax.axhline(np.min(sample), 0, n, color=color[:7] + "40")
        ax.grid(True)
    ax.legend(loc="upper right")


def get_array_identity_fraction(first: NDArray, second: NDArray) -> float:
    """Calculate identity fraction of array. Identity fraction represents how
    many of values in arrays are equal.

    Parameters
    ----------
    first : NDArray
    second : NDArray

    Returns
    -------
    float
        fraction in range 0.0-1.0
    """
    first = first.reshape(-1)
    second = second.reshape(-1)
    return np.count_nonzero(first == second) / len(first)


def get_image_identity_fraction(
    first: Image.Image, second: Image.Image
) -> float:
    """Calculate identity fraction of images. Identity fraction represents how
    many of pixels in images are equal.

    Parameters
    ----------
    first : Image.Image
    second : Image.Image

    Returns
    -------
    float
        fraction in range 0.0-1.0
    """
    return get_array_identity_fraction(np.asarray(first), np.asarray(second))
