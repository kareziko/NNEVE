from datetime import datetime
from itertools import chain
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


def plot_multi_sample(  # noqa: CFQ002 CCR001
    *samples: ArrayLike,
    x_range: Optional[ArrayLike] = None,
    labels: Optional[Tuple[str, ...]] = None,
    colors: Optional[Tuple[str, ...]] = None,
    x_axis_label: str = "Sample index",
    y_axis_label: str = "Execution time [s]",
    linewidth: int = 1,
    linestyle: str = "-",
    figsize: Tuple[int, int] = (14, 7),
    add_stats: bool = False,
    fig: Optional[Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[Figure, plt.Axes]:
    if fig is None:
        fig = plt.figure(figsize=figsize)

    if ax is None:
        ax = plt.axes()

    # find max size to ensure lines are long from left to right
    n = max(len(cast(Sized, s)) for s in chain(samples, [()]))

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
    x = np.arange(n)
    for sample, color, label in zip(samples, colors, labels):
        ax.plot(
            x,
            sample,
            linewidth=linewidth,
            label=label,
            linestyle=linestyle,
            color=color,
        )
        if add_stats:
            ax.axhline(np.max(sample), 0, n, color=color[:7] + "40")
            ax.axhline(np.mean(sample), 0, n, color=color[:7] + "c0")
            ax.axhline(np.median(sample), 0, n, color=color[:7] + "80")
            ax.axhline(np.min(sample), 0, n, color=color[:7] + "40")

    ax.grid(True)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)

    x_range_length = len(cast(Sized, x_range))

    if x_range is not None:
        if len(x) == x_range_length:
            ax.set_xticks(x, x_range)
        else:
            ax.set_xticks(np.linspace(0, n, x_range_length), x_range)

    ax.legend(loc="upper right")
    return fig, ax


def pretty_bytes(i: int) -> str:  # noqa: CFQ004
    if i < 1024:
        return f"{i}"
    if i < 1024**2:
        return f"{i//1024}Ki"
    if i < 1024**3:
        return f"{i//1024**2}Mi"
    return f"{i//1024**3}Gi"


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
