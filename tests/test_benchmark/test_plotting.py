from pathlib import Path
from typing import Callable, Optional

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from nneve.benchmark import get_image_identity_fraction, plot_multi_sample

DIR = Path(__file__).parent
DATA_DIR = DIR / "data"


def test_plot_with_stats_two_samples_auto_range(
    fig_to_image: Callable[[Optional[Figure], Optional[str]], Image.Image]
):
    fig, _ = plot_multi_sample(
        np.loadtxt(DATA_DIR / "example_1_matrix_1.gz"),
        np.loadtxt(DATA_DIR / "example_1_matrix_2.gz"),
    )
    assert (
        get_image_identity_fraction(
            Image.open(DATA_DIR / "example_1.png"), fig_to_image(None, None)
        )
        == 1.0
    )
    generate_sample(fig, 1)


def generate_sample(fig, i):  # pragma: no cover
    fig.savefig(DATA_DIR / f"example_{i}.png")


def test_plot_with_stats_two_samples_manual_range(
    fig_to_image: Callable[[Optional[Figure], Optional[str]], Image.Image]
):
    fig, _ = plot_multi_sample(
        np.loadtxt(DATA_DIR / "example_1_matrix_1.gz"),
        np.loadtxt(DATA_DIR / "example_1_matrix_2.gz"),
        x_range=[0 for _ in range(30)],
        x_axis_label="Sample size",
    )
    generate_sample(fig, 2)
    assert (
        get_image_identity_fraction(
            Image.open(DATA_DIR / "example_2.png"), fig_to_image(None, None)
        )
        == 1.0
    )
    generate_sample(fig, 2)
