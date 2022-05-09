from pathlib import Path
from typing import Callable, Optional

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from nneve.benchmark import get_image_identity_fraction, plot_with_stats

np.random.seed(0)


DIR = Path(__file__).parent
DATA_DIR = DIR / "data"


def test_plot_with_stats(
    fig_to_image: Callable[[Optional[Figure], Optional[str]], Image.Image]
):
    plot_with_stats(np.random.uniform(1, 2, 100), np.random.uniform(3, 4, 100))
    assert (
        get_image_identity_fraction(
            Image.open(DATA_DIR / "example_1.png"), fig_to_image(None, None)
        )
        == 1.0
    )
