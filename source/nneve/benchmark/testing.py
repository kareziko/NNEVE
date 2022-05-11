from typing import List, cast

import pytest
import tensorflow as tf


def disable_gpu_or_skip() -> None:  # pragma: no cover
    try:
        tf.config.set_visible_devices([], "GPU")
        assert tf.config.get_visible_devices("GPU") == []
    except Exception:
        pytest.skip("Failed to disable GPU.")


def skip_if_no_gpu() -> None:  # pragma: no cover
    if not is_gpu():
        pytest.skip("No GPU available for testing.")


def is_gpu() -> bool:
    return len(cast(List, tf.config.get_visible_devices("GPU"))) > 0
