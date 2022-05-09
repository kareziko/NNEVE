from pathlib import Path
from timeit import timeit
from typing import Any, Callable

import pytest
import tensorflow as tf

from nneve.benchmark import plot_with_stats
from nneve.benchmark.plotting import datetag

DIR = Path(__file__).parent
DATA_DIR = DIR / "data"


def disable_gpu_or_skip():
    try:
        tf.config.set_visible_devices([], "GPU")
        assert tf.config.get_visible_devices("GPU") == []
    except Exception:
        pytest.skip("Failed to disable GPU.")


def skip_if_no_gpu():
    if tf.config.get_visible_devices("GPU") == []:
        pytest.skip("No GPU available for testing.")


class TestTensorTensorVsTensorConstant:
    def benchmark_samples(
        self,
        function: Callable[..., Any],
        sampling_repeats: int,
        samples_in_single: int,
    ):
        # let it train the graph before to avoid unstable results
        timeit(function, number=samples_in_single)

        results = []
        for _ in range(sampling_repeats):
            results.append(
                timeit(
                    function,
                    number=samples_in_single,
                )
                / samples_in_single
            )

        return results

    def run_benchmark_constant_ones(self):
        @tf.function
        def __function(__x: tf.Tensor, __c: tf.Tensor):
            return tf.multiply(tf.add(__x, __c), __c)

        sampling_repeats = 100
        samples_in_single = 100

        x = tf.random.normal((1024, 1024))
        c = 1.0

        tensor_x_constant = self.benchmark_samples(
            lambda: __function(x, c),
            sampling_repeats,
            samples_in_single,
        )

        x = tf.random.normal((1024, 1024))
        c = tf.ones((1024, 1024))

        tensor_x_tensor = self.benchmark_samples(
            lambda: __function(x, c),
            sampling_repeats,
            samples_in_single,
        )

        plt = plot_with_stats(
            tensor_x_constant,
            tensor_x_tensor,
            labels=(
                "tensor x constant",
                "tensor x tensor",
            ),
        )
        return plt

    @pytest.mark.benchmark()
    def test_constant_ones_cpu(self):
        disable_gpu_or_skip()
        plt = self.run_benchmark_constant_ones()
        plt.title("test_constant_ones_cpu")
        plt.savefig(DATA_DIR / f"test_constant_ones_cpu_{datetag()}.png")

    @pytest.mark.benchmark()
    def test_constant_ones_gpu(self):
        skip_if_no_gpu()
        plt = self.run_benchmark_constant_ones()
        plt.title("test_constant_ones_cpu")
        plt.savefig(DATA_DIR / f"test_constant_ones_gpu_{datetag()}.png")
