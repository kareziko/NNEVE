from pathlib import Path
from timeit import timeit
from typing import Any, Callable, List, Tuple

import pytest
import tensorflow as tf
from matplotlib import pyplot as plt

from nneve.benchmark import (
    datetag,
    disable_gpu_or_skip,
    get_sys_info,
    plot_with_stats,
    skip_if_no_gpu,
)

DIR = Path(__file__).parent
DATA_DIR = DIR / "data"


SHAPE_SET_1 = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]


class TestTensorTensorVsTensorConstant:
    def run_benchmark_samples(
        self,
        function: Callable[..., Any],
        sampling_repeats: int,
        samples_in_single: int,
    ) -> List[float]:
        # let it train the graph before to avoid unstable results
        timeit(function, number=samples_in_single)

        results: List[float] = []
        for _ in range(sampling_repeats):
            results.append(
                timeit(
                    function,
                    number=samples_in_single,
                )
                / samples_in_single
            )

        return results

    def dump_plot(self, plot_title: str, shape: Tuple[int, ...]):
        shape_string = "x".join(str(i) for i in shape)
        plt.title(f"{plot_title} {shape_string}".capitalize())
        plot_title_no_spaces = plot_title.replace(" ", "_")
        dest_dir = DATA_DIR / plot_title_no_spaces
        dest_dir.mkdir(0o777, True, True)
        dtag = datetag()
        plt.savefig(
            dest_dir / f"{plot_title_no_spaces}_{shape_string}_{dtag}.png"
        )
        (
            dest_dir
            / f"sys_info_{plot_title_no_spaces}_{shape_string}_{dtag}.json"
        ).write_text(get_sys_info().json(), encoding="utf-8")

    def benchmark_constant_ones(self, tensor_shape: Tuple[int, int]):
        @tf.function
        def __function(__x: tf.Tensor, __c: tf.Tensor):
            return tf.multiply(tf.add(__x, __c), __c)

        sampling_repeats = 100
        samples_in_single = 100

        x = tf.random.normal(tensor_shape)
        c = 1.0

        tensor_x_constant = self.run_benchmark_samples(
            lambda: __function(x, c),
            sampling_repeats,
            samples_in_single,
        )

        x = tf.random.normal(tensor_shape)
        c = tf.ones(tensor_shape)

        tensor_x_tensor = self.run_benchmark_samples(
            lambda: __function(x, c),
            sampling_repeats,
            samples_in_single,
        )

        plot_with_stats(
            tensor_x_constant,
            tensor_x_tensor,
            labels=(
                "tensor x constant",
                "tensor x tensor",
            ),
        )

    @pytest.mark.benchmark()
    @pytest.mark.parametrize("tensor_shape", SHAPE_SET_1)
    def test_constant_ones_cpu(self, tensor_shape: Tuple[int, int]):
        disable_gpu_or_skip()
        self.benchmark_constant_ones(tensor_shape)
        self.dump_plot("constant ones cpu", tensor_shape)

    @pytest.mark.benchmark()
    @pytest.mark.parametrize("tensor_shape", SHAPE_SET_1)
    def test_constant_ones_gpu(self, tensor_shape: Tuple[int, int]):
        skip_if_no_gpu()
        self.benchmark_constant_ones(tensor_shape)
        self.dump_plot("constant ones gpu", tensor_shape)

    def benchmark_graph_non_graph(self, tensor_shape: Tuple[int, int]):
        @tf.function
        def __graph_function(__x: tf.Tensor, __c: tf.Tensor):
            return tf.multiply(tf.add(__x, __c), __c)

        sampling_repeats = 100
        samples_in_single = 100

        x = tf.random.normal(tensor_shape)
        c = 1.0

        sample_1 = self.run_benchmark_samples(
            lambda: __graph_function(x, c),
            sampling_repeats,
            samples_in_single,
        )

        def __non_graph_function(__x: tf.Tensor, __c: tf.Tensor):
            return tf.multiply(tf.add(__x, __c), __c)

        x = tf.random.normal(tensor_shape)
        c = tf.ones(tensor_shape)

        sample_2 = self.run_benchmark_samples(
            lambda: __non_graph_function(x, c),
            sampling_repeats,
            samples_in_single,
        )

        plot_with_stats(
            sample_1,
            sample_2,
            labels=(
                "graph",
                "non graph",
            ),
        )

    @pytest.mark.benchmark()
    @pytest.mark.parametrize("tensor_shape", SHAPE_SET_1)
    def test_graph_non_graph_cpu(self, tensor_shape: Tuple[int, int]):
        disable_gpu_or_skip()
        self.benchmark_graph_non_graph(tensor_shape)
        self.dump_plot("graph non graph cpu", tensor_shape)

    @pytest.mark.benchmark()
    @pytest.mark.parametrize("tensor_shape", SHAPE_SET_1)
    def test_graph_non_graph_gpu(self, tensor_shape: Tuple[int, int]):
        skip_if_no_gpu()
        self.benchmark_graph_non_graph(tensor_shape)
        self.dump_plot("graph non graph gpu", tensor_shape)
