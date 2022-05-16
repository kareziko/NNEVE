import logging
import typing
from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple, TypeVar

import tensorflow as tf
from rich.progress import Progress
from tensorflow import keras

from .constants import QOConstantsBase
from .params import QOParamsBase
from .tracker import QOTrackerBase

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811


__all__ = ["QONetworkBase", "LossFunctionT"]


QOBase_Self = TypeVar("QOBase_Self", bound="QONetworkBase")


LossFunctionT = Callable[
    [tf.Variable, tf.Tensor, QOParamsBase, Any],
    Tuple[tf.Tensor, Tuple[Any, ...]],
]


class QONetworkBase(keras.Model):

    constants: QOConstantsBase
    is_debug: bool
    loss_function: LossFunctionT
    is_console_mode: bool

    class Config:
        allow_mutation = True
        arbitrary_types_allowed = True

    def __init__(
        self,
        constants: QOConstantsBase,
        is_debug: bool = False,
        is_console_mode: bool = True,
    ):
        self.constants = constants
        self.is_console_mode = is_console_mode
        self.is_debug = is_debug

        inputs, outputs = self.assemble_hook()
        super().__init__(inputs=inputs, outputs=outputs)
        self.post_assemble_hook()
        self.loss_function = self.get_loss_function()

    @property
    def optimizer(self) -> keras.optimizers.Optimizer:
        return self.constants.optimizer

    @property
    def tracker(self) -> QOTrackerBase:
        return self.constants.tracker

    def assemble_hook(
        self,
    ) -> Tuple[List[keras.layers.InputLayer], List[keras.layers.Dense]]:
        ...

    def post_assemble_hook(self) -> None:
        ...

    def get_deepcopy(self: QOBase_Self) -> QOBase_Self:
        constants_copy = self.constants.copy(deep=True)
        weights_copy = deepcopy(self.get_weights())
        model_copy = self.__class__(constants=constants_copy)
        model_copy.set_weights(weights_copy)
        return model_copy

    def get_loss_function(self) -> LossFunctionT:
        ...

    def train(self, params: QOParamsBase, epohs: int = 10):
        if self.is_console_mode:
            return self._train_with_console(params, epohs)
        else:
            return self._train_no_console(params, epohs)

    def _train_no_console(
        self: QOBase_Self, params: QOParamsBase, epohs: int
    ) -> Optional[QOBase_Self]:
        smallest_loss = 1e20
        best_model = None

        for i in range(epohs):
            loss, stats = self.train_step(params)

            self.tracker.push_loss(loss)
            self.tracker.push_stats(*stats)
            description = self.tracker.get_trace(i)
            logging.info(description)

            # TODO Check performance impact
            if i % 5 == 0 and loss < smallest_loss:
                best_model = self.get_deepcopy()
                smallest_loss = loss
        return best_model

    def _train_with_console(
        self: QOBase_Self, params: QOParamsBase, epohs: int
    ) -> Optional[QOBase_Self]:
        smallest_loss = 1e20
        best_model = None

        with Progress() as progress:
            task = progress.add_task(
                description="Learning...", total=epohs + 1
            )

            for i in range(epohs):
                loss, stats = self.train_step(params)

                self.tracker.push_loss(loss)
                self.tracker.push_stats(*stats)
                description = self.tracker.get_trace(i)

                progress.update(
                    task,
                    advance=1,
                    description=description.capitalize(),
                )

                # TODO Check performance impact
                if i % 5 == 0 and loss < smallest_loss:
                    best_model = self.get_deepcopy()
                    smallest_loss = loss

        return best_model

    def train_step(
        self, params: QOParamsBase
    ) -> Tuple[float, Tuple[Any, ...]]:
        x = self.get_x(params)
        deriv_x = tf.Variable(initial_value=x)
        extra = self.extra_hook(params)

        with tf.GradientTape() as tape:
            loss_value, stats = self.loss_function(deriv_x, x, *extra)

        trainable_vars = self.trainable_variables
        # ; print([f"{v.shape} -> {float(tf.size(v))}" for v in trainable_vars])
        gradients = tape.gradient(loss_value, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return tf.reduce_mean(loss_value), stats

    def get_x(self, params: QOParamsBase) -> tf.Tensor:
        ...

    def extra_hook(self, params: QOParamsBase) -> Tuple[Any, ...]:
        return ()
