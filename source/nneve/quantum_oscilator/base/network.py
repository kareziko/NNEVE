import typing
from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple, TypeVar

import numpy as np
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
    loss_function: LossFunctionT

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    def __init__(self, constants: QOConstantsBase):
        self.constants = constants
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
        return self._train(params, epohs)

    def _train(
        self: QOBase_Self, params: QOParamsBase, epohs: int
    ) -> Optional[QOBase_Self]:
        smallest_loss = np.inf
        best_model = None
        with Progress(transient=True) as progress:
            task = progress.add_task("Learning...", total=epohs)

        for i in range(epohs):
            loss, *stats = self.train_step(params)

            self.tracker.push_loss(loss)
            self.tracker.push_stats(*stats)
            description = self.tracker.get_trace(i)
            progress.update(task, advance=1, description=description)

            # TODO Check performance impact
            if loss < smallest_loss:
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
            loss_tensor, *stats = self.loss_function(deriv_x, x, *extra)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_tensor, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return tf.reduce_mean(loss_tensor), *stats

    def get_x(self, params: QOParamsBase) -> tf.Tensor:
        ...

    def extra_hook(self, params: QOParamsBase) -> Tuple[Any, ...]:
        return ()
