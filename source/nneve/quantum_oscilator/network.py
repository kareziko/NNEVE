import logging
import typing
from typing import Any, Callable, List, Tuple, cast

import tensorflow as tf
from tensorflow import keras

from nneve.quantum_oscilator.base.params import QOParamsBase
from nneve.quantum_oscilator.params import QOParams

from .base import LossFunctionT, QONetworkBase
from .constants import QOConstants

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811


class QONetwork(QONetworkBase):

    constants: QOConstants

    def assemble_hook(
        self,
    ) -> Tuple[List[keras.layers.InputLayer], List[keras.layers.Dense]]:
        # 1 value input layer
        inputs = cast(
            keras.layers.InputLayer,
            keras.Input(
                shape=(1,),
            ),
        )
        # One neuron decides what value should λ have, default is 1.0
        eigenvalue = keras.layers.Dense(
            1,
            name="eigenvalue",
            kernel_initializer=keras.initializers.constant(0.0),
            # ; bias_initializer=keras.initializers.constant(0.0),
            use_bias=False,
            activation=None,
        )(tf.ones_like(inputs))
        # join x input with λ from single neuron
        # ; concat_input = keras.layers.Concatenate(axis=1)([inputs, eigenvalue])
        concat_input = tf.concat([inputs, eigenvalue], axis=1)
        # two dense layers, each with {neurons} neurons as NN body
        first = keras.layers.Dense(
            self.constants.neuron_count,
            activation=tf.sin,
            name="dense_1",
        )(concat_input)
        # internal second layer
        second = keras.layers.Dense(
            self.constants.neuron_count,
            activation=tf.sin,
            name="dense_2",
        )(first)
        # single value output from neural network
        outputs = keras.layers.Dense(
            1,
            # ; activation=tf.sin,
            name="predictions",
        )(second)
        # single output from full network, λ is accessed by single call
        # to "eigenvalue" Dense layer - much cheaper op
        return [inputs], [outputs]

    def post_assemble_hook(self) -> None:
        self.eigenvalue_function = keras.backend.function(
            [self.inputs], [self.get_layer("eigenvalue").output]
        )

    def get_loss_function(self) -> LossFunctionT:  # noqa: CFQ004, CFQ001
        @tf.function
        def potential(x: tf.Tensor) -> tf.Tensor:
            return tf.divide(
                tf.multiply(tf.constant(self.constants.k), tf.square(x)),
                2,
            )

        self._potential_function = potential

        @tf.function
        def boundary(x: tf.Tensor) -> tf.Tensor:
            return (
                1 - tf.exp(tf.negative(x) + tf.constant(self.constants.x_left))
            ) * (1 - tf.exp(x - tf.constant(self.constants.x_right)))

        @tf.function
        def parametric_solutions(x: tf.Variable) -> tf.Tensor:
            return tf.add(
                tf.constant(self.constants.fb),
                tf.multiply(boundary(x), self(x)),
            )

        @tf.function
        def differentiate(
            function: Callable[[tf.Tensor], tf.Tensor], variable: tf.Variable
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            with tf.GradientTape() as second:
                with tf.GradientTape() as first:
                    y = function(variable)
                dy_dx = first.gradient(y, variable)
            dy_dxx = second.gradient(dy_dx, variable)
            return y, dy_dxx

        @tf.function
        def get_regulators(
            y: tf.Tensor,
            eigenvalue: tf.Tensor,
            c: tf.Tensor,
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            # 1 divided by mean of y's, not mean of 1 divided by y's
            function_loss = tf.divide(
                1,
                tf.add(tf.reduce_mean(tf.square(y)), 1e-6),
            )
            function_loss = tf.multiply(
                function_loss, tf.constant(self.constants.v_lambda)
            )

            lambda_loss = tf.divide(
                1, tf.add(tf.reduce_mean(tf.square(eigenvalue)), 1e-6)
            )
            lambda_loss = tf.multiply(
                lambda_loss, tf.constant(self.constants.v_lambda)
            )

            drive_loss = tf.reduce_mean(tf.exp(tf.subtract(c, eigenvalue)))
            drive_loss = tf.multiply(
                drive_loss, tf.constant(self.constants.v_drive)
            )

            return function_loss, lambda_loss, drive_loss

        @tf.function
        def get_residuum(
            deriv_x: tf.Variable, x: tf.Tensor, eigenvalue: float
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            y, dy_dxx = differentiate(parametric_solutions, deriv_x)  # type: ignore

            # ; Lf = tf.add(
            # ;     tf.divide(
            # ;         dy_dxx, tf.multiply(-2.0, tf.constant(self.constants.mass))
            # ;     ),
            # ;     tf.multiply(potential(x), y),
            # ; )
            # ; tf.square(tf.subtract(Lf, tf.multiply(eigenvalue, y)))
            residuum = (
                tf.divide(
                    dy_dxx, tf.multiply(-2.0, tf.constant(self.constants.mass))
                )
                + (potential(x) * y)
                - (eigenvalue * y)
            )
            return tf.reduce_mean(tf.square(residuum)), y

        @tf.function
        def loss_function(
            deriv_x: tf.Variable,
            x: tf.Tensor,
            c: tf.Tensor,
            eigenvalue: tf.Tensor,
        ) -> Tuple[tf.Tensor, ...]:
            residuum, y = get_residuum(deriv_x, x, eigenvalue)  # type: ignore
            regulators = get_regulators(y, eigenvalue, c)
            return tf.reduce_sum([residuum, *regulators]), eigenvalue, residuum, *regulators, c  # type: ignore

        if self.is_debug:
            self._potential_function = potential
            self._boundary_function = boundary
            self._parametric_solutions_function = parametric_solutions
            self._differentiate_function = differentiate
            self._get_regulators_function = get_regulators
            self._get_residuum_function = get_regulators
            self._loss_function_function = loss_function

        return cast(LossFunctionT, loss_function)

    def get_x(self, params: QOParamsBase) -> tf.Tensor:
        return self.constants.sample()

    def get_eignevalue(self) -> tf.Tensor:
        return self.eigenvalue_function(
            tf.constant([[[1.0]]], dtype=tf.float32)
        )[0][0][0]

    def extra_hook(self, params: QOParams) -> Tuple[Any, ...]:
        return (params.c, self.get_eignevalue())

    def train_generations(
        self,
        params: QOParams,
        generations: int = 5,
        epochs: int = 1000,
    ) -> List["QONetwork"]:
        generation_cache: List["QONetwork"] = []

        for i in range(generations):
            logging.info(f"Generation: {i}")
            best = self.train(params, epochs)
            if best is not None:
                generation_cache.append(best)

            params.update()

        return generation_cache
