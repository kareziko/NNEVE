import typing
from typing import Optional

import tensorflow as tf
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811


class Eigenvalue(keras.layers.Layer):

    w: tf.Variable

    def __init__(self, dtype: tf.DType = tf.float32):
        super().__init__(name="eigenvalue")
        self.w: tf.Variable = self.add_weight(
            dtype=dtype,
            shape=(1, 1),
            initializer=keras.initializers.constant(0.0),
            trainable=True,
        )

    def call(
        self,
        inputs: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        if inputs is not None:
            eigenvalue = tf.multiply(tf.ones_like(inputs), self.w)
            if inputs.dtype != self.w.dtype:
                inputs = tf.cast(inputs, dtype=self.w.dtype)
            return tf.concat([inputs, eigenvalue], axis=1)
        else:
            return tf.constant(self.w)
