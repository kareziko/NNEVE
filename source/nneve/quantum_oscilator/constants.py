from typing import Optional

import tensorflow as tf
from pydantic import Field

from .base.constants import QOConstantsBase

__all__ = ["QOConstants"]


class QOConstants(QOConstantsBase):
    k: float
    mass: float
    x_left: float
    x_right: float
    fb: float
    sample_size: int = Field(default=500)
    # network configuration
    neuron_count: int = Field(default=50)
    # regularization multipliers
    v_f: float = Field(default=1.0)
    v_lambda: float = Field(default=1.0)
    v_drive: float = Field(default=1.0)
    __sample: Optional[tf.Tensor] = None

    def sample(self) -> tf.Tensor:
        if self.__sample is None:
            self.__sample = tf.cast(
                tf.reshape(
                    tf.linspace(self.x_left, self.x_right, self.sample_size),
                    shape=(-1, 1),
                ),
                dtype=tf.float32,
            )
        return self.__sample
