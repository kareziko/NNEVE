import typing

from pydantic import BaseModel, Field
from tensorflow import keras

from .tracker import QOTrackerBase

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras  # noqa: F811


__all__ = [
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BETA_1",
    "DEFAULT_BETA_2",
    "QOConstantsBase",
]

DEFAULT_LEARNING_RATE: float = 0.008
DEFAULT_BETA_1: float = 0.999
DEFAULT_BETA_2: float = 0.9999


class QOConstantsBase(BaseModel):
    # network configuration
    optimizer: keras.optimizers.Optimizer = Field(
        default=keras.optimizers.Adam(
            learning_rate=DEFAULT_LEARNING_RATE,
            beta_1=DEFAULT_BETA_1,
            beta_2=DEFAULT_BETA_2,
        )
    )
    tracker: QOTrackerBase = Field(default_factory=QOTrackerBase)

    class Config:
        allow_mutation = True
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
