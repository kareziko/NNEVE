from .base import QOParamsBase


class QOParams(QOParamsBase):

    c: float

    def update(self) -> None:
        self.c += 0.16
