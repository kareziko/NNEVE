from typing import List

from pydantic import Field

from .base import QOTrackerBase


class QOTracker(QOTrackerBase):

    eigenvalue: List[float] = Field(default_factory=list)
    total_loss: List[float] = Field(default_factory=list)
    drive_loss: List[float] = Field(default_factory=list)
    function_loss: List[float] = Field(default_factory=list)
    lambda_loss: List[float] = Field(default_factory=list)
    c: List[float] = Field(default_factory=list)

    def push_stats(  # noqa: CFQ002
        self,
        eigenvalue: float,
        total_loss: float,
        function_loss: float,
        lambda_loss: float,
        drive_loss: float,
        c: float,
    ) -> None:
        self.eigenvalue.append(float(eigenvalue))
        self.total_loss.append(float(total_loss))
        self.function_loss.append(float(function_loss))
        self.lambda_loss.append(float(lambda_loss))
        self.drive_loss.append(float(drive_loss))
        self.c.append(c)

    def get_trace(self, index: int) -> str:
        return (
            f"epoch: {index:6.0f}, loss: {self.loss[-1]:10.4f}, λ: "
            f"{self.eigenvalue[-1]:10.4f}, c: {self.c[-1]:5.2f}"
        )
