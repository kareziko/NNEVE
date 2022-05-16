from typing import Any, List

from pydantic import BaseModel, Field

__all__ = ["QOTrackerBase"]


class QOTrackerBase(BaseModel):

    loss: List[float] = Field(default_factory=list)

    def push_loss(self, loss_value: float) -> None:
        self.loss.append(float(loss_value))

    def push_stats(self, *stats: Any) -> None:
        ...

    def get_trace(self, index: int) -> str:
        return f"epoch: {index:6.0f}, loss: {self.loss[-1]:10.4f}"
