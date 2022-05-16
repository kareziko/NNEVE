from typing import Any, List

from pydantic import BaseModel, Field

__all__ = ["QOTrackerBase"]


class QOTrackerBase(BaseModel):

    loss: List[float] = Field(default_factory=list)

    def push_loss(self, loss_value: float) -> None:
        self.loss.append(loss_value)

    def push_stats(self, *stats: Any) -> None:
        ...

    def get_trace(self, index: int) -> None:
        return f"epoch: {index}, loss: {self.loss[-1]:.4f}"
