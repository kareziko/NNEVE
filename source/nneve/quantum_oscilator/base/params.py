from pydantic import BaseModel

__all__ = ["QOParamsBase"]


class QOParamsBase(BaseModel):
    class Config:
        allow_mutation = True
        arbitrary_types_allowed = True
