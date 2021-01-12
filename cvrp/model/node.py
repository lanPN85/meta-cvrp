from pydantic import BaseModel
from enum import IntEnum


class Node(BaseModel):
    id: str
    cx: float
    cy: float
