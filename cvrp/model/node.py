import math

from pydantic import BaseModel
from enum import IntEnum


class Node(BaseModel):
    id: str
    cx: float
    cy: float
    demand: float = 0.

    def distance_to(self, other: 'Node') -> float:
        return math.sqrt(
            (self.cx - other.cx) ** 2 +
            (self.cy - other.cy) ** 2
        )
