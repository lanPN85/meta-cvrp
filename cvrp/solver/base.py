from abc import ABC, abstractmethod
from typing import Optional

from cvrp.model.problem import ProblemInstance
from cvrp.model.solution import ProblemSolution


class ISolver(ABC):
    def __init__(self, logdir=None) -> None:
        self.logdir = logdir

    @abstractmethod
    def solve(self, problem: ProblemInstance) -> Optional[ProblemSolution]:
        raise NotImplementedError
