from abc import ABC, abstractmethod

from cvrp.model.problem import ProblemInstance
from cvrp.model.solution import ProblemSolution

class ISolver(ABC):
    @abstractmethod
    def solve(self, problem: ProblemInstance) -> ProblemSolution:
        raise NotImplementedError
