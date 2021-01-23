from abc import ABC, abstractmethod
from typing import IO, List
from io import BytesIO

from cvrp.model.solution import ProblemSolution
from cvrp.model.problem import ProblemInstance


class ISolutionSerializer(ABC):
    @abstractmethod
    def save_solution(self,
        solution: ProblemSolution,
        instance: ProblemInstance,
        f: BytesIO
    ):
        raise NotImplementedError


class ISolutionSummarizer(ABC):
    @abstractmethod
    def summarize_solutions(self,
        solutions: List[ProblemSolution],
        instances: List[ProblemInstance],
        f: BytesIO
    ):
        raise NotImplementedError
