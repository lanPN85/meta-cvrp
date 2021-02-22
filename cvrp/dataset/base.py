from abc import ABC, abstractmethod
from typing import Sized

from cvrp.model.problem import ProblemInstance


class IDataset(ABC, Sized):
    @abstractmethod
    def __getitem__(self, index) -> ProblemInstance:
        raise NotImplementedError
