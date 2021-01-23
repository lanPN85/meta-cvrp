from abc import ABC, abstractmethod

from cvrp.model.problem import ProblemInstance


class IDataset(ABC):
    @abstractmethod
    def __getitem__(self, index) -> ProblemInstance:
        raise NotImplementedError
