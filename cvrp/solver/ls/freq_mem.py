from typing import Dict, Tuple

from cvrp.model.solution import ProblemSolution, ProblemInstance


class FrequencyMemory:
    def __init__(self) -> None:
        self._delegate: Dict[Tuple[str, str], int] = {}

    def update(self, solution: ProblemSolution):
        for route in solution.routes:
            last_node = route.nodes[0]
            for current_node in route.nodes[1:]:
                edge = (last_node.id, current_node.id)
                self._delegate[edge] = self._delegate.get(edge, 0) + 1
                last_node = current_node

    def get_penalty_ratios(self) -> Dict[Tuple[str, str], float]:
        """
        Get the penalty ratio dictionary, where keys are tuples denoting edges
        by node id, and values are in the range (0, 1). Higher values means lower penalty

        :rtype: Dict[Tuple[str, str], float]
        """
        # Normalize values
        values = list(self._delegate.values())
        total = sum(values)
        penalty = {}
        for k in self._delegate.keys():
            penalty[k] = 1 - self._delegate[k] / total

        return penalty
