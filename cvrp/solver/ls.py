from typing import Any, Dict, Optional, List
from loguru import logger

from cvrp.model.solution import ProblemSolution, SolutionMetadata
from cvrp.model.problem import ProblemInstance
from .base import ISolver
from .cw import ClarkeWrightSolver


class LocalSearchSolver(ISolver):
    def __init__(self, max_iters=500, early_stop=20) -> None:
        self.max_iters = max_iters
        self.early_stop = early_stop

        super().__init__()

    def solve(self, problem: ProblemInstance) -> Optional[ProblemSolution]:
        logger.debug("Using CW for initial solution")
        init_solver = ClarkeWrightSolver()
        initial_solution = init_solver.solve(problem)

        if initial_solution is None:
            logger.error("Failed to get an inital solution")
            return None

        best_solution = initial_solution
        current_solution = initial_solution
        best_fitness = best_solution.total_cost(problem)
        current_fitness = best_fitness
        non_improve_count = 0
        extras: Dict[str, Any] = {"iterations": 0, "fitness": []}

        for iter in range(self.max_iters):
            logger.debug(f"Iter {iter}. Best fitness: {best_fitness}")
            logger.debug("Creating neighbors")
            neighbors = self._create_neighbors(current_solution, problem)
            logger.debug(f"Neighborhood size: {len(neighbors)}")

            iter_best_fitness = float("inf")
            iter_best_neighbor = None
            for neighbor in neighbors:
                fitness = neighbor.total_cost(problem)
                if fitness < iter_best_fitness:
                    iter_best_fitness = fitness
                    iter_best_neighbor = neighbor

            if iter_best_neighbor is not None:
                current_fitness = iter_best_fitness
                current_solution = iter_best_neighbor
            else:
                logger.debug("No best neighbor found")

            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = current_solution
                non_improve_count = 0
            else:
                logger.debug(f"Fitness did not improve ({current_fitness})")
                non_improve_count += 1

            extras["iterations"] = iter + 1
            extras["fitness"].append(current_fitness)

            if self.early_stop > 0 and non_improve_count >= self.early_stop:
                logger.warning(
                    f"Fitness did not improve over {non_improve_count} iterations. Stopping"
                )
                break

        meta = SolutionMetadata(extras=extras)
        best_solution.meta = meta
        return best_solution

    def _create_neighbors(
        self, solution: ProblemSolution, problem: ProblemInstance
    ) -> List[ProblemSolution]:
        neighbors: List[ProblemSolution] = []

        logger.debug(f"Using relocate")
        neighbors.extend(self._relocate(solution, problem))

        return neighbors

    def _relocate(
        self, solution: ProblemSolution, problem: ProblemInstance
    ) -> List[ProblemSolution]:
        neighbors: List[ProblemSolution] = []

        for r, route in enumerate(solution.routes):
            # For each route, iterate each node pair and move n1 next to n2
            for i, n1 in enumerate(route.nodes[:-1]):
                if i == 0:
                    continue
                for j, n2 in enumerate(route.nodes[:-1]):
                    if j == 0:
                        continue
                    if n1.id == n2.id:
                        continue
                    if abs(i - j) == 1:
                        # Skip adjacent nodes
                        continue

                    # Move n1 to before n2
                    new_solution_1 = solution.copy(deep=True)
                    new_nodes_1 = new_solution_1.routes[r].nodes
                    new_nodes_1.insert(j, new_nodes_1.pop(i))
                    neighbors.append(new_solution_1)

                    # Move n1 to after n2
                    new_solution_2 = solution.copy(deep=True)
                    new_nodes_2 = new_solution_2.routes[r].nodes
                    new_nodes_2.insert(j, new_nodes_2.pop(i))
                    neighbors.append(new_solution_2)

        return neighbors
