import random
import time

from typing import Any, Dict, Optional, List
from loguru import logger
from pydantic import BaseModel

from cvrp.model.solution import ProblemSolution, SolutionMetadata
from cvrp.model.problem import ProblemInstance
from cvrp.solver.base import ISolver
from cvrp.solver.cw import ClarkeWrightSolver
from .tabu_list import TabuList


class Neighbor(BaseModel):
    solution: ProblemSolution
    move_key: str


class LocalSearchSolver(ISolver):
    def __init__(
        self,
        max_iters=500,
        early_stop=20,
        max_neighbor_size=1000,
        with_tabu=True,
        tabu_size=100,
        timeout_s=None,
        operators=("relocate", "2opt*"),
    ) -> None:
        self.max_iters = max_iters
        self.early_stop = early_stop
        self.max_neighbor_size = max_neighbor_size
        self.with_tabu = with_tabu
        self._tabu_list = TabuList(tabu_size)
        self.timeout_s = timeout_s
        self.operators = set(operators)

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

        start_time = time.time()

        for iter in range(self.max_iters):
            logger.debug(f"Iter {iter}. Best fitness: {best_fitness}")
            logger.debug("Creating neighbors")

            neighbors = self._create_neighbors(current_solution, problem)
            logger.debug(f"Neighborhood size: {len(neighbors)}")

            if self.max_neighbor_size > 0 and len(neighbors) > self.max_neighbor_size:
                logger.debug(f"Sampling {self.max_neighbor_size} neighbors")
                neighbors = random.sample(neighbors, self.max_neighbor_size)

            def _is_not_tabu(n: Neighbor):
                return (
                    not self._tabu_list.contains(n.move_key)
                    or n.solution.total_cost(problem) < iter_best_fitness
                )

            # Remove neighbors in tabu list
            if self.with_tabu:
                neighbors = list(filter(_is_not_tabu, neighbors))
                logger.debug(f"Filtered neighborhood size: {len(neighbors)}")

            # Add neighbors to tabu list
            if self.with_tabu:
                logger.debug("Updating tabu list")
                self._tabu_list.add_all([n.move_key for n in neighbors])

            candidates = [n.solution for n in neighbors]
            iter_best_fitness = float("inf")
            iter_best_cand = None
            for cand in candidates:
                fitness = cand.total_cost(problem)
                if fitness < iter_best_fitness:
                    iter_best_fitness = fitness
                    iter_best_cand = cand

            if iter_best_cand is not None:
                current_fitness = iter_best_fitness
                current_solution = iter_best_cand
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

            # Check early stop
            if self.early_stop > 0 and non_improve_count >= self.early_stop:
                logger.warning(
                    f"Fitness did not improve over {non_improve_count} iterations. Stopping"
                )
                break

            # Check time limit
            elapsed = time.time() - start_time
            logger.debug(f"Elapsed time: {elapsed:.2f}s")
            if self.timeout_s is not None and elapsed >= self.timeout_s:
                logger.warning("Time limit exceeded. Stopping")
                break

        meta = SolutionMetadata(extras=extras)
        best_solution.meta = meta
        return best_solution

    def _create_neighbors(
        self, solution: ProblemSolution, problem: ProblemInstance
    ) -> List[Neighbor]:
        neighbors: List[Neighbor] = []

        if "relocate" in self.operators:
            logger.debug(f"Using relocate")
            neighbors.extend(self._relocate(solution, problem))

        if "2opt*" in self.operators:
            logger.debug(f"Using 2-opt*")
            neighbors.extend(self._2opt_star(solution, problem))

        if "oropt" in self.operators:
            logger.debug("Using or-opt")
            neighbors.extend(self._or_opt(solution, problem))

        return neighbors

    def _or_opt(
        self, solution: ProblemSolution, problem: ProblemInstance
    ) -> List[Neighbor]:
        neighbors: List[Neighbor] = []

        # Iterate each pair of routes
        for i, route1 in enumerate(solution.routes[:-1]):
            d1 = route1.total_demand()
            for j, route2 in enumerate(solution.routes[i:]):
                d2 = route2.total_demand()
                # TODO Implement

        return neighbors

    def _2opt_star(
        self, solution: ProblemSolution, problem: ProblemInstance
    ) -> List[Neighbor]:
        neighbors: List[Neighbor] = []

        # Iterate each pair of routes
        for i, route1 in enumerate(solution.routes[:-1]):
            d1 = route1.total_demand()
            for j, route2 in enumerate(solution.routes[i:]):
                d2 = route2.total_demand()
                for k, n1 in enumerate(route1.nodes[1:-1]):
                    for l, n2 in enumerate(route2.nodes[1:-1]):
                        # Check if the swap still maintains total demand constraint
                        exceed_1 = d1 - n1.demand + n2.demand > problem.vehicle_capacity
                        exceed_2 = d2 - n2.demand + n1.demand > problem.vehicle_capacity
                        if exceed_1 or exceed_2:
                            continue

                        # Swap n1 and n2 to the other's route
                        new_solution = solution.copy(deep=True)
                        new_route1 = new_solution.routes[i]
                        new_route2 = new_solution.routes[i + j]
                        new_route1.nodes[k + 1] = n2
                        new_route2.nodes[l + 1] = n1

                        id_list = sorted([n1.id, n2.id])
                        id_str = "_".join(id_list)
                        neighbors.append(
                            Neighbor(solution=new_solution, move_key=f"2opt*/{id_str}")
                        )

        return neighbors

    def _relocate(
        self, solution: ProblemSolution, problem: ProblemInstance
    ) -> List[Neighbor]:
        neighbors: List[Neighbor] = []

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
                    neighbors.append(
                        Neighbor(
                            solution=new_solution_1,
                            move_key=f"relocate/{n1.id}_{n2.id}",
                        )
                    )

                    # Move n1 to after n2
                    new_solution_2 = solution.copy(deep=True)
                    new_nodes_2 = new_solution_2.routes[r].nodes
                    new_nodes_2.insert(j, new_nodes_2.pop(i))
                    neighbors.append(
                        Neighbor(
                            solution=new_solution_2,
                            move_key=f"relocate/{n2.id}_{n1.id}",
                        )
                    )

        return neighbors
