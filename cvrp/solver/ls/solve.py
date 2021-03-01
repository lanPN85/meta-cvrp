import random
import time

from typing import Any, Deque, Dict, Optional, List, Set, Tuple
from loguru import logger
from pydantic import BaseModel
from collections import deque, namedtuple

from cvrp.model.solution import (
    ProblemSolution,
    Route,
    SolutionMetadata,
    SolutionValidity,
)
from cvrp.model.problem import ProblemInstance
from cvrp.solver.base import ISolver
from cvrp.solver.cw import ClarkeWrightSolver
from .tabu_list import TabuList
from .freq_mem import FrequencyMemory


class Neighbor:
    def __init__(self, solution: ProblemSolution, move_keys: Set[str]) -> None:
        self.solution = solution
        self.move_keys = move_keys


class SearchState:
    def __init__(
        self,
        best_alltime_fitness: float,
        best_solution: ProblemSolution,
        best_fitness: float,
        current_solution: ProblemSolution,
        current_fitness: float,
        non_improve_count: int,
        tabu_list: TabuList,
        frequency_mem: FrequencyMemory,
        elite_set: Deque[ProblemSolution],
    ) -> None:
        self.best_alltime_fitness = best_alltime_fitness
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.current_solution = current_solution
        self.current_fitness = current_fitness
        self.non_improve_count = non_improve_count
        self.tabu_list = tabu_list
        self.frequency_mem = frequency_mem
        self.elite_set = elite_set


SearchConfig = namedtuple(
    "SearchConfig",
    [
        "iter_key",
        "fitness_key",
    ],
)


class LocalSearchSolver(ISolver):
    def __init__(
        self,
        restarts=2,
        short_term_max_iters=500,
        short_term_early_stop=20,
        divers_max_iters=500,
        divers_early_stop=20,
        intens_max_iters=500,
        intens_early_stop=20,
        max_neighbor_size=1000,
        short_term_tabu_size=100,
        divers_tabu_size=200,
        intens_tabu_size=50,
        timeout_s=None,
        operators=("relocate", "2-opt*", "or-opt"),
        tabu_expire=3,
        tabu_expire_max=5,
        tabu_expire_min=2,
        elite_set_size=5,
        **kwargs,
    ) -> None:
        self.short_term_max_iters = short_term_max_iters
        self.short_term_early_stop = short_term_early_stop
        self.divers_max_iters = divers_max_iters
        self.divers_early_stop = divers_early_stop
        self.intens_max_iters = intens_max_iters
        self.intens_early_stop = intens_early_stop
        self.max_neighbor_size = max_neighbor_size
        self.timeout_s = timeout_s
        self.tabu_expire_max = tabu_expire_max
        self.tabu_expire_min = tabu_expire_min
        self.restarts = restarts
        self.operators = set(operators)
        self.short_term_tabu_size = short_term_tabu_size
        self.divers_tabu_size = divers_tabu_size
        self.intens_tabu_size = intens_tabu_size
        self.tabu_expire = tabu_expire
        self.elite_set_size = elite_set_size

        super().__init__(**kwargs)

    def solve(self, problem: ProblemInstance) -> Optional[ProblemSolution]:
        best_solution = None
        best_fitness = float("inf")
        extras: Dict[str, Any] = {"runs": []}

        # Long term memory
        frequency_mem = FrequencyMemory()
        elite_set: Deque[ProblemSolution] = deque(maxlen=self.elite_set_size)

        start_time = time.time()
        for i in range(self.restarts + 1):
            logger.info(f"Start {i + 1}/{self.restarts + 1}")

            # Perform one run
            run_best, run_extras = self._search(
                problem=problem,
                frequency_mem=frequency_mem,
                elite_set=elite_set,
                start_time=start_time,
                best_alltime_fitness=best_fitness,
            )
            extras["runs"].append(run_extras)

            if run_best is None:
                logger.error("Run failed to produce solution")
                continue

            run_fitness = run_best.total_cost(problem)

            logger.info(f"Run fitness: {run_fitness} | Best: {best_fitness}")
            if run_fitness < best_fitness:
                best_solution = run_best
                best_fitness = run_fitness

            # Check time limit
            elapsed = time.time() - start_time
            logger.debug(f"Elapsed time: {elapsed:.2f}s")
            if self.timeout_s is not None and elapsed >= self.timeout_s:
                logger.warning("Time limit exceeded. Stopping")
                break

        if best_solution is not None:
            meta = SolutionMetadata(extras=extras)
            best_solution.meta = meta
        return best_solution

    def _search(
        self,
        problem: ProblemInstance,
        frequency_mem: FrequencyMemory,
        elite_set: Deque[ProblemSolution],
        start_time: float,
        best_alltime_fitness: float,
    ) -> Tuple[Optional[ProblemSolution], Dict[str, Any]]:
        stop_now = False
        extras: Dict[str, Any] = {
            "short_term_iters": 0,
            "short_term_fitness": [],
            "intens_iters": 0,
            "intens_fitness": [],
            "divers_iters": 0,
            "divers_fitness": [],
        }

        logger.debug("Using CW for initial solution")
        init_solver = ClarkeWrightSolver(savings_weight=self.random_cw_weights(problem))
        initial = init_solver.solve(problem)

        if initial is None:
            logger.error("Failed to get an inital solution")
            return None, extras

        best_solution = initial
        best_fitness = initial.total_cost(problem)

        state = SearchState(
            best_alltime_fitness=best_alltime_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            current_solution=initial,
            current_fitness=best_fitness,
            non_improve_count=0,
            tabu_list=TabuList(self.short_term_tabu_size, self.tabu_expire),
            frequency_mem=frequency_mem,
            elite_set=elite_set,
        )

        # Start short-term phase
        logger.info("Start short-term phase")
        state.non_improve_count = 0
        config = SearchConfig(
            iter_key="short_term_iters",
            fitness_key="short_term_fitness",
        )
        for iter in range(self.short_term_max_iters):
            self._search_iter(problem, iter, state, config, extras)

            # Check early stop
            if (
                self.short_term_early_stop > 0
                and state.non_improve_count >= self.short_term_early_stop
            ):
                logger.warning(
                    f"Fitness did not improve over {state.non_improve_count} iterations. Stopping"
                )
                break

            # Check time limit
            elapsed = time.time() - start_time
            logger.debug(f"Elapsed time: {elapsed:.2f}s")
            if self.timeout_s is not None and elapsed >= self.timeout_s:
                logger.warning("Time limit exceeded. Stopping")
                stop_now = True
                break

        if state.best_fitness < best_fitness:
            best_fitness = state.best_fitness
            best_solution = state.best_solution
        if stop_now:
            return best_solution, extras
        # End short-term phase

        # ====================

        # Start intensification phase
        logger.info("Start intensification phase")

        # Select solution from the elite set
        if len(elite_set) < 1:
            raise ValueError("Empty elite set")
        state.current_solution = random.sample(elite_set, 1)[0]
        state.current_fitness = state.current_solution.total_cost(problem)
        state.best_solution = state.current_solution
        state.best_fitness = state.current_fitness

        state.tabu_list.empty()
        state.tabu_list.capacity = self.intens_tabu_size

        config = SearchConfig(iter_key="intens_iters", fitness_key="intens_fitness")
        state.non_improve_count = 0
        for iter in range(self.intens_max_iters):
            self._search_iter(problem, iter, state, config, extras)

            # Check early stop
            if (
                self.intens_early_stop > 0
                and state.non_improve_count >= self.intens_early_stop
            ):
                logger.warning(
                    f"Fitness did not improve over {state.non_improve_count} iterations. Stopping"
                )
                break

            # Check time limit
            elapsed = time.time() - start_time
            logger.debug(f"Elapsed time: {elapsed:.2f}s")
            if self.timeout_s is not None and elapsed >= self.timeout_s:
                logger.warning("Time limit exceeded. Stopping")
                stop_now = True
                break

        if state.best_fitness < best_fitness:
            best_fitness = state.best_fitness
            best_solution = state.best_solution
        if stop_now:
            return best_solution, extras
        # End intensification phase

        # =========================

        # Start diversification phase
        config = SearchConfig(iter_key="divers_iters", fitness_key="divers_fitness")
        penalty_ratios = state.frequency_mem.get_penalty_ratios()
        divers_solver = ClarkeWrightSolver(penalty_ratios)
        state.current_solution = divers_solver.solve(problem)  # type: ignore
        state.current_fitness = state.current_solution.total_cost(problem)
        state.best_solution = state.current_solution
        state.best_fitness = state.current_fitness

        if state.current_fitness < state.best_fitness:
            state.best_fitness = state.current_fitness
            state.best_solution = state.current_solution

        state.non_improve_count = 0

        state.tabu_list.empty()
        state.tabu_list.capacity = self.divers_tabu_size
        for iter in range(self.divers_max_iters):
            self._search_iter(problem, iter, state, config, extras)

            # Check early stop
            if (
                self.divers_early_stop > 0
                and state.non_improve_count >= self.divers_early_stop
            ):
                logger.warning(
                    f"Fitness did not improve over {state.non_improve_count} iterations. Stopping"
                )
                break

            # Check time limit
            elapsed = time.time() - start_time
            logger.debug(f"Elapsed time: {elapsed:.2f}s")
            if self.timeout_s is not None and elapsed >= self.timeout_s:
                logger.warning("Time limit exceeded. Stopping")
                break
        # End diversification phase

        if state.best_fitness < best_fitness:
            best_fitness = state.best_fitness
            best_solution = state.best_solution

        return best_solution, extras

    def _search_iter(
        self,
        problem: ProblemInstance,
        iter: int,
        state: SearchState,
        config: SearchConfig,
        extras: Dict[str, Any],
    ):
        # Randomize tabu expire
        state.tabu_list.retain = random.randrange(
            self.tabu_expire_min, self.tabu_expire_max
        )

        logger.info(f"Iter {iter}. Best fitness: {state.best_fitness}")
        logger.debug("Creating neighbors")

        neighbors = self._create_neighbors(state.current_solution, problem)
        logger.debug(f"Neighborhood size: {len(neighbors)}")

        if self.max_neighbor_size > 0 and len(neighbors) > self.max_neighbor_size:
            logger.debug(f"Sampling {self.max_neighbor_size} neighbors")
            neighbors = random.sample(neighbors, self.max_neighbor_size)
            random.shuffle(neighbors)

        def _is_not_tabu(n: Neighbor):
            return (
                not state.tabu_list.intersects(n.move_keys)
                or n.solution.total_cost(problem) < state.best_fitness
            )

        # Remove neighbors in tabu list
        neighbors = list(filter(_is_not_tabu, neighbors))
        logger.debug(f"Filtered neighborhood size: {len(neighbors)}")

        state.tabu_list.step()

        iter_best_fitness = float("inf")
        iter_best_cand = None
        iter_best_neighbor = None
        for n in neighbors:
            cand = n.solution
            fitness = cand.total_cost(problem)
            if fitness < iter_best_fitness:
                iter_best_fitness = fitness
                iter_best_cand = cand
                iter_best_neighbor = n

            # Update tabu list
            state.tabu_list.add_all(list(n.move_keys))

        if iter_best_neighbor is not None:
            state.tabu_list.add_all(list(iter_best_neighbor.move_keys))

        if iter_best_cand is not None:
            state.current_fitness = iter_best_fitness
            state.current_solution = iter_best_cand

            # Update frequency memory
            state.frequency_mem.update(state.current_solution)

            # Update elite set
            if state.current_fitness < state.best_alltime_fitness:
                state.best_alltime_fitness = state.current_fitness
                state.elite_set.append(state.current_solution)
        else:
            logger.debug("No best neighbor found")

        if state.current_fitness < state.best_fitness:
            state.best_fitness = state.current_fitness
            state.best_solution = state.current_solution
            state.non_improve_count = 0
        else:
            logger.debug(f"Fitness did not improve ({state.current_fitness})")
            state.non_improve_count += 1

        extras[config.iter_key] = iter + 1
        extras[config.fitness_key].append(state.current_fitness)

    def _create_neighbors(
        self, solution: ProblemSolution, problem: ProblemInstance
    ) -> List[Neighbor]:
        neighbors: List[Neighbor] = []

        if "relocate" in self.operators:
            logger.debug(f"Using relocate")
            neighbors.extend(self._relocate(solution, problem))

        if "2-opt*" in self.operators:
            logger.debug(f"Using 2-opt*")
            neighbors.extend(self._2opt_star(solution, problem))

        if "or-opt" in self.operators:
            logger.debug("Using or-opt")
            neighbors.extend(self._or_opt(solution, problem))

        return neighbors

    def _or_opt(
        self, solution: ProblemSolution, problem: ProblemInstance
    ) -> List[Neighbor]:
        neighbors: List[Neighbor] = []

        # Iterate each pair of routes
        for i, route1 in enumerate(solution.routes):
            if len(route1.nodes) < 5:
                continue
            for j, route2 in enumerate(solution.routes):
                if i == j:
                    continue
                d2 = route2.total_demand()

                for i1, n1 in enumerate(route1.nodes[1:-3]):
                    max_sigma = len(route1.nodes) - i1 - 2
                    sigma = random.randrange(2, max_sigma)
                    segment = route1.nodes[i1 + 1 : i1 + sigma + 1]
                    segment_route = Route(id="", nodes=segment)

                    # If the move causes demand violation, skip
                    if d2 + segment_route.total_demand() > problem.vehicle_capacity:
                        continue

                    for i2, n2 in enumerate(route2.nodes[1:-1]):
                        new_solution = solution.copy(deep=True)
                        new_route1 = new_solution.routes[i]
                        new_route2 = new_solution.routes[j]

                        # Cut the segment from the first route
                        new_route1.nodes = (
                            new_route1.nodes[: i1 + 1]
                            + new_route1.nodes[i1 + sigma + 1 :]
                        )
                        # Add to the second route
                        new_route2.nodes = (
                            new_route2.nodes[: i2 + 1]
                            + segment
                            + new_route2.nodes[i2 + 1 :]
                        )

                        neighbors.append(
                            Neighbor(
                                solution=new_solution,
                                move_keys=set([f"or-opt/{n1.id}", f"or-opt/{n2.id}"]),
                            )
                        )

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
                    # Get a random node from route2
                    l = random.sample(list(range(1, len(route2.nodes) - 1)), 1)[0] - 1
                    n2 = route2.nodes[l + 1]

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
                        Neighbor(
                            solution=new_solution,
                            move_keys=set([f"2opt*/{n1.id}", f"2opt*/{n2.id}"]),
                        )
                    )

        return neighbors

    def _relocate(
        self, solution: ProblemSolution, problem: ProblemInstance
    ) -> List[Neighbor]:
        neighbors: List[Neighbor] = []

        for r, route in enumerate(solution.routes):
            if len(route.nodes) < 6:
                continue

            # For each route, select a node pair and move n1 next to n2
            for i, n1 in enumerate(route.nodes[:-1]):
                if i == 0:
                    continue

                indices = list(range(1, len(route.nodes) - 1))
                indices.remove(i)
                if i < len(route.nodes) - 2:
                    indices.remove(i + 1)
                if i > 1:
                    indices.remove(i - 1)
                j = random.sample(indices, 1)[0]
                n2 = route.nodes[j]

                # Move n1 to before n2
                new_solution_1 = solution.copy(deep=True)
                new_nodes_1 = new_solution_1.routes[r].nodes
                new_nodes_1.insert(j, new_nodes_1.pop(i))
                neighbors.append(
                    Neighbor(
                        solution=new_solution_1,
                        move_keys=set(
                            [
                                f"relocate/{n1.id}_{n2.id}",
                            ]
                        ),
                    )
                )

                # Move n1 to after n2
                new_solution_2 = solution.copy(deep=True)
                new_nodes_2 = new_solution_2.routes[r].nodes
                new_nodes_2.insert(j, new_nodes_2.pop(i))
                neighbors.append(
                    Neighbor(
                        solution=new_solution_2,
                        move_keys=set(
                            [
                                f"relocate/{n2.id}_{n1.id}",
                            ]
                        ),
                    )
                )

        return neighbors

    @staticmethod
    def random_cw_weights(problem: ProblemInstance) -> Dict[Tuple[str, str], float]:
        weights: Dict[Tuple[str, str], float] = {}

        for n1 in problem.nodes:
            for n2 in problem.nodes:
                if n2.id == n1.id:
                    continue
                v = random.uniform(0.0, 1.0)
                weights[(n1.id, n2.id)] = v

        return weights
