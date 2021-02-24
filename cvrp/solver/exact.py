import pulp
import os

from typing import Optional
from threading import Thread
from pulp import LpProblem, LpVariable, PULP_CBC_CMD
from loguru import logger
from pulp.apis.coin_api import COIN
from pulp.pulp import lpSum

from .base import ISolver
from cvrp.model.problem import ProblemInstance
from cvrp.model.solution import ProblemSolution, Route
from cvrp.dataset.xml import XmlDataset


class ExactPulpSolver(ISolver):
    def __init__(
        self, timeout_s=600, solver_cls=PULP_CBC_CMD, logdir=None, **solver_args
    ) -> None:
        self.solver_args = solver_args
        self.solver_cls = solver_cls
        self.timeout_s = timeout_s

        super().__init__(logdir)

    def convert_solution(
        self, lp_prob: LpProblem, problem: ProblemInstance
    ) -> ProblemSolution:
        routes = []
        route_count = 1
        depot_id = problem.depart_node_id
        customer_nodes = list(filter(lambda v: v.id != depot_id, problem.nodes))
        v_dict = lp_prob.variablesDict()

        # Construct routes using x
        for n_start in customer_nodes:
            x_name = f"x_{depot_id}_{n_start.id}"
            x = pulp.value(v_dict[x_name])
            if x == 0:
                # Node is not a start node
                continue

            nodes = [problem.depart_node, n_start]
            current_node = n_start
            while current_node.id != depot_id:
                has_match = False
                for n_next in customer_nodes:
                    if n_next.id == current_node.id:
                        continue
                    x_name = f"x_{current_node.id}_{n_next.id}"
                    x = pulp.value(v_dict[x_name])
                    if x == 1:
                        has_match = True
                        current_node = n_next
                        nodes.append(n_next)
                        break

                if not has_match:
                    # If no match is found, that means the route is complete
                    nodes.append(problem.arrive_node)
                    break

            route = Route(id=str(route_count), nodes=nodes)
            routes.append(route)
            route_count += 1

        return ProblemSolution(instance_name=problem.name, routes=routes)

    def solve(self, problem: ProblemInstance) -> Optional[ProblemSolution]:
        logger.debug("Modeling")
        lp_prob = self.model_problem(problem)

        log_path = None
        if self.logdir is not None:
            log_path = os.path.join(self.logdir, f"{problem.name}.solver.log")

        solver = self.solver_cls(
            timeLimit=self.timeout_s, logPath=log_path, **self.solver_args
        )

        logger.debug("Solving")
        status = lp_prob.solve(solver=solver)

        if status != pulp.LpStatusOptimal:
            logger.error(f"Could not solve problem optimally ({pulp.LpStatus[status]})")
            return None

        logger.debug("Converting")
        solution = self.convert_solution(lp_prob, problem)
        return solution

    def model_problem(self, problem: ProblemInstance) -> LpProblem:
        """
        Creates problem model according to the second formulation in
        """
        prob = LpProblem("CVRP2", pulp.LpMaximize)
        assert problem.depart_node_id == problem.arrive_node_id
        depot_id = problem.depart_node_id
        customer_nodes = list(filter(lambda v: v.id != depot_id, problem.nodes))
        Q = problem.vehicle_capacity

        X = []
        for n1 in problem.nodes[:]:
            for n2 in problem.nodes[:]:
                if n1.id == n2.id:
                    continue
                x_name = f"x_{n1.id}_{n2.id}"
                x = LpVariable(x_name, cat=pulp.LpBinary)
                X.append(x)

        Y = []
        for n in customer_nodes:
            y_name = f"y_{n.id}"
            y = LpVariable(y_name, lowBound=n.demand, upBound=Q)
            Y.append(y)

        # Number of vehicle constraint
        vars = []
        for n in customer_nodes:
            x_name = f"x_{depot_id}_{n.id}"
            x = next(filter(lambda v: v.name == x_name, X))
            vars.append(x)
        prob += pulp.lpSum(vars) == problem.vehicle_count

        # Each customer is serviced once
        for n2 in customer_nodes[:]:
            vars = []
            for n1 in problem.nodes[:]:
                if n1.id == n2.id:
                    continue
                x_name = f"x_{n1.id}_{n2.id}"
                x = next(filter(lambda v: v.name == x_name, X))
                vars.append(x)
            prob += pulp.lpSum(vars) == 1

        # Each customer is departed at most once
        for n1 in customer_nodes[:]:
            vars = []
            for n2 in customer_nodes[:]:
                if n1.id == n2.id:
                    continue
                x_name = f"x_{n1.id}_{n2.id}"
                x = next(filter(lambda v: v.name == x_name, X))
                vars.append(x)
            prob += pulp.lpSum(vars) <= 1

        # Eliminate sub-tours
        for n1 in customer_nodes[:]:
            for n2 in customer_nodes[:]:
                if n1.id == n2.id:
                    continue
                x_name = f"x_{n1.id}_{n2.id}"
                x = next(filter(lambda v: v.name == x_name, X))
                y1_name = f"y_{n1.id}"
                y1 = next(filter(lambda v: v.name == y1_name, Y))
                y2_name = f"y_{n2.id}"
                y2 = next(filter(lambda v: v.name == y2_name, Y))
                prob += (y1 + n2.demand * x - Q + Q * x) <= y2

        # Optimization criteria
        vars = []
        for n1 in customer_nodes[:]:
            for n2 in customer_nodes[:]:
                if n1.id == n2.id:
                    continue
                x_name = f"x_{n1.id}_{n2.id}"
                x = next(filter(lambda v: v.name == x_name, X))
                c1 = problem.get_weight(n1.id, depot_id)
                c2 = problem.get_weight(depot_id, n2.id)
                c = problem.get_weight(n1.id, n2.id)
                s = c1 + c2 - c
                vars.append(s * x)
        prob += pulp.lpSum(vars)
        return prob


class ExactGurobiSolver(ExactPulpSolver):
    def __init__(self, timeout_s, threads=1) -> None:
        super().__init__(
            timeout_s=timeout_s,
            solver_cls=pulp.GUROBI_CMD,
            threads=threads,
            options=[("TimeLimit", str(timeout_s))],
        )


def test_1():
    instance = XmlDataset.parse_file("data/augerat-1995-A/A-n32-k05.xml")
    solver = ExactPulpSolver(timeout_s=60, solver_cls=pulp.GUROBI_CMD)
    solution = solver.solve(instance)
    print(solution)
    validity = solution.validate_(instance)
    print(validity.name)


if __name__ == "__main__":
    test_1()
