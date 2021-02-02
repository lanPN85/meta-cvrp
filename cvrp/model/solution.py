from typing import Dict, List, Set, Any, Optional
from pydantic import BaseModel, validator
from enum import IntEnum
from loguru import logger

from cvrp.model.problem import ProblemInstance, Node


class Route(BaseModel):
    id: str
    nodes: List[Node]

    def total_demand(self) -> float:
        demands = [n.demand for n in self.nodes]
        return sum(demands)

    def total_cost(self, problem: ProblemInstance) -> float:
        cost = 0.

        for j, node in enumerate(self.nodes[:-1]):
            next_node = self.nodes[j + 1]
            path_cost = problem.get_weight(node.id, next_node.id)
            cost += path_cost

        return cost


class SolutionValidity(IntEnum):
    VALID = 0
    NODE_DUPLICATED = 1
    NODE_MISSING = 2
    CAPACITY_EXCEEDED = 3
    INVALID_DEPART = 4
    INVALID_ARRIVE = 5


class SolutionMetadata(BaseModel):
    run_time_ms: int = 0
    extras: Optional[Dict[str, Any]] = None


class ProblemSolution(BaseModel):
    instance_name: str
    routes: List[Route]
    meta: Optional[SolutionMetadata] = None

    @validator("meta", pre=True, always=True)
    def default_meta(cls, v):
        if v is None:
            v = SolutionMetadata()
        return v

    def total_cost(self, problem: ProblemInstance) -> float:
        cost = 0.

        for route in self.routes:
            cost += route.total_cost(problem)

        return cost

    def max_route_cost(self, problem: ProblemInstance) -> float:
        cost = 0.

        for route in self.routes:
            route_cost = route.total_cost(problem)
            cost = max(route_cost, cost)

        return cost

    def validate_(self, problem: ProblemInstance) -> SolutionValidity:
        # Check if all nodes are covered
        covered_node_ids: Set[str] = set([
            problem.depart_node_id,
            problem.arrive_node_id
        ])

        for route in self.routes:
            if route.nodes[0].id != problem.depart_node_id:
                logger.error(f"Route {route.id} starts with invalid node {route.nodes[0].id}")
                return SolutionValidity.INVALID_DEPART

            if route.nodes[-1].id != problem.arrive_node_id:
                logger.error(f"Route {route.id} ends with invalid node {route.nodes[0].id}")
                return SolutionValidity.INVALID_ARRIVE

            # logger.debug(f"{route.id}: {[n.id for n in route.nodes]}")
            for node in route.nodes[1:-1]:
                if node.id in covered_node_ids:
                    logger.error(f"Node {node.id} duplicated")
                    return SolutionValidity.NODE_DUPLICATED
                covered_node_ids.add(node.id)

        if len(covered_node_ids) != problem.num_nodes:
            logger.error(f"Only {len(covered_node_ids)}/{problem.num_nodes} nodes covered")
            return SolutionValidity.NODE_MISSING

        # Check if any routes exceed capacity
        capacity = problem.vehicle_capacity
        for route in self.routes:
            demand = route.total_demand()
            if demand > capacity:
                logger.error(f"Capacity exceeded at route {route.id} (${demand})")
                return SolutionValidity.CAPACITY_EXCEEDED

        return SolutionValidity.VALID
