from typing import Dict, List, Tuple
from loguru import logger

from cvrp.model.problem import ProblemInstance, Node
from cvrp.model.solution import ProblemSolution, Route

from .base import ISolver


class ClarkeWrightSolver(ISolver):
    """
    Solver using the savings algorithm proposed by Clarke and Wright
    """
    def __init__(self) -> None:
        pass

    def solve(self, problem: ProblemInstance) -> ProblemSolution:
        logger.info("Generate savings list")
        savings = self._create_savings_list(problem)

        capacity = problem.vehicle_capacity
        route_counter = 1
        routes: Dict[str, Route] = {}
        assignments: Dict[str, str] = {}  # Maps node id to its assigned route

        for i, (id1, id2, s) in enumerate(savings):
            logger.info(f"Iter {i}: {id1} -> {id2} ({s})")
            is_id1_assigned = id1 in assignments.keys()
            is_id2_assigned = id2 in assignments.keys()
            n1 = problem.get_node_by_id(id1)
            n2 = problem.get_node_by_id(id2)
            total_demand = n1.demand + n2.demand

            if not is_id1_assigned and not is_id2_assigned:
                if total_demand > capacity:
                    logger.info(f"Total demand ({total_demand}) exceeds capacity. Skipping")
                    continue

                route = Route(
                    id=str(route_counter),
                    nodes=[
                        problem.depart_node,
                        n1, n2,
                        problem.arrive_node
                    ]
                )
                route_counter += 1
                assignments[id1] = route.id
                assignments[id2] = route.id
                routes[route.id] = route
                logger.info(f"Added new route {route.id}")
                continue

            if is_id1_assigned != is_id2_assigned:
                assigned_node = n1 if is_id1_assigned else n2
                assigned_id = assigned_node.id
                unassigned_node = n2 if is_id1_assigned else n1
                unassigned_id = unassigned_node.id
                route_id = assignments[assigned_id]
                route = routes[route_id]

                is_not_interior = (
                    assigned_id == route.nodes[1].id or
                    assigned_id == route.nodes[-2].id
                )
                if not is_not_interior:
                    logger.info(f"Node {assigned_id} is interior to route {route_id}. Skipping")
                    continue

                # Decide if adding the node would violate capacity constraints
                current_demand = route.total_demand()
                projected_demand = current_demand + unassigned_node.demand
                if projected_demand > capacity:
                    logger.info(f"Adding node would exceed capacity ({projected_demand}). Skipping")
                    continue

                insert_index = route.nodes.index(assigned_node)
                if assigned_id == id1:
                    insert_index += 1

                route.nodes.insert(insert_index, unassigned_node)

                assignments[unassigned_id] = route.id
                logger.info(f"Added node {unassigned_id} to route {route.id}")
                continue

            if is_id1_assigned and is_id2_assigned:
                route_id_1 = assignments[id1]
                route_id_2 = assignments[id2]
                if route_id_1 == route_id_2:
                    logger.info(f"Identical route {route_id_1}. Skipping")
                    continue

                route_1 = routes[route_id_1]
                route_2 = routes[route_id_2]
                projected_demand = route_1.total_demand() + route_2.total_demand()
                if projected_demand > capacity:
                    logger.info(f"Merging routes excceed capacity ({projected_demand}). Skipping")
                    continue

                merged_nodes = route_1.nodes[:-1] + route_2.nodes[1:]
                merged_route = Route(
                    id=str(route_counter),
                    nodes=merged_nodes
                )
                route_counter += 1

                routes[merged_route.id] = merged_route
                routes.pop(route_id_1)
                routes.pop(route_id_2)
                for node in merged_route.nodes:
                    assignments[node.id] = merged_route.id

                logger.info(f"Merged routes {route_id_1} and {route_id_2} into route {merged_route.id}")

        for node in problem.nodes:
            if node.id in (problem.depart_node_id, problem.arrive_node_id):
                continue
            
            node_id = node.id
            if node_id in assignments.keys():
                continue

            route = Route(
                id=str(route_counter),
                nodes=[
                    problem.depart_node,
                    node,
                    problem.arrive_node
                ]
            )
            route_counter += 1
            routes[route.id] = route
            logger.info(f"Added single route {route.id} for node {node_id}")

        route_list = list(routes.values())

        return ProblemSolution(
            instance_name=problem.name,
            routes=route_list
        )


    def _create_savings_list(
        self, problem: ProblemInstance
    ) -> List[Tuple[str, str, float]]:
        savings = []

        for n1 in problem.nodes[:]:
            if n1.id in (problem.depart_node_id, problem.arrive_node_id):
                continue
            for n2 in problem.nodes[:]:
                if n1.id == n2.id:
                    continue
                if n2.id in (problem.depart_node_id, problem.arrive_node_id):
                    continue

                w1 = problem.get_weight(problem.depart_node_id, n1.id)
                w2 = problem.get_weight(n2.id, problem.arrive_node_id)
                w12 = problem.get_weight(n1.id, n2.id)
                s = w1 + w2 - w12
                savings.append((n1.id, n2.id, s))

        # Sort the list
        savings = sorted(savings, key=lambda x: x[2], reverse=True)

        return savings
