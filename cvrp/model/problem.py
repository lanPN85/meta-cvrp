import networkx as nx

from pydantic import BaseModel, Field
from typing import List, Optional

from .node import Node


class ProblemInstance:
    def __init__(self,
        name: str,
        vehicle_count: int,
        vehicle_capacity: float,
        depart_node_id: str,
        arrive_node_id: str,
        nodes: List[Node]
    ) -> None:
        super().__init__()
        self.name = name
        self.vehicle_count = vehicle_count
        self.vehicle_capacity = vehicle_capacity
        self.depart_node_id = depart_node_id
        self.arrive_node_id = arrive_node_id
        self.__nodes = nodes

        self.__calculate_weights()
        self.__generate_mapping()

    def __generate_mapping(self):
        self.__id2node = {}
        for node in self.nodes:
            self.__id2node[node.id] = node

    def get_node_by_id(self, id: str) -> Node:
        return self.__id2node[id]

    def get_weight(self, id1: str, id2: str) -> float:
        if id1 == id2:
            return 0
        return self.__weights[(id1, id2)]

    def __calculate_weights(self):
        self.__weights = {}
        for i, n1 in enumerate(self.nodes):
            for n2 in self.nodes:
                if n1.id == n2.id:
                    continue
                weight = n1.distance_to(n2)
                self.__weights[(n1.id, n2.id)] = weight
                self.__weights[(n2.id, n1.id)] = weight

    @property
    def depart_node(self) -> Node:
        return self.get_node_by_id(self.depart_node_id)

    @property
    def arrive_node(self) -> Node:
        return self.get_node_by_id(self.arrive_node_id)

    @property
    def nodes(self) -> List[Node]:
        return self.__nodes

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

