from cvrp.model.node import Node
import os
import xmltodict

from typing import List, Optional

from .base import IDataset
from cvrp.model.problem import ProblemInstance


class XmlDataset(IDataset):
    def __init__(self, paths: List[str]) -> None:
        super().__init__()
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> ProblemInstance:
        return self.parse_file(self.paths[index])

    @staticmethod
    def parse_file(path) -> ProblemInstance:
        with open(path, "rb") as f:
            d = xmltodict.parse(f)
            instance_d = d["instance"]

            # Parse the dictionary
            name = instance_d["info"]["name"]
            vehicle_capacity = float(instance_d["fleet"]["vehicle_profile"]["capacity"])
            depart_node_id = instance_d["fleet"]["vehicle_profile"]["departure_node"]
            arrive_node_id = instance_d["fleet"]["vehicle_profile"]["arrival_node"]

            # Request map
            request_map = {}
            for request_d in instance_d["requests"]["request"]:
                node_id = request_d["@node"]
                quantity = request_d["quantity"]
                request_map[node_id] = quantity

            # Construct the node list
            nodes: List[Node] = []
            for node_d in instance_d["network"]["nodes"]["node"]:
                quantity = request_map.get(node_d["@id"], 0.0)
                node = Node(
                    id=node_d["@id"], cx=node_d["cx"], cy=node_d["cy"], demand=quantity
                )
                nodes.append(node)

            vehicle_count = XmlDataset.parse_vehicle_count(name)
            if vehicle_count is None:
                vehicle_count = len(
                    list(
                        filter(
                            lambda x: x.id != depart_node_id and x.id != arrive_node_id,
                            nodes,
                        )
                    )
                )

            return ProblemInstance(
                name=name,
                vehicle_count=vehicle_count,
                vehicle_capacity=vehicle_capacity,
                depart_node_id=depart_node_id,
                arrive_node_id=arrive_node_id,
                nodes=nodes,
            )

    @staticmethod
    def parse_vehicle_count(name: str) -> Optional[int]:
        try:
            if "_" in name:
                # Handle edge case for Li dataset
                num_ = int(name.split("_")[-1])
                return num_

            comp = name.split("-")[-1]
            num = comp[1:]
            return int(num)
        except ValueError:
            return None


class XmlDirDataset(XmlDataset):
    def __init__(self, dir: str) -> None:
        names = os.listdir(dir)

        # Filter by extension
        names = list(filter(lambda x: x.endswith(".xml"), names))
        paths = [os.path.join(dir, x) for x in names]

        super().__init__(paths)


def test_1():
    instance = XmlDataset.parse_file("data/augerat-1995-A/A-n32-k05.xml")
    print(instance.json(indent=2))


if __name__ == "__main__":
    test_1()
