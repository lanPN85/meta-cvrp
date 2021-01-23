import xml.etree.ElementTree as ET
import sys

from io import BytesIO
from xml.dom import minidom
from typing import IO

from .base import ISolutionSerializer
from cvrp.model.solution import ProblemSolution, Route
from cvrp.model.problem import ProblemInstance
from cvrp.model.node import Node


class XmlSolutionSerializer(ISolutionSerializer):
    def save_solution(self,
        solution: ProblemSolution,
        instance: ProblemInstance,
        f: BytesIO
    ):
        root = ET.Element("solution", {
            "instance": solution.instance_name
        })

        for route in solution.routes:
            route_x = ET.Element("route", {
                "id": route.id
            })
            for node in route.nodes:
                node_x = ET.Element("node", {
                    "id": node.id
                })
                route_x.append(node_x)
            root.append(route_x)

        tree_str = ET.tostring(root, encoding='utf-8')
        pretty_str = minidom.parseString(tree_str)\
            .toprettyxml(indent="    ")\
            .encode('utf-8')

        f.write(pretty_str)


def test_1():
    serializer = XmlSolutionSerializer()
    solution = ProblemSolution(
        instance_name="test",
        routes=[
            Route(
                id="1",
                components=[
                    Node(id="1", cx=0, cy=0),
                    Node(id="2", cx=0, cy=0),
                    Node(id="3", cx=0, cy=0),
                    Node(id="4", cx=0, cy=0),
                    Node(id="5", cx=0, cy=0),
                ]
            ),
            Route(
                id="2",
                components=[
                    Node(id="6", cx=0, cy=0),
                    Node(id="7", cx=0, cy=0),
                ]
            )
        ]
    )

    f = BytesIO()
    serializer.save_solution(solution, f)
    print(f.getvalue().decode('utf-8'))


if __name__ == "__main__":
    test_1()
