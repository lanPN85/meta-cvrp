import yaml

from typing import Any, List
from io import BytesIO, StringIO

from .base import ISolutionSummarizer
from cvrp.model.problem import ProblemInstance
from cvrp.model.solution import ProblemSolution


class YamlSolutionSummarizer(ISolutionSummarizer):
    def summarize_solutions(self,
        solutions: List[ProblemSolution],
        instances: List[ProblemInstance],
        f: BytesIO
    ):
        summaries: List[Any] = []

        for solution, instance in zip(solutions, instances):
            d = {
                "instance_name": solution.instance_name,
                "total_cost": solution.total_cost(instance),
                "max_route_cost": solution.max_route_cost(instance),
                "runtime_ms": solution.meta.run_time_ms,
                "extras": solution.meta.extras
            }
            summaries.append(d)

        content = {
            "summary": summaries
        }

        s = StringIO()
        yaml.dump(content, s, indent=2)
        content_bytes = s.getvalue().encode()
        f.write(content_bytes)
