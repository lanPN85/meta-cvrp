import pandas as pd
import yaml
import os
import csv

from typing import Any, Dict, List


def create_summary_index(instances: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = {}

    for inst in instances:
        index[inst["instance_name"]] = inst

    return index


def group_summaries_by_name(
    instance_list: List[List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}

    for instances in instance_list:
        for inst in instances:
            instance_name = inst["instance_name"]
            if instance_name not in groups.keys():
                groups[instance_name] = []
            groups[instance_name].append(inst)

    return groups


def get_runs_dataframe(dirs: List[str], method: str) -> pd.DataFrame:
    fitness, runtime, names = [], [], []
    methods = []

    for d in dirs:
        summary_path = os.path.join(d, "summary.yml")
        with open(summary_path, "rt") as f:
            summary = yaml.full_load(f)
            for inst in summary["summary"]:
                fitness.append(inst["total_cost"])
                runtime.append(inst["runtime_ms"])
                names.append(inst["instance_name"])
                methods.append(method)

    return pd.DataFrame(
        {"fitness": fitness, "runtime": runtime, "name": names, "method": methods}
    )


def parse_result_csv(path: str) -> pd.DataFrame:
    fitness, names = [], []
    methods = []

    with open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        h_methods = header[1:]

        for row in reader:
            for i, hm in enumerate(h_methods):
                names.append(row[0])
                fitness.append(float(row[i + 1]))
                methods.append(hm)

    return pd.DataFrame({"fitness": fitness, "name": names, "method": methods})
