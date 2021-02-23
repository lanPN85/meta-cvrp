#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
import csv

from loguru import logger

from cvrp.utils import create_summary_index, group_summaries_by_name


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "-i1", required=True, help="Path to the Clarke-Wright result folder"
    )
    parser.add_argument(
        "-i2", required=True, help="Path to the local search result folder"
    )
    parser.add_argument("-v1", default=0, type=int, help="CW version")
    parser.add_argument(
        "-v2",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Local search versions, separated by commas",
    )
    parser.add_argument("-m", "--metric", choices=["cost", "runtime"], default="cost")

    args = parser.parse_args()

    # Calculate metrics for CW
    logger.info("Calculating CW metrics")
    cw_result_dir = os.path.join(args.i1, f"version_{args.v1}")
    cw_summary_path = os.path.join(cw_result_dir, "summary.yml")
    with open(cw_summary_path, "rt") as f:
        cw_summary = yaml.full_load(f)

    cw_summary_index = create_summary_index(cw_summary["summary"])
    cw_metrics = {}
    for name, inst in cw_summary_index.items():
        total_cost = inst["total_cost"]
        cw_metrics[name] = {
            "total_cost": inst["total_cost"],
            "runtime": inst["runtime_ms"],
        }

    # Calculate metrics for local search
    logger.info("Calculating LS metrics")
    ls_versions = args.v2.split(",")
    ls_summary_list = []
    for version in ls_versions:
        ls_result_dir = os.path.join(args.i2, f"version_{int(version)}")
        ls_summary_path = os.path.join(ls_result_dir, "summary.yml")
        with open(ls_summary_path, "rt") as f:
            ls_summary = yaml.full_load(f)
        ls_summary_list.append(ls_summary["summary"])

    ls_groups = group_summaries_by_name(ls_summary_list)
    ls_metrics = {}
    for name, instances in ls_groups.items():
        sum_total_cost = sum([x["total_cost"] for x in instances])
        avg_total_cost = sum_total_cost / len(instances)
        sum_runtime = sum([x["runtime_ms"] for x in instances])
        avg_runtime = sum_runtime / len(instances)
        ls_metrics[name] = {"total_cost": avg_total_cost, "runtime": avg_runtime}

    instance_names = [k for k in cw_summary_index.keys()]
    instance_names.sort()

    logger.info("Writing file")
    HEADERS = ("Instance", "Clarke-Wright", "Tabu search")
    with open(args.output, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        for name in instance_names:
            key_name = ""
            if args.metric == "cost":
                key_name = "total_cost"
                cw_metric = f"{cw_metrics[name][key_name]:.2f}"
                ls_metric = f"{ls_metrics[name][key_name]:.2f}"
            elif args.metric == "runtime":
                key_name = "runtime"
                cw_metric = f"{cw_metrics[name][key_name] / 1000:.2f}s"
                ls_metric = f"{ls_metrics[name][key_name] / 1000:.2f}s"

            writer.writerow([name, cw_metric, ls_metric])


if __name__ == "__main__":
    main()
