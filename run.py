#!/usr/bin/env python3

import argparse
import os
import sys

from loguru import logger
from typing import List
from omegaconf import OmegaConf
from contexttimer import Timer

from cvrp.utils import load_class_from_config, load_config
from cvrp.solver.base import ISolver
from cvrp.dataset.base import IDataset
from cvrp.serialize.xml import XmlSolutionSerializer
from cvrp.serialize.yml import YamlSolutionSummarizer
from cvrp.model.solution import SolutionValidity


def run(config) -> bool:
    result_dir = os.path.join("results", config.run_name)
    os.makedirs(result_dir, exist_ok=True)
    version = get_next_version(result_dir)
    result_dir = os.path.join(result_dir, f"version_{version}")
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Saving results to {result_dir}")

    logfile = os.path.join(result_dir, "run.log")
    logger.add(logfile, level="INFO")

    target_config_file = os.path.join(result_dir, "config.yml")
    OmegaConf.save(config, target_config_file, resolve=True)

    logger.info("Loading dataset")
    dataset: IDataset = load_class_from_config(config.dataset)

    logger.info("Loading solver")
    solver: ISolver = load_class_from_config(config.solver, logdir=result_dir)

    xml_serializer = XmlSolutionSerializer()
    summarizer = YamlSolutionSummarizer()

    logger.info(f"Solving {len(dataset)} instances")
    instances, solutions = [], []

    is_interrupted = False

    try:
        for instance in dataset:  # type: ignore
            logger.info(instance.name)

            with Timer(factor=1000, output=None) as timer:
                solution = solver.solve(instance)
            if solution is None:
                logger.warning(
                    f"No valid solution is found for instance {instance.name}"
                )
                continue

            solution.meta.run_time_ms = timer.elapsed  # type: ignore
            validity = solution.validate_(instance)

            instances.append(instance)
            solutions.append(solution)

            if validity != SolutionValidity.VALID:
                logger.error(
                    f"Solution for instance {instance.name} is not valid ({validity.name})"
                )
                if config.exit_on_invalid:
                    exit(validity.value)

            logger.info(f"Saving solution")
            xml_save_path = os.path.join(result_dir, f"{instance.name}.solution.xml")
            with open(xml_save_path, "wb") as f:
                xml_serializer.save_solution(solution, instance, f)  # type: ignore
    except KeyboardInterrupt:
        is_interrupted = True
        logger.error("Run interrupted")
        print()
        inp = input("Run summary on completed instances?(y/N) ")
        if inp.lower() != "y":
            return is_interrupted

    logger.info("Summarizing")
    summary_path = os.path.join(result_dir, "summary.yml")
    with open(summary_path, "wb") as f:
        summarizer.summarize_solutions(solutions, instances, f)  # type: ignore
    return is_interrupted


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", action="append", default=[])
    parser.add_argument(
        "-t", type=int, default=1, help="Number of times to run. Defaults to 1"
    )

    args = parser.parse_args()

    logger.info("Loading config")
    config = load_config(args.config)

    for i in range(args.t):
        logger.info(f"Run {i}")
        interrupt = run(config)
        logger.remove()
        logger.add(sys.stderr)

        if interrupt:
            inp = input("Abort next runs? (Y/n) ")
            if inp.lower() != "n":
                break


def get_next_version(dir: str) -> int:
    version = 0
    while True:
        subdir = f"version_{version}"
        subdir_path = os.path.join(dir, subdir)
        if not os.path.exists(subdir_path):
            return version
        version += 1


if __name__ == "__main__":
    main()
