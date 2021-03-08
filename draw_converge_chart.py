#!/usr/bin/env python3

import argparse
import os
import yaml
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from loguru import logger
from pandas.core.frame import DataFrame


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", required=True)
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("-r", "--run", type=int, required=True)
    parser.add_argument("-o", required=True)

    args = parser.parse_args()

    with open(args.i, "rt") as f:
        summaries = yaml.full_load(f)["summary"]
        instance_data = next(
            filter(lambda x: x["instance_name"] == args.name, summaries)
        )
        run_data = instance_data["extras"]["runs"][args.run]

    divers_fitness = run_data["divers_fitness"]
    intens_fitness = run_data["intens_fitness"]
    short_term_fitness = run_data["short_term_fitness"]

    total_length = len(divers_fitness) + len(intens_fitness) + len(short_term_fitness)
    X = list(range(1, total_length + 1))

    p_short_term_fitness = short_term_fitness + [None] * (
        len(intens_fitness) + len(divers_fitness)
    )

    p_intens_fitness = (
        [None] * len(short_term_fitness) + intens_fitness + [None] * len(divers_fitness)
    )
    p_intens_fitness[len(short_term_fitness) - 1] = short_term_fitness[-1]

    p_divers_fitness = [None] * (
        len(short_term_fitness) + len(intens_fitness)
    ) + divers_fitness
    p_divers_fitness[
        len(short_term_fitness) + len(intens_fitness) - 1
    ] = intens_fitness[-1]

    fitness_df = DataFrame(
        {
            "Short-term phase": p_short_term_fitness,
            "Intensification phase": p_intens_fitness,
            "Diversification phase": p_divers_fitness,
        },
        index=X,
    )

    sns.lineplot(data=fitness_df)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.savefig(args.o, dpi=800)


if __name__ == "__main__":
    main()
