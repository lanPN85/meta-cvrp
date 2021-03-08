#!/usr/bin/env python3

import argparse
import os
import yaml
import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from loguru import logger
from pandas.core.frame import DataFrame

from cvrp.utils import get_runs_dataframe


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", choices=["cmt", "golden"], default="cmt")
    parser.add_argument("-o", required=True)

    args = parser.parse_args()

    result_dir = os.path.join("results", f"LS_{args.dataset}")
    version_dirs = [os.path.join(result_dir, f"version_{i}") for i in range(5)]

    runs_df = get_runs_dataframe(version_dirs, "SimpleTabu")
    runtime_df = runs_df.pivot_table(
        values="runtime", columns=["name"], aggfunc="mean"
    ).apply(lambda x: x / 1000)

    plt.rcParams["figure.figsize"] = (10.5, 4.5)

    sns.barplot(data=runtime_df)
    plt.xlabel("Instance")
    plt.ylabel("Running time (s)")
    plt.minorticks_on()
    plt.savefig(args.o, dpi=800)


if __name__ == "__main__":
    main()
