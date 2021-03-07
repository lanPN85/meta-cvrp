#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd

from loguru import logger

from cvrp.utils import get_runs_dataframe, parse_result_csv, latex as clt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset to generate table",
        choices=["cmt", "golden"],
        default="cmt",
    )
    parser.add_argument("-c", "--caption", default=None)
    parser.add_argument("-l", "--label", default=None)

    args = parser.parse_args()

    other_csv_path = os.path.join("results/others", f"{args.dataset}.csv")
    others_df = parse_result_csv(other_csv_path)

    ls_path = f"results/LS_{args.dataset}"
    ls_dirs = [os.path.join(ls_path, f"version_{i}") for i in range(5)]
    ls_df = get_runs_dataframe(ls_dirs, "SimpleTabu")
    ls_best_table = ls_df.pivot_table(
        values="fitness", index="name", columns="method", aggfunc="min"
    )

    others_best_table = others_df.pivot_table(
        values="fitness", index="name", columns="method", aggfunc="first"
    )
    fitness_df = pd.merge(ls_best_table, others_best_table, on="name")
    methods = list(fitness_df.columns.values)

    column_str = "|c|"
    for m in methods:
        column_str += "r|"

    # Generate the LaTex table
    with clt.table(column_str, caption=args.caption, label=args.label):
        clt.hline()
        clt.table_headers(["Instance"] + methods)

        for name, row in fitness_df.iterrows():
            best_fitness = row.min()
            row_ = [name]
            for mt in methods:
                fn = row[mt]
                s = ""
                if fn == best_fitness:
                    s = f"\\textbf{{ {fn:.2f} }}"
                else:
                    s = f"{fn:.2f}"
                row_ += [s]
            clt.table_row(row_)


if __name__ == "__main__":
    main()
