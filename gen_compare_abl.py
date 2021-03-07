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

    ls_paths = [
        f"results/LS_{args.dataset}",
        f"results/LS12_{args.dataset}",
        f"results/LS13_{args.dataset}",
        f"results/LS23_{args.dataset}",
    ]
    ls_method_names = [
        "SimpleTabu (Full)",
        "SimpleTabu (No Or-Opt)",
        "SimpleTabu (No 2-Opt*)",
        "SimpleTabu (No Relocate)",
    ]

    ls_dir_list = [
        [os.path.join(p, f"version_{i}") for i in range(5)] for p in ls_paths
    ]

    ls_dfs = [
        get_runs_dataframe(ls_dirs, name)
        for ls_dirs, name in zip(ls_dir_list, ls_method_names)
    ]

    ls_best_dfs = [
        ls_df.pivot_table(
            values="fitness", index="name", columns="method", aggfunc="min"
        )
        for ls_df in ls_dfs
    ]
    ls_stdev_dfs = [
        ls_df.pivot_table(
            values="fitness", index="name", columns="method", aggfunc="std"
        )
        for ls_df in ls_dfs
    ]

    fitness_df = ls_best_dfs[0]
    for ls_df in ls_best_dfs[1:]:
        fitness_df = pd.merge(fitness_df, ls_df, on="name")

    std_df = ls_stdev_dfs[0]
    for ls_df in ls_stdev_dfs[1:]:
        std_df = pd.merge(std_df, ls_df, on="name")

    # Generate the LaTex table
    with clt.table(
        "|c|r|r|r|r|r|r|r|r|", caption=args.caption, label=args.label, scale=True
    ):
        clt.hline()

        headers = [
            f"\\multicolumn{{1}}{{|c|}}{{ \\multirow{{2}}{{*}}{{ \\textbf{{ Instance }} }} }}"
        ]
        for method in ls_method_names:
            headers.append(f"\\multicolumn{{2}}{{|c|}}{{ \\textbf{{ {method} }} }}")
        clt.table_row(headers, hline=False)
        print(f"\\cline{{ 2-{len(ls_method_names) * 2 + 1} }}")

        subheaders = [""]
        for method in ls_method_names:
            subheaders.append(f"\\multicolumn{{1}}{{|c|}}{{ Best }}")
            subheaders.append(f"\\multicolumn{{1}}{{|c|}}{{ Std }}")
        clt.table_row(subheaders)

        for (name, best_row), (name2, std_row) in zip(
            fitness_df.iterrows(), std_df.iterrows()
        ):
            assert name == name2
            best_fitness = best_row.min()
            best_std = std_row.min()
            row_ = [name]
            for mt in ls_method_names:
                fitness = best_row[mt]
                std = std_row[mt]
                s1 = (
                    f"\\textbf{{ {fitness:.2f} }}"
                    if fitness == best_fitness
                    else f" {fitness:.2f}"
                )
                s2 = f"\\textbf{{ {std:.2f} }}" if std == best_std else f"{std:.2f}"
                row_ += [s1, s2]
            clt.table_row(row_)


if __name__ == "__main__":
    main()
