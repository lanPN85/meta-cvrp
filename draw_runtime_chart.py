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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", required=True)

    args = parser.parse_args()

    with open(args.i, "rt") as f:
        summaries = yaml.full_load(f)["summary"]
