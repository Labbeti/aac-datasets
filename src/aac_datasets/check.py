#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys

from argparse import ArgumentParser, Namespace
from typing import Dict

import yaml

from aac_datasets.datasets.audiocaps import AudioCaps
from aac_datasets.datasets.clotho import Clotho
from aac_datasets.datasets.macs import MACS


def get_main_check_args() -> Namespace:
    parser = ArgumentParser(description="Check datasets in specified directory.")

    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args()
    return args


def check_datasets(root: str, verbose: int = 0) -> Dict[str, Dict[str, int]]:
    datasets_lens = {"audiocaps": {}, "clotho": {}, "macs": {}}

    if verbose >= 0:
        print(f"Searching datasets in root='{root}'...")

    for subset in AudioCaps.SUBSETS:
        try:
            dataset = AudioCaps(root, subset, verbose=verbose)
            datasets_lens["audiocaps"][subset] = len(dataset)
        except RuntimeError:
            pass

    for subset in Clotho.SUBSETS:
        try:
            dataset = Clotho(root, subset, verbose=verbose)
            datasets_lens["clotho"][subset] = len(dataset)
        except RuntimeError:
            pass

    try:
        dataset = MACS(root, verbose=verbose)
        datasets_lens["macs"]["full"] = len(dataset)
    except RuntimeError:
        pass

    datasets_lens = {
        dataset_name: subsets_lens
        for dataset_name, subsets_lens in datasets_lens.items()
        if len(subsets_lens) > 0
    }
    return datasets_lens


def main_check() -> None:
    logger = logging.getLogger("aac_datasets")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    args = get_main_check_args()

    datasets_lens = check_datasets(args.root, args.verbose)

    if args.verbose >= 0:
        print(f"Found {len(datasets_lens)} dataset(s) in root='{args.root}'.")
        if len(datasets_lens) > 0:
            print(yaml.dump(datasets_lens, sort_keys=False))


if __name__ == "__main__":
    main_check()
