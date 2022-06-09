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


logger = logging.getLogger(__name__)


def get_main_check_args() -> Namespace:
    parser = ArgumentParser(description="Check datasets in specified directory.")

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="The path to the parent directory of the datasets.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbose level of the script. 0 means silent mode, 1 is default mode and 2 add additional debugging outputs.",
    )

    args = parser.parse_args()
    return args


def check_datasets(root: str, verbose: int = 0) -> Dict[str, Dict[str, int]]:
    datasets_lens = {"audiocaps": {}, "clotho": {}, "macs": {}}

    if verbose >= 1:
        logger.info(f"Searching for audiocaps in root='{root}'...")

    for subset in AudioCaps.SUBSETS:
        try:
            dataset = AudioCaps(root, subset, verbose=0)
            _ = dataset[0]
            datasets_lens["audiocaps"][subset] = len(dataset)
        except RuntimeError:
            if verbose >= 2:
                logger.info(f"Cannot find audiocaps_{subset}.")

    if verbose >= 1:
        logger.info(f"Searching for clotho in root='{root}'...")

    for subset in Clotho.SUBSETS:
        try:
            dataset = Clotho(root, subset, verbose=0)
            _ = dataset[0]
            datasets_lens["clotho"][subset] = len(dataset)
        except RuntimeError:
            if verbose >= 2:
                logger.info(f"Cannot find clotho_{subset}.")

    if verbose >= 1:
        logger.info(f"Searching for macs in root='{root}'...")

    try:
        dataset = MACS(root, verbose=0)
        _ = dataset[0]
        datasets_lens["macs"]["full"] = len(dataset)
    except RuntimeError:
        if verbose >= 2:
            logger.info("Cannot find macs.")

    datasets_lens = {
        dataset_name: subsets_lens
        for dataset_name, subsets_lens in datasets_lens.items()
        if len(subsets_lens) > 0
    }
    return datasets_lens


def main_check() -> None:
    format_ = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(format_))
    logger = logging.getLogger("aac_datasets")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    args = get_main_check_args()

    if args.verbose >= 2:
        print(yaml.dump({"Arguments": args.__dict__}, sort_keys=False))

    datasets_lens = check_datasets(args.root, args.verbose)

    if args.verbose >= 1:
        print(f"Found {len(datasets_lens)} dataset(s) in root='{args.root}'.")
        if len(datasets_lens) > 0:
            print(yaml.dump(datasets_lens, sort_keys=False))


if __name__ == "__main__":
    main_check()
