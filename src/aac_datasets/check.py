#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys

from argparse import ArgumentParser, Namespace
from typing import Iterable, List

import yaml

from aac_datasets.datasets.audiocaps import AudioCaps
from aac_datasets.datasets.clotho import Clotho
from aac_datasets.datasets.macs import MACS


logger = logging.getLogger(__name__)


def check_directory(
    root: str,
    verbose: int = 0,
    datasets: Iterable[str] = ("audiocaps", "clotho", "macs"),
) -> List[str]:
    """Check which datasets are installed in root.

    :param root: The directory to check.
    :param verbose: The verbose level. defaults to 0.
    :param datasets: The datasets to search in root. defaults to ("audiocaps", "clotho", "macs").
    :returns: A list of dataset and subsets names found in root directory.
    """
    valid_subsets = []

    dataname = "audiocaps"
    if dataname in datasets:
        if verbose >= 1:
            logger.info(f"Searching for {dataname} in root='{root}'...")

        for subset in AudioCaps.SUBSETS:
            try:
                _ = AudioCaps(root, subset, verbose=0)
                valid_subsets.append(f"{dataname}_{subset}")

            except RuntimeError:
                if verbose >= 2:
                    logger.info(f"Cannot find {dataname}_{subset}.")

    dataname = "clotho"
    if dataname in datasets:
        if verbose >= 1:
            logger.info(f"Searching for {dataname} in root='{root}'...")

        for subset in Clotho.SUBSETS:
            try:
                _ = Clotho(root, subset, verbose=0)
                valid_subsets.append(f"{dataname}_{subset}")

            except RuntimeError:
                if verbose >= 2:
                    logger.info(f"Cannot find {dataname}_{subset}.")

    dataname = "macs"
    if dataname in datasets:
        if verbose >= 1:
            logger.info(f"Searching for {dataname} in root='{root}'...")

        for subset in MACS.SUBSETS:
            try:
                _ = MACS(root, subset, verbose=0)
                valid_subsets.append(f"{dataname}_{subset}")
            except RuntimeError:
                if verbose >= 2:
                    logger.info(f"Cannot find {dataname}_{subset}.")

    return valid_subsets


def _get_main_check_args() -> Namespace:
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
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=("audiocaps", "clotho", "macs"),
        help="The datasets to check in root directory.",
    )

    args = parser.parse_args()
    return args


def _main_check() -> None:
    format_ = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(format_))
    pkg_logger = logging.getLogger("aac_datasets")
    pkg_logger.setLevel(logging.DEBUG)
    pkg_logger.addHandler(handler)

    args = _get_main_check_args()

    if args.verbose >= 2:
        logger.debug(yaml.dump({"Arguments": args.__dict__}, sort_keys=False))

    valid_subsets = check_directory(args.root, args.verbose, args.datasets)

    if args.verbose >= 1:
        print(f"Found {len(valid_subsets)} subset(s) in root='{args.root}'.")
        if len(valid_subsets) > 0:
            print(yaml.dump(valid_subsets, sort_keys=False))


if __name__ == "__main__":
    _main_check()
