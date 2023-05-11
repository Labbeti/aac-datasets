#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp
import sys

from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable

import yaml

from aac_datasets.datasets.audiocaps import AudioCaps
from aac_datasets.datasets.clotho import Clotho
from aac_datasets.datasets.macs import MACS


pylog = logging.getLogger(__name__)


def check_directory(
    root: str,
    verbose: int = 0,
    datasets: Iterable[str] = ("audiocaps", "clotho", "macs"),
) -> Dict[str, Dict[str, int]]:
    """Check which datasets are installed in root.

    :param root: The directory to check.
    :param verbose: The verbose level. defaults to 0.
    :param datasets: The datasets to search in root directory. defaults to ("audiocaps", "clotho", "macs").
    :returns: A dictionary of datanames containing the length of each subset.
    """
    data_infos = [
        ("audiocaps", AudioCaps),
        ("clotho", Clotho),
        ("macs", MACS),
    ]
    data_infos = [
        (ds_name, class_) for ds_name, class_ in data_infos if ds_name in datasets
    ]

    if verbose >= 1:
        pylog.info(f"Start searching datasets in root='{root}'.")

    all_found_dsets = {}
    for ds_name, ds_class in data_infos:
        if verbose >= 1:
            pylog.info(f"Searching for {ds_name}...")

        found_dsets = {}
        for subset in ds_class.SUBSETS:
            try:
                ds = ds_class(root, subset, verbose=0)
                found_dsets[subset] = ds

            except RuntimeError:
                if verbose >= 2:
                    pylog.info(f"Cannot find {ds_name}_{subset}.")

        if len(found_dsets) > 0:
            all_found_dsets[ds_name] = found_dsets

    if verbose >= 1:
        pylog.info(
            f"Checking if audio files exists for {len(all_found_dsets)} datasets..."
        )

    for ds_name, dsets in all_found_dsets.items():
        for subset, ds in dsets.items():
            fpaths = ds[:, "fpath"]
            is_valid = [osp.isfile(fpath) for fpath in fpaths]
            if not all(is_valid):
                pylog.error(f"Cannot find all audio files for {ds_name}.{subset}.")
            else:
                pylog.info(f"Dataset {ds_name}.{subset} is valid.")

    all_valids_lens = {
        ds_name: {subset: len(ds) for subset, ds in dsets.items()}
        for ds_name, dsets in all_found_dsets.items()
    }
    return all_valids_lens


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
        pylog.debug(yaml.dump({"Arguments": args.__dict__}, sort_keys=False))

    valid_datasubsets = check_directory(args.root, args.verbose, args.datasets)

    if args.verbose >= 1:
        print(
            f"Found {len(valid_datasubsets)}/{len(args.datasets)} dataset(s) in root='{args.root}':"
        )
        if len(valid_datasubsets) > 0:
            print(yaml.dump(valid_datasubsets, sort_keys=False))


if __name__ == "__main__":
    _main_check()
