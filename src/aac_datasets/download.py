#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys

from argparse import ArgumentParser, Namespace
from typing import Iterable

import yaml

from aac_datasets.datasets.audiocaps import AudioCaps
from aac_datasets.datasets.clotho import Clotho, CLOTHO_LAST_VERSION
from aac_datasets.datasets.macs import MACS


def to_bool(s: str) -> bool:
    s = s.lower()
    if s in ("true",):
        return True
    elif s in ("false",):
        return False
    else:
        raise ValueError(f"Invalid argument value {s}. (not a boolean)")


def get_main_download_args() -> Namespace:
    parser = ArgumentParser(
        description="Download a dataset at specified root directory."
    )

    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--force", type=to_bool, default=False, choices=(False, True))

    subparsers = parser.add_subparsers(dest="dataset", required=True)

    audiocaps_subparser = subparsers.add_parser("audiocaps")
    audiocaps_subparser.add_argument("--ffmpeg", type=str, default="ffmpeg")
    audiocaps_subparser.add_argument("--youtube_dl", type=str, default="youtube-dl")
    audiocaps_subparser.add_argument(
        "--load_tags",
        type=to_bool,
        default=True,
        choices=(False, True),
    )
    audiocaps_subparser.add_argument(
        "--subsets",
        type=str,
        default=AudioCaps.SUBSETS,
        nargs="+",
        choices=AudioCaps.SUBSETS,
    )

    clotho_subparser = subparsers.add_parser("clotho")
    clotho_subparser.add_argument(
        "--version", type=str, default=CLOTHO_LAST_VERSION, choices=Clotho.VERSIONS
    )
    clotho_subparser.add_argument(
        "--clean_archives", type=to_bool, default=False, choices=(False, True)
    )
    clotho_subparser.add_argument(
        "--subsets",
        type=str,
        default=Clotho.SUBSETS,
        nargs="+",
        choices=Clotho.SUBSETS,
    )

    macs_subparser = subparsers.add_parser("macs")
    macs_subparser.add_argument(
        "--clean_archives", type=to_bool, default=False, choices=(False, True)
    )

    args = parser.parse_args()
    return args


def download_audiocaps(
    root: str = ".",
    verbose: int = 1,
    force: bool = False,
    ffmpeg: str = "ffmpeg",
    youtube_dl: str = "youtube-dl",
    load_tags: bool = False,
    subsets: Iterable[str] = AudioCaps.SUBSETS,
) -> None:
    AudioCaps.FORCE_PREPARE_DATA = force
    AudioCaps.FFMPEG_PATH = ffmpeg
    AudioCaps.YOUTUBE_DL_PATH = youtube_dl

    for subset in subsets:
        _ = AudioCaps(root, subset, download=True, verbose=verbose, load_tags=load_tags)


def download_clotho(
    root: str = ".",
    verbose: int = 1,
    force: bool = False,
    version: str = "v2.1",
    clean_archives: bool = False,
    subsets: Iterable[str] = Clotho.SUBSETS,
) -> None:
    Clotho.FORCE_PREPARE_DATA = force
    Clotho.CLEAN_ARCHIVES = clean_archives
    for subset in subsets:
        _ = Clotho(root, subset, download=True, verbose=verbose, version=version)


def download_macs(
    root: str = ".",
    verbose: int = 1,
    force: bool = False,
    clean_archives: bool = False,
) -> None:
    MACS.FORCE_PREPARE_DATA = force
    MACS.CLEAN_ARCHIVES = clean_archives
    _ = MACS(root, download=True, verbose=verbose)


def main_download() -> None:
    format_ = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(format_))
    logger = logging.getLogger("aac_datasets")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    args = get_main_download_args()

    if args.verbose >= 2:
        print(yaml.dump({"Arguments": args.__dict__}, sort_keys=False))

    if args.dataset == "audiocaps":
        download_audiocaps(
            root=args.root,
            verbose=args.verbose,
            force=args.force,
            ffmpeg=args.ffmpeg,
            youtube_dl=args.youtube_dl,
            load_tags=args.load_tags,
            subsets=args.subsets,
        )

    elif args.dataset == "clotho":
        download_clotho(
            root=args.root,
            verbose=args.verbose,
            force=args.force,
            version=args.version,
            clean_archives=args.clean_archives,
            subsets=args.subsets,
        )

    elif args.dataset == "macs":
        download_macs(
            root=args.root,
            verbose=args.verbose,
            force=args.force,
            clean_archives=args.clean_archives,
        )

    else:
        DATASETS = ("audiocaps", "clotho" or "macs")
        raise ValueError(
            f"Invalid argument {args.dataset}. (expected one of {DATASETS})"
        )


if __name__ == "__main__":
    main_download()
