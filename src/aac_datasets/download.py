#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys

from argparse import ArgumentParser, Namespace

from aac_datasets.datasets.audiocaps import AudioCaps
from aac_datasets.datasets.clotho import Clotho
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

    clotho_subparser = subparsers.add_parser("clotho")
    clotho_subparser.add_argument(
        "--version", type=str, default="v2.1", choices=["v1", "v2", "v2.1"]
    )
    clotho_subparser.add_argument(
        "--clean_archives", type=to_bool, default=True, choices=(False, True)
    )

    macs_subparser = subparsers.add_parser("macs")
    macs_subparser.add_argument(
        "--clean_archives", type=to_bool, default=True, choices=(False, True)
    )

    args = parser.parse_args()
    return args


def download_audiocaps(
    root: str = ".",
    verbose: int = 1,
    ffmpeg: str = "ffmpeg",
    youtube_dl: str = "youtube-dl",
    load_tags: bool = True,
) -> None:
    AudioCaps.FFMPEG_PATH = ffmpeg
    AudioCaps.YOUTUBE_DL_PATH = youtube_dl

    for subset in AudioCaps.SUBSETS:
        _ = AudioCaps(root, subset, download=True, verbose=verbose, load_tags=load_tags)


def download_clotho(
    root: str = ".",
    verbose: int = 1,
    version: str = "v2.1",
    clean_archives: bool = True,
) -> None:
    Clotho.CLEAN_ARCHIVES = clean_archives
    for subset in Clotho.SUBSETS_DICT[version]:
        _ = Clotho(root, subset, download=True, verbose=verbose, version=version)


def download_macs(
    root: str = ".", verbose: int = 1, clean_archives: bool = True
) -> None:
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

    if args.dataset == "audiocaps":
        download_audiocaps(
            args.root,
            args.verbose,
            args.ffmpeg,
            args.youtube_dl,
            args.load_tags,
        )

    elif args.dataset == "clotho":
        download_clotho(args.root, args.verbose, args.version, args.clean_archives)

    elif args.dataset == "macs":
        download_macs(args.root, args.verbose, args.clean_archives)

    else:
        raise ValueError(
            f"Invalid argument {args.dataset}. (expected 'audiocaps', 'clotho' or 'macs')"
        )


if __name__ == "__main__":
    main_download()
