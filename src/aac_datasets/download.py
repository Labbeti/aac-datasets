#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace

from aac_datasets.datasets.audiocaps import AudioCaps
from aac_datasets.datasets.clotho import Clotho
from aac_datasets.datasets.macs import MACS


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


def download_clotho(root: str = ".", verbose: int = 1, version: str = "v2.1") -> None:
    for subset in Clotho.SUBSETS_DICT[version]:
        _ = Clotho(root, subset, download=True, verbose=verbose, version=version)


def download_macs(root: str = ".", verbose: int = 1) -> None:
    _ = MACS(root, download=True, verbose=verbose)


def get_main_download_args() -> Namespace:
    parser = ArgumentParser(
        description="Download a dataset at specified root directory."
    )

    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--verbose", type=int, default=1)

    subparsers = parser.add_subparsers(dest="dataset")

    audiocaps_sp = subparsers.add_parser("audiocaps")
    audiocaps_sp.add_argument("--ffmpeg", type=str, default="ffmpeg")
    audiocaps_sp.add_argument("--youtube_dl", type=str, default="youtube-dl")
    audiocaps_sp.add_argument(
        "--load_tags",
        type=lambda s: s == "true",
        default="true",
        choices=["false", "true"],
    )

    clotho_sp = subparsers.add_parser("clotho")
    clotho_sp.add_argument(
        "--version", type=str, default="v2.1", choices=["v1", "v2", "v2.1"]
    )

    _ = subparsers.add_parser("macs")

    args = parser.parse_args()
    return args


def main_download() -> None:
    args = get_main_download_args()

    if args.dataset == "audiocaps":
        download_audiocaps(
            args.root, args.verbose, args.ffmpeg, args.youtube_dl, args.load_tags
        )

    elif args.dataset == "clotho":
        download_clotho(args.root, args.verbose, args.version)

    elif args.dataset == "macs":
        download_macs(args.root, args.verbose)

    else:
        raise ValueError(
            f"Invalid argument {args.dataset}. (expected 'audiocaps', 'clotho' or 'macs')"
        )


if __name__ == "__main__":
    main_download()
