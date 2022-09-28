#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys

from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, Optional

import yaml

from aac_datasets.datasets.audiocaps import AudioCaps
from aac_datasets.datasets.clotho import Clotho, CLOTHO_LAST_VERSION
from aac_datasets.datasets.macs import MACS


logger = logging.getLogger(__name__)


def download_audiocaps(
    root: str = ".",
    verbose: int = 1,
    force: bool = False,
    download: bool = True,
    ffmpeg: str = "ffmpeg",
    youtube_dl: str = "youtube-dl",
    with_tags: bool = False,
    subsets: Iterable[str] = AudioCaps.SUBSETS,
) -> Dict[str, AudioCaps]:
    """Download :class:`~aac_datasets.datasets.audiocaps.AudioCaps` dataset subsets."""
    AudioCaps.FORCE_PREPARE_DATA = force
    AudioCaps.FFMPEG_PATH = ffmpeg
    AudioCaps.YOUTUBE_DL_PATH = youtube_dl

    datasets = {}
    for subset in subsets:
        datasets[subset] = AudioCaps(
            root, subset, download=download, verbose=verbose, with_tags=with_tags
        )
    return datasets


def download_clotho(
    root: str = ".",
    verbose: int = 1,
    force: bool = False,
    download: bool = True,
    version: str = "v2.1",
    clean_archives: bool = False,
    subsets: Optional[Iterable[str]] = None,
) -> Dict[str, Clotho]:
    """Download :class:`~aac_datasets.datasets.clotho.Clotho` dataset subsets."""
    if subsets is None:
        subsets = Clotho.SUBSETS_DICT[version]
    Clotho.FORCE_PREPARE_DATA = force
    Clotho.CLEAN_ARCHIVES = clean_archives

    datasets = {}
    for subset in subsets:
        datasets[subset] = Clotho(
            root, subset, download=download, verbose=verbose, version=version
        )
    return datasets


def download_macs(
    root: str = ".",
    verbose: int = 1,
    force: bool = False,
    download: bool = True,
    clean_archives: bool = False,
) -> Dict[str, MACS]:
    """Download :class:`~aac_datasets.datasets.macs.MACS` dataset."""
    MACS.FORCE_PREPARE_DATA = force
    MACS.CLEAN_ARCHIVES = clean_archives

    datasets = {}
    for subset in MACS.SUBSETS:
        datasets[subset] = MACS(root, download=download, verbose=verbose)
    return datasets


def _to_bool(s: str) -> bool:
    s = s.lower()
    if s in ("true",):
        return True
    elif s in ("false",):
        return False
    else:
        raise ValueError(f"Invalid argument value {s}. (not a boolean)")


def _get_main_download_args() -> Namespace:
    parser = ArgumentParser(
        description="Download a dataset at specified root directory.",
    )

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
        "--force",
        type=_to_bool,
        default=False,
        choices=(False, True),
        help="Force download of files, even if they are already downloaded.",
    )

    subparsers = parser.add_subparsers(
        dest="dataset",
        required=True,
        description="The dataset to download.",
    )

    audiocaps_subparser = subparsers.add_parser("audiocaps")
    audiocaps_subparser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="Path to ffmpeg used to download audio from youtube.",
    )
    audiocaps_subparser.add_argument(
        "--youtube_dl",
        type=str,
        default="youtube-dl",
        help="Path to youtube-dl used to extract metadata from a youtube video.",
    )
    audiocaps_subparser.add_argument(
        "--with_tags",
        type=_to_bool,
        default=True,
        choices=(False, True),
        help="Download additional audioset tags corresponding to audiocaps audio.",
    )
    audiocaps_subparser.add_argument(
        "--subsets",
        type=str,
        default=AudioCaps.SUBSETS,
        nargs="+",
        choices=AudioCaps.SUBSETS,
        help="AudioCaps subsets to download.",
    )

    clotho_subparser = subparsers.add_parser("clotho")
    clotho_subparser.add_argument(
        "--version",
        type=str,
        default=CLOTHO_LAST_VERSION,
        choices=Clotho.VERSIONS,
        help="The version of the Clotho dataset.",
    )
    clotho_subparser.add_argument(
        "--clean_archives",
        type=_to_bool,
        default=False,
        choices=(False, True),
        help="Remove archives files after extraction.",
    )
    clotho_subparser.add_argument(
        "--subsets",
        type=str,
        default=Clotho.SUBSETS,
        nargs="+",
        choices=Clotho.SUBSETS,
        help="Clotho subsets to download. Available subsets depends of the Clotho version used.",
    )

    macs_subparser = subparsers.add_parser("macs")
    macs_subparser.add_argument(
        "--clean_archives",
        type=_to_bool,
        default=False,
        choices=(False, True),
        help="Remove archives files after extraction.",
    )
    # Note : MACS only have 1 subset, so we do not add MACS subsets arg

    args = parser.parse_args()
    return args


def _main_download() -> None:
    format_ = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(format_))
    pkg_logger = logging.getLogger("aac_datasets")
    pkg_logger.setLevel(logging.DEBUG)
    pkg_logger.addHandler(handler)

    args = _get_main_download_args()

    if args.verbose >= 2:
        logger.debug(yaml.dump({"Arguments": args.__dict__}, sort_keys=False))

    if args.dataset == "audiocaps":
        download_audiocaps(
            root=args.root,
            verbose=args.verbose,
            force=args.force,
            download=True,
            ffmpeg=args.ffmpeg,
            youtube_dl=args.youtube_dl,
            with_tags=args.with_tags,
            subsets=args.subsets,
        )

    elif args.dataset == "clotho":
        download_clotho(
            root=args.root,
            verbose=args.verbose,
            force=args.force,
            download=True,
            version=args.version,
            clean_archives=args.clean_archives,
            subsets=args.subsets,
        )

    elif args.dataset == "macs":
        download_macs(
            root=args.root,
            verbose=args.verbose,
            force=args.force,
            download=True,
            clean_archives=args.clean_archives,
        )

    else:
        DATASETS = ("audiocaps", "clotho" or "macs")
        raise ValueError(
            f"Invalid argument {args.dataset}. (expected one of {DATASETS})"
        )


if __name__ == "__main__":
    _main_download()
