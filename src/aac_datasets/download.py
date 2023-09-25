#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys

from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, Optional

import yaml

from aac_datasets.datasets.audiocaps import AudioCaps, AudioCapsCard
from aac_datasets.datasets.clotho import Clotho, ClothoCard
from aac_datasets.datasets.macs import MACS, MACSCard
from aac_datasets.datasets.wavcaps import WavCaps, WavCapsCard, HUGGINGFACE_HUB_CACHE
from aac_datasets.utils.paths import (
    get_default_root,
    get_default_ffmpeg_path,
    get_default_ytdl_path,
)


pylog = logging.getLogger(__name__)


_TRUE_VALUES = ("true", "1", "t")
_FALSE_VALUES = ("false", "0", "f")


def download_audiocaps(
    root: str = ...,
    verbose: int = 1,
    force: bool = False,
    download: bool = True,
    ffmpeg_path: str = ...,
    ytdl_path: str = ...,
    with_tags: bool = False,
    subsets: Iterable[str] = AudioCapsCard.SUBSETS,
) -> Dict[str, AudioCaps]:
    """Download :class:`~aac_datasets.datasets.audiocaps.AudioCaps` dataset subsets."""
    AudioCaps.FORCE_PREPARE_DATA = force
    datasets = {}
    for subset in subsets:
        datasets[subset] = AudioCaps(
            root,
            subset,
            download=download,
            verbose=verbose,
            with_tags=with_tags,
            ffmpeg_path=ffmpeg_path,
            ytdl_path=ytdl_path,
        )
    return datasets


def download_clotho(
    root: str = ...,
    verbose: int = 1,
    force: bool = False,
    download: bool = True,
    version: str = ClothoCard.DEFAULT_VERSION,
    clean_archives: bool = False,
    subsets: Iterable[str] = ClothoCard.SUBSETS,
) -> Dict[str, Clotho]:
    """Download :class:`~aac_datasets.datasets.clotho.Clotho` dataset subsets."""
    subsets = list(subsets)
    if version == "v1":
        if "val" in subsets:
            if verbose >= 0:
                pylog.warning(
                    f"Excluding val subset since it did not exists for version '{version}'."
                )
            subsets = [subset for subset in subsets if subset != "val"]

    Clotho.FORCE_PREPARE_DATA = force
    Clotho.CLEAN_ARCHIVES = clean_archives

    datasets = {}
    for subset in subsets:
        datasets[subset] = Clotho(
            root, subset, download=download, verbose=verbose, version=version
        )
    return datasets


def download_macs(
    root: str = ...,
    verbose: int = 1,
    force: bool = False,
    download: bool = True,
    clean_archives: bool = False,
) -> Dict[str, MACS]:
    """Download :class:`~aac_datasets.datasets.macs.MACS` dataset."""
    MACS.FORCE_PREPARE_DATA = force
    MACS.CLEAN_ARCHIVES = clean_archives

    datasets = {}
    for subset in MACSCard.SUBSETS:
        datasets[subset] = MACS(root, download=download, verbose=verbose)
    return datasets


def download_wavcaps(
    root: str = ...,
    verbose: int = 1,
    force: bool = False,
    download: bool = True,
    clean_archives: bool = False,
    subsets: Iterable[str] = WavCapsCard.SUBSETS,
    hf_cache_dir: Optional[str] = HUGGINGFACE_HUB_CACHE,
    revision: Optional[str] = WavCapsCard.DEFAULT_REVISION,
) -> Dict[str, WavCaps]:
    """Download :class:`~aac_datasets.datasets.wavcaps.WavCaps` dataset."""

    WavCaps.FORCE_PREPARE_DATA = force
    WavCaps.CLEAN_ARCHIVES = clean_archives

    datasets = {}
    for subset in subsets:
        datasets[subset] = WavCaps(
            root,
            download=download,
            hf_cache_dir=hf_cache_dir,
            revision=revision,
            verbose=verbose,
        )
    return datasets


def _str_to_bool(s: str) -> bool:
    s = str(s).strip().lower()
    if s in _TRUE_VALUES:
        return True
    elif s in _FALSE_VALUES:
        return False
    else:
        raise ValueError(
            f"Invalid argument s={s}. (expected one of {_TRUE_VALUES + _FALSE_VALUES})"
        )


def _get_main_download_args() -> Namespace:
    parser = ArgumentParser(
        description="Download a dataset at specified root directory.",
    )

    parser.add_argument(
        "--root",
        type=str,
        default=get_default_root(),
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
        type=_str_to_bool,
        default=False,
        help="Force download of files, even if they are already downloaded.",
    )

    subparsers = parser.add_subparsers(
        dest="dataset",
        required=True,
        description="The dataset to download.",
    )

    audiocaps_subparser = subparsers.add_parser(AudioCapsCard.NAME)
    audiocaps_subparser.add_argument(
        "--ffmpeg_path",
        type=str,
        default=get_default_ffmpeg_path(),
        help="Path to ffmpeg used to download audio from youtube.",
    )
    audiocaps_subparser.add_argument(
        "--ytdl_path",
        type=str,
        default=get_default_ytdl_path(),
        help="Path to youtube-dl used to extract metadata from a youtube video.",
    )
    audiocaps_subparser.add_argument(
        "--with_tags",
        type=_str_to_bool,
        default=True,
        help="Download additional audioset tags corresponding to audiocaps audio.",
    )
    audiocaps_subparser.add_argument(
        "--subsets",
        type=str,
        default=AudioCapsCard.SUBSETS,
        nargs="+",
        choices=AudioCapsCard.SUBSETS,
        help="AudioCaps subsets to download.",
    )

    clotho_subparser = subparsers.add_parser(ClothoCard.NAME)
    clotho_subparser.add_argument(
        "--version",
        type=str,
        default=ClothoCard.DEFAULT_VERSION,
        choices=ClothoCard.VERSIONS,
        help="The version of the Clotho dataset.",
    )
    clotho_subparser.add_argument(
        "--clean_archives",
        type=_str_to_bool,
        default=False,
        help="Remove archives files after extraction.",
    )
    clotho_subparser.add_argument(
        "--subsets",
        type=str,
        default=ClothoCard.SUBSETS,
        nargs="+",
        choices=ClothoCard.SUBSETS,
        help="Clotho subsets to download. Available subsets depends of the Clotho version used.",
    )

    macs_subparser = subparsers.add_parser(MACSCard.NAME)
    macs_subparser.add_argument(
        "--clean_archives",
        type=_str_to_bool,
        default=False,
        help="Remove archives files after extraction.",
    )
    # Note : MACS only have 1 subset, so we do not add MACS subsets arg

    wavcaps_subparser = subparsers.add_parser(WavCapsCard.NAME)
    wavcaps_subparser.add_argument(
        "--clean_archives",
        type=_str_to_bool,
        default=False,
        help="Remove archives files after extraction.",
    )
    wavcaps_subparser.add_argument(
        "--subsets",
        type=str,
        default=WavCapsCard.SUBSETS,
        nargs="+",
        choices=WavCapsCard.SUBSETS,
        help="WavCaps subsets to download.",
    )
    wavcaps_subparser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=HUGGINGFACE_HUB_CACHE,
        help="Hugging face cache dir.",
    )
    wavcaps_subparser.add_argument(
        "--revision",
        type=str,
        default=WavCapsCard.DEFAULT_REVISION,
        help="Revision of the WavCaps dataset.",
    )

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
        pylog.debug(yaml.dump({"Arguments": args.__dict__}, sort_keys=False))

    if args.dataset == AudioCapsCard.NAME:
        download_audiocaps(
            root=args.root,
            verbose=args.verbose,
            force=args.force,
            download=True,
            ffmpeg_path=args.ffmpeg_path,
            ytdl_path=args.ytdl_path,
            with_tags=args.with_tags,
            subsets=args.subsets,
        )

    elif args.dataset == ClothoCard.NAME:
        download_clotho(
            root=args.root,
            verbose=args.verbose,
            force=args.force,
            download=True,
            version=args.version,
            clean_archives=args.clean_archives,
            subsets=args.subsets,
        )

    elif args.dataset == MACSCard.NAME:
        download_macs(
            root=args.root,
            verbose=args.verbose,
            force=args.force,
            download=True,
            clean_archives=args.clean_archives,
        )

    elif args.dataset == WavCapsCard.NAME:
        download_wavcaps(
            root=args.root,
            verbose=args.verbose,
            force=args.force,
            download=True,
            clean_archives=args.clean_archives,
            subsets=args.subsets,
            hf_cache_dir=args.hf_cache_dir,
            revision=args.revision,
        )

    else:
        DATASETS = ("audiocaps", "clotho" or "macs")
        raise ValueError(
            f"Invalid argument {args.dataset}. (expected one of {DATASETS})"
        )


if __name__ == "__main__":
    _main_download()
