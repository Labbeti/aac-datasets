#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from argparse import ArgumentParser, Namespace

import yaml

from aac_datasets.datasets.functional.audiocaps import (
    AudioCapsCard,
    download_audiocaps_datasets,
)
from aac_datasets.datasets.functional.clotho import ClothoCard, download_clotho_datasets
from aac_datasets.datasets.functional.macs import MACSCard, download_macs_datasets
from aac_datasets.datasets.functional.wavcaps import (
    WavCapsCard,
    download_wavcaps_datasets,
)
from aac_datasets.utils.cmdline import _str_to_bool, _str_to_opt_int, _str_to_opt_str
from aac_datasets.utils.globals import (
    get_default_ffmpeg_path,
    get_default_root,
    get_default_ytdlp_path,
    get_default_zip_path,
)
from aac_datasets.utils.log_utils import setup_logging_verbose

pylog = logging.getLogger(__name__)


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
        "--force",
        type=_str_to_bool,
        default=False,
        help="Force download of files, even if they are already downloaded.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbose level of the script. 0 means silent mode, 1 is default mode and 2 add additional debugging outputs.",
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
        "--ytdlp_path",
        type=str,
        default=get_default_ytdlp_path(),
        help="Path to yt-dl program used to extract metadata from a youtube video.",
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
    audiocaps_subparser.add_argument(
        "--max_workers",
        type=_str_to_opt_int,
        default=1,
        help="Number of workers used for downloading multiple files in parallel.",
    )
    audiocaps_subparser.add_argument(
        "--version",
        type=str,
        default=AudioCapsCard.DEFAULT_VERSION,
        choices=AudioCapsCard.VERSIONS,
        help="The version of the AudioCaps dataset.",
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
    # Note : MACS only have 1 subset, so we do not add MACS subsets arg
    macs_subparser.add_argument(
        "--clean_archives",
        type=_str_to_bool,
        default=False,
        help="Remove archives files after extraction.",
    )
    macs_subparser.add_argument(
        "--verify_files",
        type=_str_to_bool,
        default=True,
        help="Verify if downloaded files have a valid checksum.",
    )

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
        type=_str_to_opt_str,
        default=None,
        help="Hugging face cache dir.",
    )
    wavcaps_subparser.add_argument(
        "--revision",
        type=str,
        default=WavCapsCard.DEFAULT_REVISION,
        help="Revision of the WavCaps dataset.",
    )
    wavcaps_subparser.add_argument(
        "--zip_path",
        type=str,
        default=get_default_zip_path(),
        help="Path to zip executable to combine and extract WavCaps archives.",
    )

    args = parser.parse_args()
    return args


def _main_download() -> None:
    args = _get_main_download_args()
    setup_logging_verbose("aac_datasets", args.verbose)

    if args.verbose >= 2:
        pylog.debug(yaml.dump({"Arguments": args.__dict__}, sort_keys=False))

    if args.dataset == AudioCapsCard.NAME:
        download_audiocaps_datasets(
            root=args.root,
            subsets=args.subsets,
            force=args.force,
            verbose=args.verbose,
            ffmpeg_path=args.ffmpeg_path,
            max_workers=args.max_workers,
            with_tags=args.with_tags,
            ytdlp_path=args.ytdlp_path,
            version=args.version,
        )

    elif args.dataset == ClothoCard.NAME:
        download_clotho_datasets(
            root=args.root,
            subsets=args.subsets,
            force=args.force,
            verbose=args.verbose,
            clean_archives=args.clean_archives,
            version=args.version,
        )

    elif args.dataset == MACSCard.NAME:
        download_macs_datasets(
            root=args.root,
            force=args.force,
            verbose=args.verbose,
            clean_archives=args.clean_archives,
        )

    elif args.dataset == WavCapsCard.NAME:
        download_wavcaps_datasets(
            root=args.root,
            subsets=args.subsets,
            force=args.force,
            verbose=args.verbose,
            clean_archives=args.clean_archives,
            hf_cache_dir=args.hf_cache_dir,
            revision=args.revision,
            zip_path=args.zip_path,
        )

    else:
        DATASETS = (
            AudioCapsCard.NAME,
            ClothoCard.NAME,
            MACSCard.NAME,
            WavCapsCard.NAME,
        )
        raise ValueError(
            f"Invalid argument {args.dataset}. (expected one of {DATASETS})"
        )


if __name__ == "__main__":
    _main_download()
