#!/usr/bin/env python
# -*- coding: utf-8 -*-

from aac_datasets.audiocaps import AudioCaps
from aac_datasets.clotho import Clotho
from aac_datasets.macs import MACS


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
