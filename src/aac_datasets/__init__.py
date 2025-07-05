#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Audio Captioning datasets for PyTorch.  """


__author__ = "Étienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__docs__ = "Audio Captioning Datasets"
__docs_url__ = "https://aac-datasets.readthedocs.io/en/stable/"
__license__ = "MIT"
__maintainer__ = "Étienne Labbé (Labbeti)"
__name__ = "aac-datasets"
__status__ = "Development"
__version__ = "0.6.0"


from .datasets.audiocaps import AudioCaps
from .datasets.clotho import Clotho
from .datasets.macs import MACS
from .datasets.wavcaps import WavCaps
from .utils.globals import (
    get_default_ffmpeg_path,
    get_default_root,
    get_default_ytdlp_path,
    set_default_ffmpeg_path,
    set_default_root,
    set_default_ytdlp_path,
)

__all__ = [
    "AudioCaps",
    "Clotho",
    "MACS",
    "WavCaps",
    "get_default_ffmpeg_path",
    "get_default_root",
    "get_default_ytdlp_path",
    "set_default_ffmpeg_path",
    "set_default_root",
    "set_default_ytdlp_path",
]
