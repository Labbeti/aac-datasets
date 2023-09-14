#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audio Captioning Datasets package.
"""

from .datasets.audiocaps import AudioCaps
from .datasets.clotho import Clotho
from .datasets.macs import MACS
from .datasets.wavcaps import WavCaps
from .utils.paths import (
    get_default_ffmpeg_path,
    get_default_root,
    get_default_ytdl_path,
    set_default_ffmpeg_path,
    set_default_root,
    set_default_ytdl_path,
)


__author__ = "Etienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__license__ = "MIT"
__maintainer__ = "Etienne Labbé (Labbeti)"
__name__ = "aac-datasets"
__status__ = "Development"
__version__ = "0.4.0"
