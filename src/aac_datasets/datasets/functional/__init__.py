#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functional interface to load/download datasets.

Each module contains the its own logic for data preparation, but returns a similar dict structure for each dataset class.
"""

from .audiocaps import AudioCapsCard, load_audiocaps_dataset, download_audiocaps_dataset
from .clotho import ClothoCard, load_clotho_dataset, download_clotho_dataset
from .macs import MACSCard, load_macs_dataset, download_macs_dataset
from .wavcaps import WavCapsCard, load_wavcaps_dataset, download_wavcaps_dataset
