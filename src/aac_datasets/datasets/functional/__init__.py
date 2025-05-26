#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functional interface to load/download datasets.

Each module contains the its own logic for data preparation, but returns a similar dict structure for each dataset class.
"""

from .audiocaps import AudioCapsCard, download_audiocaps_dataset, load_audiocaps_dataset
from .clotho import ClothoCard, download_clotho_dataset, load_clotho_dataset
from .macs import MACSCard, download_macs_dataset, load_macs_dataset
from .wavcaps import WavCapsCard, download_wavcaps_dataset, load_wavcaps_dataset
