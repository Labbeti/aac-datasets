#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal, Tuple, get_args, overload

from .audiocaps import AudioCaps
from .base import AACDataset
from .clotho import Clotho
from .macs import MACS
from .wavcaps import WavCaps

DatasetName = Literal["AudioCaps", "Clotho", "MACS", "WavCaps"]


def list_datasets_names() -> Tuple[DatasetName, ...]:
    return get_args(DatasetName)


@overload
def load_dataset(name: Literal["AudioCaps"], *args, **kwargs) -> AudioCaps: ...


@overload
def load_dataset(name: Literal["Clotho"], *args, **kwargs) -> Clotho: ...


@overload
def load_dataset(name: Literal["MACS"], *args, **kwargs) -> MACS: ...


@overload
def load_dataset(name: Literal["WavCaps"], *args, **kwargs) -> WavCaps: ...


def load_dataset(name: DatasetName, *args, **kwargs) -> AACDataset:
    if name == "AudioCaps":
        return AudioCaps(*args, **kwargs)
    elif name == "Clotho":
        return Clotho(*args, **kwargs)
    elif name == "MACS":
        return MACS(*args, **kwargs)
    elif name == "WavCaps":
        return WavCaps(*args, **kwargs)
    else:
        msg = f"Invalid argument {name=}. (expected one of {get_args(DatasetName)})"
        raise ValueError(msg)
