#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import os.path as osp
from pathlib import Path
from typing import Dict, Union

from torch.hub import download_url_to_file

_AUDIOSET_INFOS = {
    "class_labels_indices": {
        "fname": "class_labels_indices.csv",
        "url": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
    },
}
_DEFAULT_CACHE_PATH = Path.home().joinpath(".cache", "audioset_mapping")


def get_audioset_mapping_cache_path(cache_path: Union[str, Path, None] = None) -> Path:
    if cache_path is not None:
        return Path(cache_path)
    else:
        return _DEFAULT_CACHE_PATH


def download_audioset_mapping(
    cache_path: Union[str, Path, None] = None,
    verbose: int = 0,
) -> None:
    cache_path = get_audioset_mapping_cache_path(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    info = _AUDIOSET_INFOS["class_labels_indices"]
    map_fname = info["fname"]
    map_fpath = cache_path.joinpath(map_fname)

    url = info["url"]
    download_url_to_file(url, str(map_fpath), progress=verbose >= 1)


def load_audioset_mapping(
    key_name: str = "index",
    val_name: str = "display_name",
    offline: bool = False,
    cache_path: Union[str, Path, None] = None,
    verbose: int = 0,
) -> Dict:
    NAMES = ("index", "mid", "display_name")
    if key_name not in NAMES:
        raise ValueError(f"Invalid argument {key_name=}. (expected one of {NAMES})")
    if val_name not in NAMES:
        raise ValueError(f"Invalid argument {val_name=}. (expected one of {NAMES})")
    if key_name == val_name:
        raise ValueError(
            f"Invalid arguments key_name={key_name} with {val_name=}. (expected different values)"
        )

    cache_path = get_audioset_mapping_cache_path(cache_path)

    info = _AUDIOSET_INFOS["class_labels_indices"]
    map_fname = info["fname"]
    map_fpath = cache_path.joinpath(map_fname)

    if not osp.isfile(map_fpath):
        if offline:
            raise FileNotFoundError(
                f"Cannot find or download audioset mapping file in '{map_fpath}' with mode {offline=}."
            )

        download_audioset_mapping(cache_path, verbose)

    with open(map_fpath, "r") as file:
        reader = csv.DictReader(file, skipinitialspace=True, strict=True)
        data = list(reader)

    keys = [data_i[key_name] for data_i in data]
    values = [data_i[val_name] for data_i in data]

    if key_name == "index":
        keys = list(map(int, keys))
    if val_name == "index":
        values = list(map(int, values))

    mapping = dict(zip(keys, values))
    return mapping


def load_audioset_name_to_idx(
    offline: bool = False,
    cache_path: Union[str, Path, None] = None,
    verbose: int = 0,
) -> Dict[str, int]:
    return load_audioset_mapping("display_name", "index", offline, cache_path, verbose)
