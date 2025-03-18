#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import os
import os.path as osp
from pathlib import Path
from typing import List, Union

import tqdm
from torch.hub import download_url_to_file
from typing_extensions import Literal

HASH_TYPES = ("sha256", "md5")
DEFAULT_CHUNK_SIZE = 256 * 1024**2  # 256 MiB


def download_file(
    url: str,
    fpath: Union[str, Path],
    make_intermediate: bool = False,
    verbose: int = 0,
) -> None:
    if make_intermediate:
        dpath = osp.dirname(fpath)
        os.makedirs(dpath, exist_ok=True)

    download_url_to_file(url, fpath, progress=verbose > 0)


def safe_rmdir(
    root: Union[str, Path],
    rm_root: bool = True,
    error_on_non_empty_dir: bool = True,
    followlinks: bool = False,
    verbose: int = 0,
) -> List[str]:
    """Remove all empty sub-directories.

    :param root: Root directory path.
    :param rm_root: If True, remove the root directory too. defaults to True.
    :param error_on_non_empty_dir: If True, raises a RuntimeError if a subdirectory contains at least 1 file. Otherwise it will leave non-empty directories. defaults to True.
    :param followlinks: Indicates whether or not symbolic links shound be followed. defaults to False.
    :param verbose: Verbose level. defaults to 0.
    :returns: The list of directories paths deleted.
    """
    to_delete = []

    for dpath, dnames, fnames in tqdm.tqdm(
        os.walk(root, topdown=False, followlinks=followlinks),
        disable=verbose < 2,
    ):
        if not rm_root and dpath == root:
            continue
        elif len(fnames) == 0 and (
            len(dnames) == 0
            or all(osp.join(dpath, dname) in to_delete for dname in dnames)
        ):
            to_delete.append(dpath)
        elif error_on_non_empty_dir:
            raise RuntimeError(f"Cannot remove non-empty directory '{dpath}'.")

    for dpath in to_delete:
        os.rmdir(dpath)

    return to_delete


def hash_file(
    fpath: Union[str, Path],
    hash_type: Literal["sha256", "md5"],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Return the hash value for a file.

    BASED ON https://github.com/pytorch/audio/blob/v0.13.0/torchaudio/datasets/utils.py#L110
    """
    if hash_type == "sha256":
        hasher = hashlib.sha256()
    elif hash_type == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(
            f"Invalid argument {hash_type=}. (expected one of {HASH_TYPES})"
        )

    with open(fpath, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    hash_value = hasher.hexdigest()
    return hash_value
