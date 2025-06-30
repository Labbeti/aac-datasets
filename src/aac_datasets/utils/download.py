#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
from pathlib import Path
from typing import List, Union

import tqdm
from pythonwrench.hashlib import hash_file  # noqa: F401
from torch.hub import download_url_to_file


def download_file(
    url: str,
    fpath: Union[str, Path],
    make_intermediate: bool = False,
    verbose: int = 0,
) -> None:
    if make_intermediate:
        dpath = osp.dirname(fpath)
        os.makedirs(dpath, exist_ok=True)

    download_url_to_file(url, str(fpath), progress=verbose > 0)


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
            msg = f"Cannot remove non-empty directory '{dpath}'."
            raise RuntimeError(msg)

    for dpath in to_delete:
        os.rmdir(dpath)

    return to_delete
