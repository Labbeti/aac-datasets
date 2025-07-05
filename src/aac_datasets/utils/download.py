#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
from pathlib import Path
from typing import Union

from pythonwrench.hashlib import hash_file  # noqa: F401
from pythonwrench.os import safe_rmdir  # noqa: F401
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
