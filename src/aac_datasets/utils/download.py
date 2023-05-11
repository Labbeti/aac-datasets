#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib

from pathlib import Path
from typing import Union


HASH_TYPES = ("sha256", "md5")
DEFAULT_CHUNK_SIZE = 256 * 1024**2


def validate_file(
    fpath: Union[str, Path],
    hash_value: str,
    hash_type: str = "sha256",
) -> bool:
    """Validate a given file object with its hash.

    :param fpath: The filepath or the file.
    :param hash_value: The hash value as string.
    :param hash_type: The hash type. defaults to "sha256".
    :returns: True if the file hash corresponds to the hash value.
    """
    hash_value_found = _hash_file(fpath, hash_type)
    is_valid = hash_value_found == hash_value
    return is_valid


def _hash_file(
    fpath: Union[str, Path],
    hash_type: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Return the hash value for a file.

    FROM https://github.com/pytorch/audio/blob/v0.13.0/torchaudio/datasets/utils.py#L110
    """
    if hash_type == "sha256":
        hasher = hashlib.sha256()
    elif hash_type == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(
            f"Invalid argument hash_type={hash_type}. (expected one of {HASH_TYPES})"
        )

    with open(fpath, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    hash_value = hasher.hexdigest()
    return hash_value
