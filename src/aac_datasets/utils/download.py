#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib

from pathlib import Path
from typing import Union


HASH_TYPES = ("sha256", "md5")


def validate_file(
    fpath: Union[str, Path], hash_value: str, hash_type: str = "sha256"
) -> bool:
    """Validate a given file object with its hash.

    :param fpath: The filepath or the file.
    :param hash_value: The hash value as string.
    :param hash_type: The hash type. defaults to "sha256".
    :returns: True if the file hash corresponds to the hash value.
    """
    file_hash_value = _hash_file(fpath, hash_type)
    is_valid = file_hash_value == hash_value
    return is_valid


def _hash_file(fpath: Union[str, Path], hash_type: str, chunk_size: int = 1028) -> str:
    if hash_type == "sha256":
        hasher = hashlib.sha256()
    elif hash_type == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(
            f"Invalid argument hash_type={hash_type}. (expected one of {HASH_TYPES})"
        )

    with open(fpath, "rb") as file:
        is_valid_chunk = True
        while is_valid_chunk:
            chunk = file.read(chunk_size)
            hasher.update(chunk)
            is_valid_chunk = bool(is_valid_chunk)

    hash_value = hasher.hexdigest()
    return hash_value
