#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp

from pathlib import Path
from typing import Dict, Union, overload


pylog = logging.getLogger(__name__)


__DEFAULT_GLOBALS: Dict[str, Dict[str, Union[str, None]]] = {
    "root": {
        "user": None,
        "env": "AAC_DATASETS_ROOT",
        "package": ".",
    },
    "ytdl": {
        "user": None,
        "env": "AAC_DATASETS_YTDL_PATH",
        "package": "ytdlp",
    },
    "ffmpeg": {
        "user": None,
        "env": "AAC_DATASETS_FFMPEG_PATH",
        "package": "ffmpeg",
    },
    "zip": {
        "user": None,
        "env": "AAC_DATASETS_ZIP_PATH",
        "package": "zip",
    },
}


# Public functions
def get_default_root() -> str:
    """Returns the default root directory path.

    If :func:`~aac_datasets.utils.path.set_default_root` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_DATASETS_ROOT has been set to a string, it will return its value.
    Else it will be equal to "." by default.
    """
    return __get_default_value("root")


def get_default_ytdl_path() -> str:
    """Returns the default youtube-dl executable path.

    If :func:`~aac_datasets.utils.path.set_default_ytdl_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_DATASETS_YTDL_PATH has been set to a string, it will return its value.
    Else it will be equal to "youtube-dl" by default.
    """
    return __get_default_value("ytdl")


def get_default_ffmpeg_path() -> str:
    """Returns the default ffmpeg executable path.

    If :func:`~aac_datasets.utils.path.set_default_ffmpeg_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_DATASETS_FFMPEG_PATH has been set to a string, it will return its value.
    Else it will be equal to "ffmpeg" by default.
    """
    return __get_default_value("ffmpeg")


def get_default_zip_path() -> str:
    """Returns the default zip executable path.

    If :func:`~aac_datasets.utils.path.set_default_zip_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_DATASETS_ZIP_PATH has been set to a string, it will return its value.
    Else it will be equal to "zip" by default.
    """
    return __get_default_value("zip")


def set_default_root(cache_path: Union[str, Path, None]) -> None:
    """Override default root directory path."""
    __set_default_value("root", cache_path)


def set_default_ytdl_path(java_path: Union[str, Path, None]) -> None:
    """Override default youtube-dl executable path."""
    __set_default_value("ytdl", java_path)


def set_default_ffmpeg_path(tmp_path: Union[str, Path, None]) -> None:
    """Override default ffmpeg executable path."""
    __set_default_value("ffmpeg", tmp_path)


def set_default_zip_path(tmp_path: Union[str, Path, None]) -> None:
    """Override default zip executable path."""
    __set_default_value("zip", tmp_path)


# Private functions
def _get_root(root: Union[str, Path, None] = None) -> str:
    return __get_value("root", root)


def _get_ytdl_path(ytdl_path: Union[str, Path, None] = None) -> str:
    return __get_value("ytdl", ytdl_path)


def _get_ffmpeg_path(ffmpeg_path: Union[str, Path, None] = None) -> str:
    return __get_value("ffmpeg", ffmpeg_path)


def _get_zip_path(zip_path: Union[str, Path, None] = None) -> str:
    return __get_value("zip", zip_path)


def __get_default_value(value_name: str) -> str:
    values = __DEFAULT_GLOBALS[value_name]

    for source, value_or_env_varname in values.items():
        if value_or_env_varname is None:
            continue

        if source.startswith("env"):
            path = os.getenv(value_or_env_varname, None)
        else:
            path = value_or_env_varname

        if path is not None:
            path = __process_value(path)
            return path

    pylog.error(f"Paths values: {values}")
    raise RuntimeError(
        f"Invalid default path for path_name={value_name}. (all default paths are None)"
    )


def __set_default_value(
    value_name: str,
    value: Union[str, Path, None],
) -> None:
    value = __process_value(value)
    __DEFAULT_GLOBALS[value_name]["user"] = value


def __get_value(value_name: str, value: Union[str, Path, None] = None) -> str:
    if value is ... or value is None:
        return __get_default_value(value_name)
    else:
        value = __process_value(value)
        return value


@overload
def __process_value(value: None) -> None:
    ...


@overload
def __process_value(value: Union[str, Path]) -> str:
    ...


def __process_value(value: Union[str, Path, None]) -> Union[str, None]:
    if value is None or value is ...:
        return None

    value = str(value)
    value = osp.expanduser(value)
    value = osp.expandvars(value)
    return value
