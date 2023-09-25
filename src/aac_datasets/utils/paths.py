#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp

from typing import Dict, Optional, Union


pylog = logging.getLogger(__name__)


__DEFAULT_PATHS: Dict[str, Dict[str, Optional[str]]] = {
    "root": {
        "user": None,
        "env": "AAC_DATASETS_ROOT",
        "package": ".",
    },
    "ytdl": {
        "user": None,
        "env": "AAC_DATASETS_YTDL_PATH",
        "package": "youtube-dl",
    },
    "ffmpeg": {
        "user": None,
        "env": "AAC_DATASETS_FFMPEG_PATH",
        "package": "ffmpeg",
    },
}


# Public functions
def get_default_root() -> str:
    """Returns the default root directory path.

    If :func:`~aac_datasets.utils.path.set_default_root` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_DATASETS_ROOT has been set to a string, it will return its value.
    Else it will be equal to "." by default.
    """
    return __get_default_path("root")


def get_default_ytdl_path() -> str:
    """Returns the default youtube-dl executable path.

    If :func:`~aac_datasets.utils.path.set_default_ytdl_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_DATASETS_YTDL_PATH has been set to a string, it will return its value.
    Else it will be equal to "youtube-dl" by default.
    """
    return __get_default_path("ytdl")


def get_default_ffmpeg_path() -> str:
    """Returns the default ffmpeg executable path.

    If :func:`~aac_datasets.utils.path.set_default_ffmpeg_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_DATASETS_FFMPEG_PATH has been set to a string, it will return its value.
    Else it will be equal to "ffmpeg" by default.
    """
    return __get_default_path("ffmpeg")


def set_default_root(cache_path: Optional[str]) -> None:
    """Override default root directory path."""
    __set_default_path("root", cache_path)


def set_default_ytdl_path(java_path: Optional[str]) -> None:
    """Override default youtube-dl executable path."""
    __set_default_path("ytdl", java_path)


def set_default_ffmpeg_path(tmp_path: Optional[str]) -> None:
    """Override default ffmpeg executable path."""
    __set_default_path("ffmpeg", tmp_path)


# Private functions
def _get_root(root: Union[str, None] = ...) -> str:
    return __get_path("root", root)


def _get_ytdl_path(ytdl_path: Union[str, None] = ...) -> str:
    return __get_path("ytdl", ytdl_path)


def _get_ffmpeg_path(ffmpeg_path: Union[str, None] = ...) -> str:
    return __get_path("ffmpeg", ffmpeg_path)


def __get_default_path(path_name: str) -> str:
    paths = __DEFAULT_PATHS[path_name]

    for name, path_or_var in paths.items():
        if path_or_var is None:
            continue

        if name.startswith("env"):
            path = os.getenv(path_or_var, None)
        else:
            path = path_or_var

        if path is not None:
            path = __process_path(path)
            return path

    pylog.error(f"Paths values: {paths}")
    raise RuntimeError(
        f"Invalid default path for path_name={path_name}. (all default paths are None)"
    )


def __set_default_path(
    path_name: str,
    path: Optional[str],
) -> None:
    if path is not ... and path is not None:
        path = __process_path(path)
    __DEFAULT_PATHS[path_name]["user"] = path


def __get_path(path_name: str, path: Union[str, None] = ...) -> str:
    if path is ... or path is None:
        return __get_default_path(path_name)
    else:
        path = __process_path(path)
        return path


def __process_path(path: str) -> str:
    path = osp.expanduser(path)
    path = osp.expandvars(path)
    return path
