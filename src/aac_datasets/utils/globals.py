#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
from pathlib import Path
from typing import Any, Union

pylog = logging.getLogger(__name__)


# Public functions
def get_default_root() -> str:
    """Returns the default root directory path.

    If :func:`~aac_datasets.utils.globals.set_default_root` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_DATASETS_ROOT has been set to a string, it will return its value.
    Else it will be equal to "." by default.
    """
    return __get_default_value("root")


def get_default_ffmpeg_path() -> str:
    """Returns the default ffmpeg executable path.

    If :func:`~aac_datasets.utils.globals.set_default_ffmpeg_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_DATASETS_FFMPEG_PATH has been set to a string, it will return its value.
    Else it will be equal to "ffmpeg" by default.
    """
    return __get_default_value("ffmpeg")


def get_default_ytdlp_path() -> str:
    """Returns the default yt-dlp executable path.

    If :func:`~aac_datasets.utils.globals.set_default_ytdlp_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_DATASETS_YTDLP_PATH has been set to a string, it will return its value.
    Else it will be equal to "yt-dlp" by default.
    """
    return __get_default_value("ytdlp")


def get_default_zip_path() -> str:
    """Returns the default zip executable path.

    If :func:`~aac_datasets.utils.globals.set_default_zip_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_DATASETS_ZIP_PATH has been set to a string, it will return its value.
    Else it will be equal to "zip" by default.
    """
    return __get_default_value("zip")


def set_default_root(cache_path: Union[str, Path, None]) -> None:
    """Override default root directory path."""
    __set_default_value("root", cache_path)


def set_default_ffmpeg_path(tmp_path: Union[str, Path, None]) -> None:
    """Override default ffmpeg executable path."""
    __set_default_value("ffmpeg", tmp_path)


def set_default_ytdlp_path(java_path: Union[str, Path, None]) -> None:
    """Override default yt-dl executable path."""
    __set_default_value("ytdlp", java_path)


def set_default_zip_path(tmp_path: Union[str, Path, None]) -> None:
    """Override default zip executable path."""
    __set_default_value("zip", tmp_path)


# Private functions
def _get_root(root: Union[str, Path, None] = None) -> str:
    return __get_value("root", root)


def _get_ffmpeg_path(ffmpeg_path: Union[str, Path, None] = None) -> str:
    return __get_value("ffmpeg", ffmpeg_path)


def _get_ytdlp_path(ytdlp_path: Union[str, Path, None] = None) -> str:
    return __get_value("ytdlp", ytdlp_path)


def _get_zip_path(zip_path: Union[str, Path, None] = None) -> str:
    return __get_value("zip", zip_path)


def __get_default_value(value_name: str) -> str:
    values = __DEFAULT_GLOBALS[value_name]["values"]
    process_func = __DEFAULT_GLOBALS[value_name]["process"]

    for source, value_or_env_varname in values.items():
        if source.startswith("env"):
            value = os.getenv(value_or_env_varname, None)
        else:
            value = value_or_env_varname

        if value is not None:
            value = process_func(value)
            return value

    pylog.error(f"Values: {values}")
    raise RuntimeError(
        f"Invalid default value for {value_name=}. (all default values are None)"
    )


def __set_default_value(
    value_name: str,
    value: Any,
) -> None:
    __DEFAULT_GLOBALS[value_name]["values"]["user"] = value


def __get_value(value_name: str, value: Any = None) -> Any:
    if value is None or value is ...:
        return __get_default_value(value_name)
    else:
        process_func = __DEFAULT_GLOBALS[value_name]["process"]
        value = process_func(value)
        return value


def __process_path(value: Union[str, Path, None]) -> Union[str, None]:
    if value is None or value is ...:
        return None

    value = str(value)
    value = osp.expanduser(value)
    value = osp.expandvars(value)
    return value


__DEFAULT_GLOBALS = {
    "root": {
        "values": {
            "user": None,
            "env": "AAC_DATASETS_ROOT",
            "package": ".",
        },
        "process": __process_path,
    },
    "ytdlp": {
        "values": {
            "user": None,
            "env": "AAC_DATASETS_YTDLP_PATH",
            "package": "yt-dlp",
        },
        "process": __process_path,
    },
    "ffmpeg": {
        "values": {
            "user": None,
            "env": "AAC_DATASETS_FFMPEG_PATH",
            "package": "ffmpeg",
        },
        "process": __process_path,
    },
    "zip": {
        "values": {
            "user": None,
            "env": "AAC_DATASETS_ZIP_PATH",
            "package": "zip",
        },
        "process": __process_path,
    },
}
