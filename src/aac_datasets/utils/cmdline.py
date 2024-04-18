#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
from logging import Logger
from types import ModuleType
from typing import Optional, Sequence, Union

_TRUE_VALUES = ("true", "t", "yes", "y", "1")
_FALSE_VALUES = ("false", "f", "no", "n", "0")
_NONE_VALUES = ("none",)


def _str_to_bool(s: str) -> bool:
    s = str(s).strip().lower()
    if s in _TRUE_VALUES:
        return True
    elif s in _FALSE_VALUES:
        return False
    else:
        raise ValueError(
            f"Invalid argument s={s}. (expected one of {_TRUE_VALUES + _FALSE_VALUES})"
        )


def _str_to_opt_int(s: str) -> Optional[int]:
    s = str(s).strip().lower()
    if s in _NONE_VALUES:
        return None
    else:
        return int(s)


def _str_to_opt_str(s: str) -> Optional[str]:
    s = str(s)
    if s.lower() in _NONE_VALUES:
        return None
    else:
        return s


def setup_logging(
    package_or_logger: Union[
        str,
        ModuleType,
        None,
        Logger,
        Sequence[Union[str, ModuleType, None]],
        Sequence[Logger],
    ],
    verbose: int,
    format_: Optional[str] = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
) -> None:
    if package_or_logger is None or isinstance(
        package_or_logger, (str, Logger, ModuleType)
    ):
        package_or_logger_lst = [package_or_logger]
    else:
        package_or_logger_lst = list(package_or_logger)

    name_or_logger_lst = [
        pkg.__name__ if isinstance(pkg, ModuleType) else pkg
        for pkg in package_or_logger_lst
    ]
    logger_lst = [
        logging.getLogger(pkg_i) if not isinstance(pkg_i, Logger) else pkg_i
        for pkg_i in name_or_logger_lst
    ]

    handler = logging.StreamHandler(sys.stdout)
    if format_ is not None:
        handler.setFormatter(logging.Formatter(format_))

    for logger in logger_lst:
        found = False
        for handler in logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream is sys.stdout
            ):
                found = True
                break
        if not found:
            logger.addHandler(handler)

        level = _verbose_to_logging_level(verbose)
        logger.setLevel(level)


def _verbose_to_logging_level(verbose: int) -> int:
    if verbose < 0:
        level = logging.ERROR
    elif verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    return level
