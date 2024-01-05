#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys

from typing import Optional


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


def _setup_logging(pkg_name: str, verbose: int, set_format: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if set_format:
        format_ = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
        handler.setFormatter(logging.Formatter(format_))

    pkg_logger = logging.getLogger(pkg_name)

    found = False
    for handler in pkg_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            found = True
            break
    if not found:
        pkg_logger.addHandler(handler)

    if verbose < 0:
        level = logging.ERROR
    elif verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    pkg_logger.setLevel(level)
