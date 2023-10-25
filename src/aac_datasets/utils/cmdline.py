#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys


_TRUE_VALUES = ("true", "1", "t", "yes", "y")
_FALSE_VALUES = ("false", "0", "f", "no", "n")


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


def _setup_logging(pkg_name: str, verbose: int) -> None:
    format_ = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
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
