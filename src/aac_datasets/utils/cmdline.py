#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
            f"Invalid argument {s=}. (expected one of {_TRUE_VALUES + _FALSE_VALUES})"
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
