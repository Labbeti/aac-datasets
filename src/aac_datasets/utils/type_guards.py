#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable, List

from typing_extensions import TypeGuard


def is_iterable_bool(x: Any) -> TypeGuard[Iterable[bool]]:
    return isinstance(x, Iterable) and all(isinstance(xi, bool) for xi in x)


def is_iterable_int(x: Any) -> TypeGuard[Iterable[int]]:
    return isinstance(x, Iterable) and all(isinstance(xi, int) for xi in x)


def is_iterable_str(x: Any, *, accept_str: bool) -> TypeGuard[Iterable[str]]:
    return (accept_str and isinstance(x, str)) or (
        not isinstance(x, str)
        and isinstance(x, Iterable)
        and all(isinstance(xi, str) for xi in x)
    )


def is_list_bool(x: Any) -> TypeGuard[List[bool]]:
    return isinstance(x, list) and all(isinstance(xi, bool) for xi in x)


def is_list_int(x: Any) -> TypeGuard[List[int]]:
    return isinstance(x, list) and all(isinstance(xi, int) for xi in x)
