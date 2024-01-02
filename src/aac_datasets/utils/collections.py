#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterable, List, TypeVar


T = TypeVar("T")


def list_dict_to_dict_list(
    lst: List[Dict[str, T]],
    key_mode: str = "intersect",
    default: Any = None,
) -> Dict[str, List[T]]:
    """Convert list of dicts to dict of lists.

    :param lst: The list of dict to merge.
    :param key_mode: Can be "same" or "intersect".
        If "same", all the dictionaries must contains the same keys otherwise a ValueError will be raised.
        If "intersect", only the intersection of all keys will be used in output.
        If "union", the output dict will contains the union of all keys, and the missing value will use the argument default.
    :returns: The dictionary of lists.
    """
    if len(lst) <= 0:
        return {}

    keys = set(lst[0].keys())
    if key_mode == "same":
        if not all(keys == set(item.keys()) for item in lst[1:]):
            raise ValueError("Invalid keys for batch.")
    elif key_mode == "intersect":
        keys = intersect_lists([item.keys() for item in lst])
    elif key_mode == "union":
        keys = union_lists([item.keys() for item in lst])
    else:
        KEY_MODES = ("same", "intersect", "union")
        raise ValueError(
            f"Invalid argument key_mode={key_mode}. (expected one of {KEY_MODES})"
        )

    return {key: [item.get(key, default) for item in lst] for key in keys}


def intersect_lists(lst_of_lst: List[Iterable[T]]) -> List[T]:
    """Performs intersection of elements in lists (like set intersection), but keep their original order."""
    if len(lst_of_lst) <= 0:
        return []
    out = list(dict.fromkeys(lst_of_lst[0]))
    for lst_i in lst_of_lst[1:]:
        out = [name for name in out if name in lst_i]
        if len(out) == 0:
            break
    return out


def union_lists(lst_of_lst: Iterable[Iterable[T]]) -> List[T]:
    """Performs union of elements in lists (like set union), but keep their original order."""
    out = {}
    for lst_i in lst_of_lst:
        out |= dict.fromkeys(lst_i)
    out = list(out)
    return out
