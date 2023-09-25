#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, TypeVar


T = TypeVar("T")


def list_dict_to_dict_list(
    lst: List[Dict[str, T]],
    key_mode: str = "intersect",
) -> Dict[str, List[T]]:
    """Convert list of dicts to dict of lists.

    :param lst: The list of dict to merge.
    :param key_mode: Can be "same" or "intersect".
        If "same", all the dictionaries must contains the same keys otherwise a ValueError will be raised.
        If "intersect", only the intersection of all keys will be used in output.
    :returns: The dictionary of lists.
    """
    if len(lst) <= 0:
        return {}
    keys = set(lst[0].keys())
    if key_mode == "same":
        if not all(keys == set(item.keys()) for item in lst[1:]):
            raise ValueError("Invalid keys for batch.")
    elif key_mode == "intersect":
        keys = intersect_lists([list(item.keys()) for item in lst])
    else:
        KEY_MODES = ("same", "intersect")
        raise ValueError(
            f"Invalid argument key_mode={key_mode}. (expected one of {KEY_MODES})"
        )

    return {key: [item[key] for item in lst] for key in keys}


def intersect_lists(lst_of_lst: List[List[T]]) -> List[T]:
    """Performs intersection of elements in lists (like set intersection), but keep their original order."""
    if len(lst_of_lst) <= 0:
        return []
    out = list(dict.fromkeys(lst_of_lst[0]))
    for lst_i in lst_of_lst[1:]:
        out = [name for name in out if name in lst_i]
        if len(out) == 0:
            break
    return out
