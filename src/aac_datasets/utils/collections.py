#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import Literal

K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

KEY_MODES = ("same", "intersect", "union")
KeyMode = Literal["intersect", "same", "union"]


def all_eq(it: Iterable[T], eq_fn: Optional[Callable[[T, T], bool]] = None) -> bool:
    """Returns true if all elements in inputs are equal."""
    it = list(it)
    first = it[0]
    if eq_fn is None:
        return all(first == elt for elt in it)
    else:
        return all(eq_fn(first, elt) for elt in it)


@overload
def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: Literal["intersect", "same"],
    default_val: Any = None,
) -> Dict[K, List[V]]:
    ...


@overload
def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: Literal["union"] = "union",
    default_val: W = None,
) -> Dict[K, List[Union[V, W]]]:
    ...


def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: KeyMode = "union",
    default_val: W = None,
) -> Dict[K, List[Union[V, W]]]:
    """Convert list of dicts to dict of lists.

    Args:
        lst: The list of dict to merge.
        key_mode: Can be "same" or "intersect".
            If "same", all the dictionaries must contains the same keys otherwise a ValueError will be raised.
            If "intersect", only the intersection of all keys will be used in output.
            If "union", the output dict will contains the union of all keys, and the missing value will use the argument default_val.
        default_val: Default value of an element when key_mode is "union". defaults to None.
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
        raise ValueError(f"Invalid argument {key_mode=}. (expected one of {KEY_MODES})")

    return {key: [item.get(key, default_val) for item in lst] for key in keys}


def intersect_lists(lst_of_lst: Sequence[Iterable[T]]) -> List[T]:
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
        out.update(dict.fromkeys(lst_i))
    out = list(out)
    return out


@overload
def dict_list_to_list_dict(
    dic: Mapping[T, Sequence[U]],
    key_mode: Literal["same", "intersect"],
    default_val: Any = None,
) -> List[Dict[T, U]]:
    ...


@overload
def dict_list_to_list_dict(
    dic: Mapping[T, Sequence[U]],
    key_mode: Literal["union"] = "union",
    default_val: W = None,
) -> List[Dict[T, Union[U, W]]]:
    ...


def dict_list_to_list_dict(
    dic: Mapping[T, Sequence[U]],
    key_mode: KeyMode = "union",
    default_val: W = None,
) -> List[Dict[T, Union[U, W]]]:
    """Convert dict of lists with same sizes to list of dicts.

    Example 1
    ----------
    ```
    >>> dic = {"a": [1, 2], "b": [3, 4]}
    >>> dict_list_to_list_dict(dic)
    ... [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
    ```
    """
    if len(dic) == 0:
        return []

    lengths = [len(seq) for seq in dic.values()]

    if key_mode == "same":
        if not all_eq(lengths):
            raise ValueError("Invalid sequences for batch.")
        length = lengths[0]
    elif key_mode == "intersect":
        length = min(lengths)
    elif key_mode == "union":
        length = max(lengths)
    else:
        msg = f"Invalid argument {key_mode=}. (expected one of {KEY_MODES})"
        raise ValueError(msg)

    result = [
        {k: (v[i] if i < len(v) else default_val) for k, v in dic.items()}
        for i in range(length)
    ]
    return result


def union_dicts(*dicts: Dict[T, U]) -> Dict[T, U]:
    """Performs union of elements in dicts."""
    out = {}
    for dic in dicts:
        out.update(dic)
    return out
