#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Union

from pythonwrench.warnings import deprecated_alias, deprecated_function
from torch import Tensor
from torchwrench.nn.functional.padding import pad_dim
from torchwrench.nn.functional.predicate import is_stackable
from torchwrench.utils.data.collate import AdvancedCollateDict, CollateDict, KeyMode


class BasicCollate(CollateDict):
    """Collate object for :class:`~torch.utils.data.dataloader.DataLoader`.

    Merge lists in dicts into a single dict of lists. No padding is applied.
    """

    def __init__(self, key_mode: KeyMode = "intersect") -> None:
        super().__init__(key_mode)


class AdvancedCollate(AdvancedCollateDict):
    """Advanced collate object for :class:`~torch.utils.data.dataloader.DataLoader`.

    Merge lists in dicts into a single dict of lists.
    Audio will be padded if a fill value is given in `__init__`.

    .. code-block:: python
        :caption:  Example

        >>> collate = AdvancedCollate({"audio": 0.0})
        >>> loader = DataLoader(..., collate_fn=collate)
        >>> next(iter(loader))
        ... {"audio": tensor([[...]]), ...}

    """

    def __init__(self, fill_values: Dict[str, Union[float, int]]) -> None:
        super().__init__(fill_values, key_mode="intersect")


@deprecated_function()
def pad_last_dim(tensor: Tensor, target_length: int, pad_value: float) -> Tensor:
    """Left padding tensor at last dim.

    :param tensor: Tensor of at least 1 dim. (..., T)
    :param target_length: Target length of the last dim. If target_length <= T, the function has no effect.
    :param pad_value: Fill value used to pad tensor.
    :returns: A tensor of shape (..., target_length).
    """
    return pad_dim(tensor, target_length, pad_value=pad_value)


@deprecated_alias(is_stackable)
def can_be_stacked(*args, **kwargs): ...
