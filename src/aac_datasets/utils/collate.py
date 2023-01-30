#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Union

import torch

from torch import Tensor
from torch.nn import functional as F


class BasicCollate:
    """Collate object for :class:`~torch.utils.data.dataloader.DataLoader`.

    Merge lists in dicts into a single dict of lists. No padding is applied.
    """

    def __call__(self, batch_lst: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        return _lst_dic_to_dic_lst(batch_lst)


class AdvancedCollate:
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
        super().__init__()
        self.fill_values = fill_values

    def __call__(self, batch_lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_dic: Dict[str, Any] = _lst_dic_to_dic_lst(batch_lst)
        keys = list(batch_dic.keys())

        for key in keys:
            values = batch_dic[key]

            if len(values) == 0:
                if key in self.fill_values:
                    values = torch.as_tensor(values)
                batch_dic[key] = values
                continue

            if key in self.fill_values:
                values = list(map(torch.as_tensor, values))

            are_tensors = [isinstance(value, Tensor) for value in values]
            if not all(are_tensors):
                batch_dic[key] = values
                continue

            are_stackables = [value.shape == values[0].shape for value in values]
            if all(are_stackables):
                values = torch.stack(values)
                batch_dic[key] = values
                continue

            if key in self.fill_values:
                are_paddable = [
                    value.ndim > 0 and value.shape[:-1] == values[0].shape[:-1]
                    for value in values
                ]
                if all(are_paddable):
                    target_length = max(audio_i.shape[-1] for audio_i in values)
                    values = torch.stack(
                        [
                            pad_last_dim(audio_i, target_length, self.fill_values[key])
                            for audio_i in values
                        ]
                    )

            batch_dic[key] = values
        return batch_dic


def pad_last_dim(tensor: Tensor, target_length: int, pad_value: float) -> Tensor:
    """Left padding tensor at last dim.

    :param tensor: Tensor of at least 1 dim. (..., T)
    :param target_length: Target length of the last dim. If target_length <= T, the function has no effect.
    :param pad_value: Fill value used to pad tensor.
    :returns: A tensor of shape (..., target_length).
    """
    pad_len = max(target_length - tensor.shape[-1], 0)
    return F.pad(tensor, [0, pad_len], value=pad_value)


def _lst_dic_to_dic_lst(
    lst: List[Dict[str, Any]], key_mode: str = "intersect"
) -> Dict[str, List[Any]]:
    """Convert list of dicts to dict of lists.

    :param lst: The list of dict to merge.
    :param key_mode: Can be "same" or "intersect".
        If "same", all the dictionaries must contains the same keys otherwise a ValueError will be raised.
        If "intersect", only the intersection of all keys will be used in output.
    :returns: The dictionary of lists.
    """
    if len(lst) == 0:
        return {}
    keys = set(lst[0].keys())
    if key_mode == "same":
        if not all(keys == set(item.keys()) for item in lst[1:]):
            raise ValueError("Invalid keys for batch.")
    elif key_mode == "intersect":
        for item in lst[1:]:
            keys = keys.intersection(item.keys())
    else:
        raise ValueError(f"Invalid argument key_mode={key_mode}.")

    return {key: [item[key] for item in lst] for key in keys}
