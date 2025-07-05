#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from aac_datasets.utils.collections import list_dict_to_dict_list


class BasicCollate:
    """Collate object for :class:`~torch.utils.data.dataloader.DataLoader`.

    Merge lists in dicts into a single dict of lists. No padding is applied.
    """

    def __call__(self, batch_lst: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        return list_dict_to_dict_list(batch_lst, key_mode="intersect")


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
        batch_dic: Dict[str, Any] = list_dict_to_dict_list(
            batch_lst,
            key_mode="intersect",
        )
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

            else:
                are_tensors = [isinstance(value, Tensor) for value in values]
                if not all(are_tensors):
                    batch_dic[key] = values
                    continue

            if can_be_stacked(values):
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


def can_be_stacked(tensors: List[Tensor]) -> bool:
    """Returns true if a list of tensors can be stacked with torch.stack function."""
    if len(tensors) == 0:
        return False
    shape0 = tensors[0].shape
    are_stackables = [tensor.shape == shape0 for tensor in tensors]
    return all(are_stackables)
