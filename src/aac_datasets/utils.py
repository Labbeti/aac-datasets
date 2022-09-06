#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List

import torch

from torch import Tensor
from torch.nn import functional as F


class BasicCollate:
    """Collate object for :class:`~torch.utils.data.dataloader.DataLoader`.

    Returns a tuple of (audio Tensor, captions list).
    The audio will be padded to be stacked into a single tensor.
    """

    def __init__(self, audio_fill_value: float = 0.0) -> None:
        super().__init__()
        self.audio_fill_value = audio_fill_value

    def __call__(self, batch_lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_dic: Dict[str, Any] = _lst_dic_to_dic_lst(batch_lst)

        # Pad audio tensors
        audio_batch = batch_dic["audio"]
        if len(audio_batch) == 0:
            raise ValueError("Cannot collate an empty list of items.")

        are_tensors = [isinstance(audio, Tensor) for audio in audio_batch]
        if not all(are_tensors):
            raise TypeError(
                f"Invalid audio type in {self.__class__.__name__}. (found {are_tensors=})"
            )

        are_paddable = [
            audio.ndim > 0 and audio.shape[:-1] == audio_batch[0].shape[:-1]
            for audio in audio_batch
        ]
        if not all(are_paddable):
            raise TypeError(
                f"Invalid audio shapes in {self.__class__.__name__}. (found {are_paddable=})"
            )

        target_length = max(audio_i.shape[-1] for audio_i in audio_batch)
        audio_batch = torch.stack(
            [
                pad_last_dim(audio_i, target_length, self.audio_fill_value)
                for audio_i in audio_batch
            ]
        )
        batch_dic["audio"] = audio_batch

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


def _lst_dic_to_dic_lst(lst: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Convert list of dicts to dict of lists."""
    if len(lst) == 0:
        return {}
    keys = set(lst[0].keys())
    if not all(keys == set(item.keys()) for item in lst):
        raise ValueError("Invalid keys for batch.")

    return {key: [item[key] for item in lst] for key in keys}
