#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class BasicCollate:
    def __init__(self, pad_value: float = 0.0) -> None:
        super().__init__()
        self.pad_value = pad_value

    def __call__(self, batch: List[Tuple]) -> Tuple:
        audio_batch = [item[0] for item in batch]
        captions_batch = [item[1] for item in batch]

        if len(audio_batch) == 0:
            raise ValueError("Cannot collate an empty list of audio.")

        are_tensors = [isinstance(audio, Tensor) for audio in audio_batch]
        if not all(are_tensors):
            raise TypeError(
                f"Invalid audio type in {self.__class__.__name__}. (found {are_tensors=})"
            )

        are_paddable = [
            audio.ndim > 0 and audio.shape[1:] == audio_batch[0].shape[1:]
            for audio in audio_batch
        ]
        if not all(are_paddable):
            raise TypeError(
                f"Invalid audio shapes in {self.__class__.__name__}. (found {are_paddable=})"
            )

        audio_batch = pad_sequence(
            audio_batch,
            batch_first=True,
            padding_value=self.pad_value,
        )
        return audio_batch, captions_batch
