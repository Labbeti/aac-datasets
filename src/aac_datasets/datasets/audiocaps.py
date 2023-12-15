#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os.path as osp

from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torchaudio

from torch import Tensor
from typing_extensions import TypedDict, NotRequired

try:
    # To support torchaudio >=2.1.0
    from torchaudio import AudioMetaData  # type: ignore
except ImportError:
    from torchaudio.backend.common import AudioMetaData

from aac_datasets.datasets.base import AACDataset
from aac_datasets.datasets.functional.audiocaps import (
    AudioCapsCard,
    load_audiocaps_dataset,
    load_class_labels_indices,
    prepare_audiocaps_dataset,
    _get_audio_subset_dpath,
)
from aac_datasets.utils.globals import _get_root, _get_ffmpeg_path, _get_ytdl_path


pylog = logging.getLogger(__name__)


class AudioCapsItem(TypedDict, total=True):
    r"""Class representing a single AudioCaps item."""

    # Common attributes
    audio: Tensor
    captions: List[str]
    dataset: str
    fname: str
    index: int
    subset: str
    sr: int
    # AudioCaps-specific attributes
    audiocaps_ids: List[int]
    start_time: int
    tags: NotRequired[List[int]]
    youtube_id: str


class AudioCaps(AACDataset[AudioCapsItem]):
    r"""Unofficial AudioCaps PyTorch dataset.

    Subsets available are 'train', 'val' and 'test'.

    Audio is a waveform tensor of shape (1, n_times) of 10 seconds max, sampled at 32kHz by default.
    Target is a list of strings containing the captions.
    The 'train' subset has only 1 caption per sample and 'val' and 'test' have 5 captions.
    Download requires 'yt-dlp' and 'ffmpeg' commands.

    AudioCaps paper : https://www.aclweb.org/anthology/N19-1011.pdf

    .. code-block:: text
        :caption:  Dataset folder tree

        {root}
        └── AUDIOCAPS
            ├── train.csv
            ├── val.csv
            ├── test.csv
            └── audio_32000Hz
                ├── train
                │    └── (46231/49838 flac files, ~42G for 32kHz)
                ├── val
                │    └── (465/495 flac files, ~425M for 32kHz)
                └── test
                    └── (913/975 flac files, ~832M for 32kHz)

    """

    # Common globals
    CARD: ClassVar[AudioCapsCard] = AudioCapsCard()
    FORCE_PREPARE_DATA: ClassVar[bool] = False
    VERIFY_FILES: ClassVar[bool] = False

    # AudioCaps-specific globals
    AUDIO_DURATION: ClassVar[float] = 10.0
    AUDIO_FORMAT: ClassVar[str] = "flac"
    AUDIO_N_CHANNELS: ClassVar[int] = 1
    DOWNLOAD_AUDIO: ClassVar[bool] = True

    # Initialization
    def __init__(
        self,
        root: Union[str, Path, None] = None,
        subset: str = "train",
        download: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        flat_captions: bool = False,
        verbose: int = 0,
        exclude_removed_audio: bool = True,
        with_tags: bool = False,
        sr: int = 32_000,
        ffmpeg_path: Union[str, Path, None] = None,
        ytdl_path: Union[str, Path, None] = None,
    ) -> None:
        """
        :param root: Dataset root directory.
            The data will be stored in the 'AUDIOCAPS' subdirectory.
            defaults to ".".
        :param subset: The subset of AudioCaps to use. Can be one of :attr:`~AudioCapsCard.SUBSETS`.
            defaults to "train".
        :param download: Download the dataset if download=True and if the dataset is not already downloaded.
            defaults to False.
        :param transform: The transform to apply to the global dict item. This transform is applied only in getitem method when argument is an integer.
            defaults to None.
        :param flat_captions: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param verbose: Verbose level.
            defaults to 0.
        :param exclude_removed_audio: If True, the dataset will exclude from the dataset the audio not downloaded from youtube (i.e. not present on disk).
            If False, invalid audios will return an empty tensor of shape (0,).
            defaults to True.
        :param with_tags: If True, load the tags from AudioSet dataset.
            Note: tags needs to be downloaded with download=True & with_tags=True before being used.
            defaults to False.
        :param sr: The sample rate used for audio files in the dataset (in Hz).
            Since original YouTube videos are recorded in various settings, this parameter allow to download allow audio files with a specific sample rate.
            defaults to 32000.
        :param ffmpeg_path: Path to ffmpeg executable file.
            defaults to "ffmpeg".
        :param ytdl_path: Path to yt-dlp or ytdlp executable.
            defaults to "yt-dlp".
        """
        if subset not in AudioCapsCard.SUBSETS:
            raise ValueError(
                f"Invalid argument subset={subset} for AudioCaps. (expected one of {AudioCapsCard.SUBSETS})"
            )

        root = _get_root(root)
        ytdl_path = _get_ytdl_path(ytdl_path)
        ffmpeg_path = _get_ffmpeg_path(ffmpeg_path)

        if download:
            prepare_audiocaps_dataset(
                root=root,
                subset=subset,
                sr=sr,
                with_tags=with_tags,
                verbose=verbose,
                force=AudioCaps.FORCE_PREPARE_DATA,
                ytdl_path=ytdl_path,
                ffmpeg_path=ffmpeg_path,
                audio_format=AudioCaps.AUDIO_FORMAT,
                audio_duration=AudioCaps.AUDIO_DURATION,
                n_channels=AudioCaps.AUDIO_N_CHANNELS,
                verify_files=AudioCaps.VERIFY_FILES,
                download_audio=AudioCaps.DOWNLOAD_AUDIO,
            )

        raw_data, index_to_tagname = load_audiocaps_dataset(
            root=root,
            subset=subset,
            sr=sr,
            with_tags=with_tags,
            exclude_removed_audio=exclude_removed_audio,
            verbose=verbose,
            audio_format=AudioCaps.AUDIO_FORMAT,
        )
        audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr)
        size = len(next(iter(raw_data.values())))
        raw_data["dataset"] = [AudioCapsCard.NAME] * size
        raw_data["subset"] = [subset] * size
        raw_data["fpath"] = [
            osp.join(audio_subset_dpath, fname) for fname in raw_data["fname"]
        ]
        raw_data["index"] = list(range(size))

        column_names = list(AudioCapsItem.__required_keys__) + list(
            AudioCapsItem.__optional_keys__
        )
        if not with_tags:
            column_names.remove("tags")

        super().__init__(
            raw_data=raw_data,
            transform=transform,
            column_names=column_names,
            flat_captions=flat_captions,
            sr=sr,
            verbose=verbose,
        )

        # Attributes
        self._root = root
        self._subset = subset
        self._download = download
        self._exclude_removed_audio = exclude_removed_audio
        self._with_tags = with_tags
        self._index_to_tagname = index_to_tagname

        self.add_post_columns(
            {
                "audio": AudioCaps._load_audio,
                "audio_metadata": AudioCaps._load_audio_metadata,
                "duration": AudioCaps._load_duration,
                "num_channels": AudioCaps._load_num_channels,
                "num_frames": AudioCaps._load_num_frames,
                "sr": AudioCaps._load_sr,
            }
        )

    # Properties
    @property
    def download(self) -> bool:
        return self._download

    @property
    def exclude_removed_audio(self) -> bool:
        return self._exclude_removed_audio

    @property
    def index_to_tagname(self) -> List[str]:
        return self._index_to_tagname

    @property
    def root(self) -> str:
        return self._root

    @property
    def sr(self) -> int:
        return self._sr  # type: ignore

    @property
    def subset(self) -> str:
        return self._subset

    @property
    def with_tags(self) -> bool:
        return self._with_tags

    # Public class methods
    @classmethod
    def load_class_labels_indices(
        cls,
        root: str,
        sr: int = 32_000,
    ) -> List[Dict[str, str]]:
        return load_class_labels_indices(root, sr)

    # Magic methods
    def __repr__(self) -> str:
        repr_dic = {
            "subset": self._subset,
            "size": len(self),
            "num_columns": len(self.column_names),
            "with_tags": self._with_tags,
            "exclude_removed_audio": self._exclude_removed_audio,
        }
        repr_str = ", ".join(f"{k}={v}" for k, v in repr_dic.items())
        return f"{AudioCapsCard.PRETTY_NAME}({repr_str})"

    # Private methods
    def _load_audio(self, idx: int) -> Tensor:
        if not self._raw_data["is_on_disk"][idx]:
            return torch.empty((0,))
        fpath = self.at(idx, "fpath")
        audio, sr = torchaudio.load(fpath)  # type: ignore

        # Sanity check
        if audio.nelement() == 0:
            raise RuntimeError(
                f"Invalid audio number of elements in {fpath}. (expected audio.nelement()={audio.nelement()} > 0)"
            )

        if self._sr is not None and (self._sr != sr):
            raise RuntimeError(
                f"Invalid sample rate {sr}Hz for audio {fpath}. (expected {self._sr}Hz)"
            )
        return audio

    def _load_audio_metadata(self, idx: int) -> AudioMetaData:
        if not self._raw_data["is_on_disk"][idx]:
            return AudioMetaData(-1, -1, -1, -1, "unknown_encoding")
        fpath = self.at(idx, "fpath")
        audio_metadata = torchaudio.info(fpath)  # type: ignore
        return audio_metadata
