#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

import torch
import torchaudio
from torch import Tensor
from typing_extensions import NotRequired, TypedDict

try:
    # To support torchaudio >= 2.1.0
    from torchaudio import AudioMetaData  # type: ignore
except ImportError:
    from torchaudio.backend.common import AudioMetaData

from aac_datasets.datasets.base import AACDataset
from aac_datasets.datasets.functional.audiocaps import (
    AudioCapsCard,
    AudioCapsSubset,
    AudioCapsVersion,
    _get_audio_subset_dpath,
    download_audiocaps_dataset,
    load_audiocaps_dataset,
)
from aac_datasets.utils.globals import _get_ffmpeg_path, _get_root, _get_ytdlp_path

pylog = logging.getLogger(__name__)


class AudioCapsItem(TypedDict):
    r"""Class representing a single AudioCaps item."""

    # Common attributes
    audio: Tensor
    captions: List[str]
    dataset: str
    fname: str
    index: int
    subset: AudioCapsSubset
    sr: int
    duration: float
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
        :caption:  Dataset folder tree (for version v1)

        {root}
        └── AUDIOCAPS
            ├── csv_files_v1
            │   ├── train.csv
            │   ├── val.csv
            │   └── test.csv
            └── audio_32000Hz
                ├── train
                │   └── (46231/49838 flac files, ~42G for 32kHz)
                ├── val
                │   └── (465/495 flac files, ~425M for 32kHz)
                └── test
                    └── (913/975 flac files, ~832M for 32kHz)

    """

    # Common globals
    CARD: ClassVar[AudioCapsCard] = AudioCapsCard()

    # Initialization
    def __init__(
        self,
        # Common args
        root: Union[str, Path, None] = None,
        subset: AudioCapsSubset = AudioCapsCard.DEFAULT_SUBSET,
        download: bool = False,
        transform: Optional[Callable[[AudioCapsItem], Any]] = None,
        verbose: int = 0,
        force_download: bool = False,
        verify_files: bool = False,
        *,
        # AudioCaps-specific args
        audio_duration: float = 10.0,
        audio_format: str = "flac",
        audio_n_channels: int = 1,
        download_audio: bool = True,
        exclude_removed_audio: bool = True,
        ffmpeg_path: Union[str, Path, None] = None,
        flat_captions: bool = False,
        max_workers: Optional[int] = 1,
        sr: int = 32_000,
        with_tags: bool = False,
        ytdlp_path: Union[str, Path, None] = None,
        version: AudioCapsVersion = AudioCapsCard.DEFAULT_VERSION,
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
        :param verbose: Verbose level.
            defaults to 0.
        :param force_download: If True, force to re-download file even if they exists on disk.
            defaults to False.
        :param verify_files: If True, check hash value when possible.
            defaults to False.

        :param audio_duration: Extracted duration for each audio file in seconds.
            defaults to 10.0.
        :param audio_format: Audio format and extension name.
            defaults to "flac".
        :param audio_n_channels: Number of channels extracted for each audio file.
            defaults to 1.
        :param download_audio: If True, download audio, metadata and labels files. Otherwise it will only donwload metadata and labels files.
            defaults to True.
        :param exclude_removed_audio: If True, the dataset will exclude from the dataset the audio not downloaded from youtube (i.e. not present on disk).
            If False, invalid audios will return an empty tensor of shape (0,).
            defaults to True.
        :param ffmpeg_path: Path to ffmpeg executable file.
            defaults to "ffmpeg".
        :param flat_captions: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param max_workers: Number of threads to download audio files in parallel.
            Do not use a value too high to avoid "Too Many Requests" error.
            The value None will use `min(32, os.cpu_count() + 4)` workers, which is the default of ThreadPoolExecutor.
            defaults to 1.
        :param sr: The sample rate used for audio files in the dataset (in Hz).
            Since original YouTube videos are recorded in various settings, this parameter allow to download allow audio files with a specific sample rate.
            defaults to 32000.
        :param verify_files: If True, check all file already downloaded are valid.
            defaults to False.
        :param with_tags: If True, load the tags from AudioSet dataset.
            Note: tags needs to be downloaded with download=True & with_tags=True before being used.
            defaults to False.
        :param ytdlp_path: Path to yt-dlp or ytdlp executable.
            defaults to "yt-dlp".
        :param version: The version of the dataset. Can be one of :attr:`~AudioCapsCard.VERSIONS`.
            defaults to 'v1'.
        """
        if subset not in AudioCapsCard.SUBSETS:
            msg = f"Invalid argument {subset=} for AudioCaps. (expected one of {AudioCapsCard.SUBSETS})"
            raise ValueError(msg)

        root = _get_root(root)
        ytdlp_path = _get_ytdlp_path(ytdlp_path)
        ffmpeg_path = _get_ffmpeg_path(ffmpeg_path)

        if download:
            download_audiocaps_dataset(
                root=root,
                subset=subset,
                force=force_download,
                verbose=verbose,
                verify_files=verify_files,
                audio_duration=audio_duration,
                audio_format=audio_format,
                audio_n_channels=audio_n_channels,
                download_audio=download_audio,
                ffmpeg_path=ffmpeg_path,
                max_workers=max_workers,
                sr=sr,
                with_tags=with_tags,
                ytdlp_path=ytdlp_path,
                version=version,
            )

        raw_data, index_to_name = load_audiocaps_dataset(
            root=root,
            subset=subset,
            verbose=verbose,
            audio_format=audio_format,
            exclude_removed_audio=exclude_removed_audio,
            sr=sr,
            with_tags=with_tags,
        )
        audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr, version)
        size = len(next(iter(raw_data.values())))
        raw_data["dataset"] = [AudioCapsCard.NAME] * size
        raw_data["subset"] = [subset] * size
        raw_data["fpath"] = [
            osp.join(audio_subset_dpath, fname) for fname in raw_data["fname"]
        ]
        raw_data["index"] = list(range(size))

        column_names = list(AudioCapsItem.__required_keys__) + list(  # type: ignore
            AudioCapsItem.__optional_keys__  # type: ignore
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
        self._subset: AudioCapsSubset = subset
        self._download = download
        self._exclude_removed_audio = exclude_removed_audio
        self._with_tags = with_tags
        self._version: AudioCapsVersion = version
        self._index_to_name = index_to_name

        self.add_online_columns(
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
    def index_to_name(self) -> Dict[int, str]:
        return self._index_to_name

    @property
    def root(self) -> str:
        return self._root

    @property
    def sr(self) -> int:
        return self._sr  # type: ignore

    @property
    def subset(self) -> AudioCapsSubset:
        return self._subset

    @property
    def version(self) -> AudioCapsVersion:
        return self._version

    @property
    def with_tags(self) -> bool:
        return self._with_tags

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
    def _load_audio(self, index: int) -> Tensor:
        if not self._raw_data["is_on_disk"][index]:
            return torch.empty((0,))

        fpath = self.at(index, "fpath")
        audio, sr = torchaudio.load(fpath)  # type: ignore

        # Sanity check
        if audio.nelement() == 0:
            raise RuntimeError(
                f"Invalid audio number of elements in {fpath}. (expected {audio.nelement()=} > 0)"
            )

        if self._sr is not None and (self._sr != sr):
            raise RuntimeError(
                f"Invalid sample rate {sr}Hz for audio {fpath}. (expected {self._sr}Hz)"
            )
        return audio

    def _load_audio_metadata(self, index: int) -> AudioMetaData:
        if not self._raw_data["is_on_disk"][index]:
            return AudioMetaData(-1, -1, -1, -1, "unknown_encoding")
        fpath = self.at(index, "fpath")
        audio_metadata = torchaudio.info(fpath)  # type: ignore
        return audio_metadata
