#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp
from pathlib import Path
from typing import Any, Callable, ClassVar, List, Optional, Union

from torch import Tensor
from typing_extensions import NotRequired, TypedDict

from aac_datasets.datasets.base import AACDataset
from aac_datasets.datasets.functional.clotho import (
    ClothoCard,
    ClothoSubset,
    ClothoVersion,
    _get_audio_subset_dpath,
    download_clotho_dataset,
    load_clotho_dataset,
)
from aac_datasets.utils.globals import _get_root

pylog = logging.getLogger(__name__)


class ClothoItem(TypedDict):
    r"""Class representing a single Clotho item."""

    # Common attributes
    audio: NotRequired[Tensor]
    captions: NotRequired[List[str]]
    dataset: str
    fname: NotRequired[str]
    index: int
    subset: ClothoSubset
    sr: NotRequired[int]
    duration: NotRequired[float]
    # Clotho-specific attributes
    keywords: NotRequired[List[str]]
    sound_id: NotRequired[str]  # warning: some files contains "Not found"
    sound_link: NotRequired[str]  # warning: some files contains "NA"
    start_end_samples: NotRequired[str]  # warning: some files contains ""
    manufacturer: NotRequired[str]
    license: NotRequired[str]


class Clotho(AACDataset[ClothoItem]):
    r"""Unofficial Clotho PyTorch dataset.

    Subsets available are 'train', 'val', 'eval', 'dcase_aac_test', 'dcase_aac_analysis', 'dcase_t2a_audio' and 'dcase_t2a_captions'.

    Audio are waveform sounds of 15 to 30 seconds, sampled at 44100 Hz.
    Target is a list of 5 different sentences strings describing an audio sample.
    The maximal number of words in captions is 20.

    Clotho V1 Paper : https://arxiv.org/pdf/1910.09387.pdf

    .. code-block:: text
        :caption:  Dataset folder tree for version 'v2.1', with all subsets

        {root}
        └── CLOTHO_v2.1
            ├── archives
            |   └── (5 7z files, ~8.9GB)
            ├── clotho_audio_files
            │   ├── clotho_analysis
            │   │    └── (8360 wav files, ~19GB)
            │   ├── development
            │   │    └── (3839 wav files, ~7.1GB)
            │   ├── evaluation
            │   │    └── (1045 wav files, ~2.0GB)
            │   ├── test
            │   |    └── (1043 wav files, ~2.0GB)
            │   ├── test_retrieval_audio
            │   |    └── (1000 wav files, ~2.0GB)
            │   └── validation
            │        └── (1045 wav files, ~2.0GB)
            └── clotho_csv_files
                ├── clotho_captions_development.csv
                ├── clotho_captions_evaluation.csv
                ├── clotho_captions_validation.csv
                ├── clotho_metadata_development.csv
                ├── clotho_metadata_evaluation.csv
                ├── clotho_metadata_test.csv
                ├── clotho_metadata_validation.csv
                ├── retrieval_audio_metadata.csv
                └── retrieval_captions.csv
    """

    # Common globals
    CARD: ClassVar[ClothoCard] = ClothoCard()

    # Clotho-specific globals
    INVALID_SOUND_ID: ClassVar[str] = "Not found"
    INVALID_SOUND_LINK: ClassVar[str] = "NA"
    INVALID_START_END_SAMPLES: ClassVar[str] = ""

    # Initialization
    def __init__(
        self,
        # Common args
        root: Union[str, Path, None] = None,
        subset: ClothoSubset = ClothoCard.DEFAULT_SUBSET,
        download: bool = False,
        transform: Optional[Callable[[ClothoItem], Any]] = None,
        verbose: int = 0,
        force_download: bool = False,
        verify_files: bool = False,
        *,
        # Clotho-specific args
        clean_archives: bool = True,
        flat_captions: bool = False,
        version: ClothoVersion = ClothoCard.DEFAULT_VERSION,
    ) -> None:
        """
        :param root: The parent of the dataset root directory.
            Note: The data is stored in the 'CLOTHO_{version}' subdirectory.
            defaults to ".".
        :param subset: The subset of Clotho to use. Can be one of :attr:`~ClothoCard.SUBSETS`.
            defaults to "dev".
        :param download: Download the dataset if download=True and if the dataset is not already downloaded.
            defaults to False.
        :param transform: The transform to apply to the global dict item. This transform is applied only in getitem method when argument is an integer.
            defaults to None.
        :param verbose: Verbose level to use. Can be 0 or 1.
            defaults to 0.
        :param force_download: If True, force to re-download file even if they exists on disk.
            defaults to False.
        :param verify_files: If True, check hash value when possible.
            defaults to False.

        :param clean_archives: If True, remove the compressed archives from disk to save space.
            defaults to True.
        :param flat_captions: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param version: The version of the dataset. Can be one of :attr:`~ClothoCard.VERSIONS`.
            defaults to 'v2.1'.
        """
        if version not in ClothoCard.VERSIONS:
            msg = f"Invalid Clotho argument {version=}. Must be one of {ClothoCard.VERSIONS}."
            raise ValueError(msg)

        if version == "v2":
            msg = (
                f"The version '{version}' of the Clotho dataset contains minor some errors in file names and few corrupted files."
                f"Please consider using the fixed version 'v2.1'."
            )
            pylog.warning(msg)

        if subset not in ClothoCard.SUBSETS:
            msg = f"Invalid Clotho argument subset={subset} for {version=}. Must be one of {ClothoCard.SUBSETS}."
            raise ValueError(msg)

        root = _get_root(root)

        if download:
            download_clotho_dataset(
                root=root,
                subset=subset,
                force=force_download,
                verbose=verbose,
                verify_files=verify_files,
                clean_archives=clean_archives,
                version=version,
            )

        # Exclude some columns containing empty values for several subsets
        column_names = list(ClothoItem.__required_keys__) + list(  # type: ignore
            ClothoItem.__optional_keys__  # type: ignore
        )
        if subset == "dcase_aac_test":
            removed_columns = ("captions", "sound_id", "keywords", "sound_link")
        elif subset == "dcase_aac_analysis":
            removed_columns = (
                "captions",
                "sound_id",
                "keywords",
                "sound_link",
                "license",
                "manufacturer",
                "start_end_samples",
            )
        elif subset == "dcase_t2a_audio":
            removed_columns = ("captions",)
        elif subset == "dcase_t2a_captions":
            removed_columns = (
                "audio",
                "sr",
                "duration",
                "fname",
                "sound_id",
                "keywords",
                "sound_link",
                "license",
                "manufacturer",
                "start_end_samples",
            )
        else:
            removed_columns = ()

        for name in removed_columns:
            column_names.remove(name)

        raw_data = load_clotho_dataset(
            root=root,
            subset=subset,
            verbose=verbose,
            version=version,
        )

        size = len(next(iter(raw_data.values())))
        raw_data["dataset"] = [ClothoCard.NAME] * size
        raw_data["subset"] = [subset] * size
        raw_data["index"] = list(range(size))

        if "audio" not in removed_columns:
            audio_subset_dpath = _get_audio_subset_dpath(root, version, subset)
            assert (
                audio_subset_dpath is not None
            ), "Internal error. (expected audio column but audio dname is None)"
            raw_data["fpath"] = [
                osp.join(audio_subset_dpath, fname) for fname in raw_data["fname"]
            ]

        super().__init__(
            raw_data=raw_data,
            transform=transform,
            column_names=column_names,
            flat_captions=flat_captions,
            sr=ClothoCard.SAMPLE_RATE,
            verbose=verbose,
        )
        self._root = root
        self._subset: ClothoSubset = subset
        self._download = download
        self._version: ClothoVersion = version

        if "audio" not in removed_columns:
            self.add_online_columns(
                {
                    "audio": Clotho._load_audio,
                    "audio_metadata": Clotho._load_audio_metadata,
                    "duration": Clotho._load_duration,
                    "num_channels": Clotho._load_num_channels,
                    "num_frames": Clotho._load_num_frames,
                    "sr": Clotho._load_sr,
                }
            )

    # Properties
    @property
    def download(self) -> bool:
        return self._download

    @property
    def root(self) -> str:
        return self._root

    @property
    def sr(self) -> int:
        return self._sr  # type: ignore

    @property
    def subset(self) -> ClothoSubset:
        return self._subset

    @property
    def version(self) -> ClothoVersion:
        return self._version

    # Magic methods
    def __repr__(self) -> str:
        repr_dic = {
            "subset": self._subset,
            "size": len(self),
            "num_columns": len(self.column_names),
            "version": self._version,
        }
        repr_str = ", ".join(f"{k}={v}" for k, v in repr_dic.items())
        return f"{ClothoCard.PRETTY_NAME}({repr_str})"
