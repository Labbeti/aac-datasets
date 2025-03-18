#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp
from pathlib import Path
from typing import Any, Callable, ClassVar, List, Optional, Union

from torch import Tensor
from typing_extensions import TypedDict

from aac_datasets.datasets.base import AACDataset
from aac_datasets.datasets.functional.wavcaps import (
    WavCapsCard,
    WavCapsSource,
    WavCapsSubset,
    _get_audio_subset_dpath,
    download_wavcaps_dataset,
    load_wavcaps_dataset,
)
from aac_datasets.utils.globals import _get_root, _get_zip_path

pylog = logging.getLogger(__name__)


class WavCapsItem(TypedDict):
    # Common attributes
    audio: Tensor
    captions: List[str]
    dataset: str
    fname: str
    index: int
    subset: WavCapsSubset
    sr: int
    duration: float
    # WavCaps-specific attributes
    author: Optional[str]  # FSD and SB
    description: Optional[str]  # BBC, FSD and SB only
    duration: float
    download_link: Optional[str]  # BBC, FSD and SB only
    href: Optional[str]  # FSD and SB only
    id: str
    source: WavCapsSource
    tags: List[str]  # FSD only


class WavCaps(AACDataset[WavCapsItem]):
    r"""Unofficial WavCaps PyTorch dataset.

    WavCaps Paper : https://arxiv.org/pdf/2303.17395.pdf
    HuggingFace source : https://huggingface.co/datasets/cvssp/WavCaps

    This dataset contains 4 training subsets, extracted from different sources:
    - BBC Sound Effects "bbc"
    - SoundBible "soundbible"
    - AudioSet strongly labeled without AudioCaps V1 val and test subsets "audioset_no_audiocaps_v1"
    - FreeSound without Clotho dev, val, eval and test subsets "freesound_no_clotho_v2"

    Other subsets exists but they does not comply DCASE Challenge rules:
    - AudioSet strongly labeled "audioset"
    - FreeSound "freesound"

    .. warning::
        WavCaps download is experimental ; it requires a lot of disk space and can take very long time to download and extract, so you might expect errors.

    .. code-block:: text
        :caption:  Dataset folder tree

        {root}
        └── WavCaps
            ├── Audio
            │   ├── AudioSet_SL
            │   │    └── (108317 flac files, ~64GB)
            │   ├── BBC_Sound_Effects
            │   │    └── (31201 flac files, ~142GB)
            │   ├── FreeSound
            │   │    └── (262300 flac files, ~1.4TB)
            │   └── SoundBible
            │        └── (1232 flac files, ~884MB)
            ├── Zip_files
            │   ├── AudioSet_SL
            │   │    └── (8 zip files, ~76GB)
            │   ├── BBC_Sound_Effects
            │   │    └── (26 zip files, ~562GB)
            │   ├── FreeSound
            │   │    └── (123 zip? files, ~1.4TB)
            │   └── SoundBible
            │        └── (1 zip? files, ~624GB)
            ├── json_files
            │    ├── AudioSet_SL
            │    │    └── as_final.json
            │    ├── BBC_Sound_Effects
            │    │    └── bbc_final.json
            │    ├── FreeSound
            │    │    ├── fsd_final_2s.json
            │    │    └── fsd_final.json
            │    ├── SoundBible
            │    │    └── sb_final.json
            │    └── blacklist
            │         ├── blacklist_exclude_all_ac.json
            │         ├── blacklist_exclude_test_ac.json
            │         └── blacklist_exclude_ubs8k_esc50_vggsound.json
            ├── .gitattributes
            └── README.md
    """

    # Common globals
    CARD: ClassVar[WavCapsCard] = WavCapsCard()

    def __init__(
        self,
        # Common args
        root: Union[str, Path, None] = None,
        subset: WavCapsSubset = WavCapsCard.DEFAULT_SUBSET,
        download: bool = False,
        transform: Optional[Callable[[WavCapsItem], Any]] = None,
        verbose: int = 0,
        force_download: bool = False,
        verify_files: bool = False,
        *,
        # WavCaps-specific args
        clean_archives: bool = False,
        hf_cache_dir: Optional[str] = None,
        repo_id: Optional[str] = None,
        revision: Optional[str] = WavCapsCard.DEFAULT_REVISION,
        zip_path: Union[str, Path, None] = None,
    ) -> None:
        """
        :param root: The parent of the dataset root directory.
            The data will be stored in the 'MACS' subdirectory.
            defaults to ".".
        :param subset: The subset of the dataset. Can be one of :attr:`~WavCapsCard.SUBSETS`.
            defaults to "audioset".
        :param download: Download the dataset if download=True and if the dataset is not already downloaded.
            defaults to False.
        :param transform: The transform to apply to the global dict item. This transform is applied only in getitem method when argument is an integer.
            defaults to None.
        :param verbose: Verbose level. Can be 0 or 1.
            defaults to 0.
        :param force_download: If True, force to re-download file even if they exists on disk.
            defaults to False.
        :param verify_files: If True, check hash value when possible.
            defaults to False.

        :param clean_archives: If True, remove the compressed archives from disk to save space.
            defaults to False.
        :param hf_cache_dir: HuggingFace cache directory. If None, use the global value :variable:`~huggingface_hub.constants.HUGGINGFACE_HUB_CACHE`.
            defaults to None.
        :param repo_id: Repository ID on HuggingFace.
            defaults to "cvssp/WavCaps".
        :param revision: The HuggingFace revision tag.
            defaults to :attr:`~WavCapsCard.DEFAULT_REVISION`.
        :param zip_path: Path to zip executable path in shell.
            defaults to "zip".
        """
        if subset not in WavCapsCard.SUBSETS:
            msg = f"Invalid argument {subset=} for {WavCapsCard.PRETTY_NAME}. (expected one of {WavCapsCard.SUBSETS})"
            raise ValueError(msg)

        root = _get_root(root)
        zip_path = _get_zip_path(zip_path)

        if download:
            download_wavcaps_dataset(
                root=root,
                subset=subset,
                force=force_download,
                verbose=verbose,
                clean_archives=clean_archives,
                hf_cache_dir=hf_cache_dir,
                repo_id=repo_id,
                revision=revision,
                verify_files=verify_files,
                zip_path=zip_path,
            )

        raw_data = load_wavcaps_dataset(
            root=root,
            subset=subset,
            verbose=verbose,
            hf_cache_dir=hf_cache_dir,
            revision=revision,
        )

        size = len(next(iter(raw_data.values())))
        raw_data["dataset"] = [WavCapsCard.NAME] * size
        raw_data["subset"] = [subset] * size
        raw_data["fpath"] = [
            osp.join(
                _get_audio_subset_dpath(
                    root, hf_cache_dir, revision, raw_data["source"][i]
                ),
                fname,
            )
            for i, fname in enumerate(raw_data["fname"])
        ]
        raw_data["index"] = list(range(size))

        column_names = list(WavCapsItem.__required_keys__) + list(  # type: ignore
            WavCapsItem.__optional_keys__  # type: ignore
        )

        super().__init__(
            raw_data=raw_data,
            transform=transform,
            column_names=column_names,
            flat_captions=False,
            sr=WavCapsCard.SAMPLE_RATE,
            verbose=verbose,
        )
        self._root = root
        self._subset: WavCapsSubset = subset
        self._download = download
        self._hf_cache_dir = hf_cache_dir
        self._revision = revision

        self.add_online_columns(
            {
                "audio": WavCaps._load_audio,
                "audio_metadata": WavCaps._load_audio_metadata,
                "duration": WavCaps._load_duration,
                "num_channels": WavCaps._load_num_channels,
                "num_frames": WavCaps._load_num_frames,
                "sr": WavCaps._load_sr,
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
    def subset(self) -> WavCapsSubset:
        return self._subset
