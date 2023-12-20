#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp

from pathlib import Path
from typing import Callable, ClassVar, List, Optional, Tuple, Union

from torch import Tensor
from typing_extensions import TypedDict

from aac_datasets.datasets.base import AACDataset
from aac_datasets.datasets.functional.wavcaps import (
    WavCapsCard,
    load_wavcaps_dataset,
    prepare_wavcaps_dataset,
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
    subset: str
    sr: int
    # WavCaps-specific attributes
    author: Optional[str]  # FSD and SB
    description: Optional[str]  # BBC, FSD and SB only
    duration: float
    download_link: Optional[str]  # BBC, FSD and SB only
    href: Optional[str]  # FSD and SB only
    id: str
    source: str
    tags: List[str]  # FSD only


class WavCaps(AACDataset[WavCapsItem]):
    r"""Unofficial WavCaps PyTorch dataset.

    WavCaps Paper : https://arxiv.org/pdf/2303.17395.pdf
    HuggingFace source : https://huggingface.co/datasets/cvssp/WavCaps

    This dataset contains 4 training subsets, extracted from different sources:
    - AudioSet strongly labeled (as)
    - BBC Sound Effects (bbc)
    - FreeSound (fsd)
    - SoundBible (sb)
    - AudioSet strongly labeled without AudioCaps (as_noac)
    - FreeSound without Clotho (fsd_nocl)

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
    FORCE_PREPARE_DATA: ClassVar[bool] = False
    VERIFY_FILES: ClassVar[bool] = False

    # WavCaps-specific globals
    CLEAN_ARCHIVES: ClassVar[bool] = False
    RESUME_DL: ClassVar[bool] = True
    SIZE_CATEGORIES: Tuple[str, ...] = ("100K<n<1M",)

    def __init__(
        self,
        # Common args
        root: Union[str, Path, None] = None,
        subset: str = "as_noac",
        download: bool = False,
        transform: Optional[Callable] = None,
        verbose: int = 0,
        # WavCaps-specific args
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
            defaults to "as".
        :param download: Download the dataset if download=True and if the dataset is not already downloaded.
            defaults to False.
        :param transform: The transform to apply to the global dict item. This transform is applied only in getitem method when argument is an integer.
            defaults to None.
        :param verbose: Verbose level. Can be 0 or 1.
            defaults to 0.

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
            raise ValueError(
                f"Invalid argument subset={subset} for {WavCapsCard.PRETTY_NAME}. (expected one of {WavCapsCard.SUBSETS})"
            )

        root = _get_root(root)
        zip_path = _get_zip_path(zip_path)

        if download:
            prepare_wavcaps_dataset(
                root=root,
                subset=subset,
                revision=revision,
                hf_cache_dir=hf_cache_dir,
                repo_id=repo_id,
                resume_dl=WavCaps.RESUME_DL,
                force=WavCaps.FORCE_PREPARE_DATA,
                verify_files=WavCaps.VERIFY_FILES,
                clean_archives=WavCaps.CLEAN_ARCHIVES,
                zip_path=zip_path,
                verbose=verbose,
            )

        raw_data = load_wavcaps_dataset(
            root=root,
            subset=subset,
            verbose=verbose,
            hf_cache_dir=hf_cache_dir,
            repo_id=repo_id,
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

        super().__init__(
            raw_data=raw_data,
            transform=transform,
            column_names=WavCapsItem.__required_keys__,
            flat_captions=False,
            sr=WavCapsCard.SAMPLE_RATE,
            verbose=verbose,
        )
        self._root = root
        self._subset = subset
        self._download = download
        self._hf_cache_dir = hf_cache_dir
        self._revision = revision

        self.add_post_columns(
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
    def subset(self) -> str:
        return self._subset
