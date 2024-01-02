#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import logging
import os.path as osp

from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Union,
)

from torch import Tensor
from typing_extensions import TypedDict

from aac_datasets.datasets.base import AACDataset
from aac_datasets.datasets.functional.macs import (
    MACSCard,
    load_macs_dataset,
    download_macs_dataset,
    _get_audio_dpath,
)
from aac_datasets.utils.globals import _get_root


pylog = logging.getLogger(__name__)


class MACSItem(TypedDict):
    r"""Dataclass representing a single MACS item."""

    # Common attributes
    audio: Tensor
    captions: List[str]
    dataset: str
    fname: str
    index: int
    subset: str
    sr: int
    # MACS-specific attributes
    annotators_ids: List[str]
    competences: List[float]
    identifier: str
    scene_label: str
    tags: List[List[str]]


class MACS(AACDataset[MACSItem]):
    r"""Unofficial MACS PyTorch dataset.

    .. code-block:: text
        :caption: Dataset folder tree

        {root}
        └── MACS
            ├── audio
            │    └── (3930 wav files, ~13GB)
            ├── LICENCE.txt
            ├── MACS.yaml
            ├── MACS_competence.csv
            └── tau_meta
                ├── fold1_evaluate.csv
                ├── fold1_test.csv
                ├── fold1_train.csv
                └── meta.csv
    """

    # Common globals
    CARD: ClassVar[MACSCard] = MACSCard()

    # Initialization
    def __init__(
        self,
        # Common args
        root: Union[str, Path, None] = None,
        subset: str = MACSCard.DEFAULT_SUBSET,
        download: bool = False,
        transform: Optional[Callable] = None,
        verbose: int = 0,
        force_download: bool = False,
        verify_files: bool = False,
        *,
        # MACS-specific args
        clean_archives: bool = True,
        flat_captions: bool = False,
    ) -> None:
        """
        :param root: The parent of the dataset root directory.
            The data will be stored in the 'MACS' subdirectory.
            defaults to ".".
        :param subset: The subset of the dataset. This parameter is here only to accept the same interface than the other datasets.
            The only valid subset is "full" and other values will raise a ValueError.
            defaults to "full".
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
        """
        if subset not in MACSCard.SUBSETS:
            raise ValueError(
                f"Invalid argument subset={subset} for MACS. (expected one of {MACSCard.SUBSETS})"
            )

        root = _get_root(root)

        if download:
            download_macs_dataset(
                root=root,
                subset=subset,
                force=force_download,
                verbose=verbose,
                clean_archives=clean_archives,
                verify_files=verify_files,
            )

        raw_data, annotator_id_to_competence = load_macs_dataset(
            root=root,
            subset=subset,
            verbose=verbose,
        )

        audio_dpath = _get_audio_dpath(root)
        size = len(next(iter(raw_data.values())))
        raw_data["dataset"] = [MACSCard.NAME] * size
        raw_data["subset"] = [subset] * size
        raw_data["fpath"] = [
            osp.join(audio_dpath, fname) for fname in raw_data["fname"]
        ]
        raw_data["index"] = list(range(size))

        super().__init__(
            raw_data=raw_data,
            transform=transform,
            column_names=MACSItem.__required_keys__,
            flat_captions=flat_captions,
            sr=MACSCard.SAMPLE_RATE,
            verbose=verbose,
        )
        self._root = root
        self._subset = subset
        self._download = download
        self._transform = transform
        self._flat_captions = flat_captions
        self._verbose = verbose
        self._annotator_id_to_competence = annotator_id_to_competence

        self.add_online_columns(
            {
                "audio": MACS._load_audio,
                "audio_metadata": MACS._load_audio_metadata,
                "duration": MACS._load_duration,
                "num_channels": MACS._load_num_channels,
                "num_frames": MACS._load_num_frames,
                "sr": MACS._load_sr,
                "competences": MACS._load_competences,
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

    # Public methods
    def get_annotator_id_to_competence_dict(self) -> Dict[int, float]:
        """Get annotator to competence dictionary."""
        # Note : copy to prevent any changes on this attribute
        return copy.deepcopy(self._annotator_id_to_competence)

    def get_competence(self, annotator_id: int) -> float:
        """Get competence value for a specific annotator id."""
        return self._annotator_id_to_competence[annotator_id]

    def _load_competences(self, idx: int) -> List[float]:
        annotators_ids: List[int] = self.at(idx, "annotators_ids")
        competences = [self.get_competence(id_) for id_ in annotators_ids]
        return competences

    # Magic methods
    def __repr__(self) -> str:
        repr_dic = {
            "subset": self._subset,
            "size": len(self),
            "num_columns": len(self.column_names),
        }
        repr_str = ", ".join(f"{k}={v}" for k, v in repr_dic.items())
        return f"{MACSCard.PRETTY_NAME}({repr_str})"
