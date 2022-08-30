#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import csv
import logging
import os
import os.path as osp
import shutil
import zipfile

from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
import torchaudio
import yaml

from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data.dataset import Dataset
from torchaudio.datasets.utils import validate_file


logger = logging.getLogger(__name__)


@dataclass
class MACSItem:
    """Dataclass representing a single MACS item."""

    # Common attributes
    audio: Tensor = torch.empty((0,))
    captions: List[str] = field(default_factory=list)
    dataset: str = "macs"
    fname: str = "unknown"
    index: int = -1
    subset: str = "unknown"
    sr: int = -1
    # MACS-specific attributes
    annotators_ids: List[str] = field(default_factory=list)
    competences: List[float] = field(default_factory=list)
    identifier: str = "unknown"
    scene_label: str = "unknown"
    tags: List[List[str]] = field(default_factory=list)


class MACS(Dataset):
    r"""Unofficial MACS pytorch dataset.

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

    # Global
    AUDIO_MAX_SIZE = 480001
    AUDIO_MIN_SIZE = 479999
    AUDIO_N_CHANNELS = 2
    CLEAN_ARCHIVES: bool = False
    FORCE_PREPARE_DATA: bool = False
    MAX_CAPTIONS_PER_AUDIO = {"full": 5}
    MIN_CAPTIONS_PER_AUDIO = {"full": 2}
    SAMPLE_RATE = 48000  # in Hz
    SUBSETS = ("full",)

    def __init__(
        self,
        root: str = ".",
        subset: str = "full",
        download: bool = False,
        item_transform: Optional[Callable] = None,
        unfold: bool = False,
        verbose: int = 0,
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
        :param item_transform: The transform to apply to the global dict item. This transform is applied AFTER each field transform.
            defaults to None.
        :param unfold: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param verbose: Verbose level to use. Can be 0 or 1.
            defaults to 0.
        """
        if subset not in self.SUBSETS:
            raise ValueError(
                f"Invalid argument {subset=} for MACS. (expected one of {self.SUBSETS})"
            )

        super().__init__()
        self.__root = root
        self.__subset = subset
        self.__download = download
        self.__item_transform = item_transform
        self.__unfold = unfold
        self.__verbose = verbose

        self.__annotator_id_to_competence = {}
        self.__all_items = {}
        self.__is_loaded = False

        if self.__download:
            self._prepare_data()
        self._load_data()

    def get_field(
        self, key: str, index: Union[int, slice, Iterable[int]] = slice(None)
    ) -> Any:
        """Get a specific data field.

        :param key: The name of the field. Can be any attribute name of :class:`~MACSItem`.
        :param index: The index or slice of the value in range [0, len(dataset)-1].
        :returns: The field value. The type depends of the transform applied to the field.
        """
        if isinstance(index, (int, slice)) and key in self.__all_items.keys():
            return self.__all_items[key][index]

        if isinstance(index, slice):
            index = range(len(self))[index]

        if isinstance(index, Iterable):
            return [self.get_field(key, idx) for idx in index]

        if key == "audio":
            fpath = self.get_field("fpath", index)
            audio, sr = torchaudio.load(fpath)  # type: ignore

            # Sanity check
            if audio.nelement() == 0:
                raise RuntimeError(
                    f"Invalid audio number of elements in {fpath}. (expected {audio.nelement()=} > 0)"
                )
            if sr != self.SAMPLE_RATE:
                raise RuntimeError(
                    f"Invalid sample rate in {fpath}. (expected {self.SAMPLE_RATE} but found {sr=})"
                )
            return audio

        elif key == "audio_metadata":
            fpath = self.get_field("fpath", index)
            audio_metadata = torchaudio.info(fpath)  # type: ignore
            return audio_metadata

        elif key == "competences":
            annotators_ids = self.get_field("annotators_ids", index)
            competences = [self.get_competence(id_) for id_ in annotators_ids]
            return competences

        elif key == "dataset":
            return "macs"

        elif key == "fpath":
            fname = self.get_field("fname", index)
            fpath = osp.join(self._dpath_audio, fname)
            return fpath

        elif key == "index":
            return index

        elif key == "num_channels":
            audio_metadata = self.get_field("audio_metadata", index)
            return audio_metadata.num_channels

        elif key == "num_frames":
            audio_metadata = self.get_field("audio_metadata", index)
            return audio_metadata.num_frames

        elif key == "sr":
            audio_metadata = self.get_field("audio_metadata", index)
            return audio_metadata.sample_rate

        elif key == "subset":
            return self.__subset

        else:
            keys = [field.name for field in fields(MACSItem)]
            raise ValueError(
                f"Invalid argument {key=} at {index=}. (expected one of {tuple(keys)})"
            )

    def get_competence(self, annotator_id: int) -> float:
        """Get competence value for a specific annotator id."""
        return self.__annotator_id_to_competence[annotator_id]

    def get_annotator_to_competence_dict(self) -> Dict[int, float]:
        """Get annotator to competence dictionary."""
        # Note : copy to prevent any changes on this attribute
        return copy.deepcopy(self.__annotator_id_to_competence)

    @cached_property
    def _dpath_archives(self) -> str:
        return osp.join(self._dpath_data, "archives")

    @cached_property
    def _dpath_audio(self) -> str:
        return osp.join(self._dpath_data, "audio")

    @cached_property
    def _dpath_data(self) -> str:
        return osp.join(self.__root, "MACS")

    @cached_property
    def _dpath_tau_meta(self) -> str:
        return osp.join(self._dpath_data, "tau_meta")

    def _is_loaded(self) -> bool:
        return self.__is_loaded

    def _is_prepared(self) -> bool:
        if not osp.isdir(self._dpath_audio):
            return False
        captions_fpath = osp.join(self._dpath_data, MACS_FILES["captions"]["fname"])
        if not osp.isfile(captions_fpath):
            return False

        with open(captions_fpath, "r") as file:
            data = yaml.safe_load(file)
        data = data["files"]
        fnames = os.listdir(self._dpath_audio)
        return len(data) == len(fnames)

    def _load_data(self) -> None:
        if not self._is_prepared():
            raise RuntimeError(
                f"Cannot load data: macs is not prepared in data root={self.__root}. Please use download=True in dataset constructor."
            )

        # Read data files
        captions_fname = MACS_FILES["captions"]["fname"]
        captions_fpath = osp.join(self._dpath_data, captions_fname)
        if self.__verbose >= 2:
            logger.debug(f"Reading captions file {captions_fname}...")

        with open(captions_fpath, "r") as file:
            caps_data = yaml.safe_load(file)

        tau_meta_fname = "meta.csv"
        tau_meta_fpath = osp.join(self._dpath_tau_meta, tau_meta_fname)
        if self.__verbose >= 2:
            logger.debug(
                f"Reading Tau Urban acoustic scene meta file {tau_meta_fname}..."
            )

        with open(tau_meta_fpath, "r") as file:
            reader = csv.DictReader(file, delimiter="\t")
            tau_tags_data = list(reader)

        competence_fname = "MACS_competence.csv"
        competence_fpath = osp.join(self._dpath_data, competence_fname)
        if self.__verbose >= 2:
            logger.debug(f"Reading file {competence_fname}...")

        with open(competence_fpath, "r") as file:
            reader = csv.DictReader(file, delimiter="\t")
            competences_data = list(reader)

        # Store MACS data
        all_items: Dict[str, List[Any]] = {
            "fname": [item["filename"] for item in caps_data["files"]],
            "captions": [
                [subitem["sentence"] for subitem in item["annotations"]]
                for item in caps_data["files"]
            ],
            "tags": [
                [subitem["tags"] for subitem in item["annotations"]]
                for item in caps_data["files"]
            ],
            "annotators_ids": [
                [subitem["annotator_id"] for subitem in item["annotations"]]
                for item in caps_data["files"]
            ],
        }
        dataset_size = len(all_items["fname"])

        # Build global mappings
        fname_to_idx = {fname: i for i, fname in enumerate(all_items["fname"])}
        annotator_id_to_competence = {
            int(annotator["annotator_id"]): float(annotator["competence"])
            for annotator in competences_data
        }

        # Store TAU Urban acoustic scenes data
        tau_additional_keys = ("scene_label", "identifier")
        all_items |= {
            key: [None for _ in range(dataset_size)] for key in tau_additional_keys
        }
        tau_meta_fpath = osp.join(self._dpath_tau_meta, "meta.csv")
        for tau_tags in tau_tags_data:
            fname = osp.basename(tau_tags["filename"])
            if fname in fname_to_idx:
                idx = fname_to_idx[fname]
                for key in tau_additional_keys:
                    all_items[key][idx] = tau_tags[key]

        if self.__unfold and self.MIN_CAPTIONS_PER_AUDIO[self.__subset] > 1:
            all_infos_unfolded = {key: [] for key in all_items.keys()}

            for i, captions in enumerate(all_items["captions"]):
                for caption in captions:
                    for key in all_items.keys():
                        all_infos_unfolded[key].append(all_items[key][i])
                    all_infos_unfolded["captions"] = [caption]

            all_items = all_infos_unfolded

        # Sanity checks
        assert all(
            all(value is not None for value in all_items[key])
            for key in tau_additional_keys
        )
        assert all(len(values) == dataset_size for values in all_items.values())

        # Set attributes
        self.__all_items = all_items
        self.__annotator_id_to_competence = annotator_id_to_competence
        self.__is_loaded = True

        if self.__verbose >= 1:
            logger.info(f"{repr(self)} has been loaded. (len={len(self)})")

    def _prepare_data(self) -> None:
        if not osp.isdir(self.__root):
            raise RuntimeError(f"Cannot find root directory '{self.__root}'.")

        os.makedirs(self._dpath_archives, exist_ok=True)
        os.makedirs(self._dpath_audio, exist_ok=True)
        os.makedirs(self._dpath_tau_meta, exist_ok=True)

        # Download MACS specific files
        for file_info in MACS_FILES.values():
            dpath = self._dpath_data
            fname = file_info["fname"]
            fpath = osp.join(dpath, fname)

            if not osp.isfile(fpath) or self.FORCE_PREPARE_DATA:
                if self.__verbose >= 1:
                    logger.info(f"Downloading captions file '{fname}'...")

                url = file_info["url"]
                download_url_to_file(
                    url,
                    fpath,
                    progress=self.__verbose >= 1,
                )

            hash_value = file_info["hash_value"]
            with open(fpath, "rb") as file:
                valid = validate_file(file, hash_value, hash_type="md5")
            if not valid:
                raise RuntimeError(f"Invalid checksum for file {fname}.")

        captions_fpath = osp.join(self._dpath_data, MACS_FILES["captions"]["fname"])
        with open(captions_fpath, "r") as file:
            captions_data = yaml.safe_load(file)
        captions_data = captions_data["files"]

        # Download TAU Urban Sound audio archives files
        for i, file_info in enumerate(MACS_ARCHIVES_FILES.values()):
            dpath = self._dpath_archives
            zip_fname = file_info["fname"]
            zip_fpath = osp.join(dpath, zip_fname)

            if not osp.isfile(zip_fpath) or self.FORCE_PREPARE_DATA:
                if self.__verbose >= 1:
                    logger.info(
                        f"Downloading audio zip file '{zip_fpath}'... ({i+1}/{len(MACS_ARCHIVES_FILES)})"
                    )

                url = file_info["url"]
                download_url_to_file(
                    url,
                    zip_fpath,
                    progress=self.__verbose >= 1,
                )

            hash_value = file_info["hash_value"]
            with open(zip_fpath, "rb") as file:
                valid = validate_file(file, hash_value, hash_type="md5")
            if not valid:
                raise RuntimeError(f"Invalid checksum for file {zip_fname}.")

        # Extract files from TAU Urban Sound archives
        macs_fnames = dict.fromkeys(data["filename"] for data in captions_data)
        for i, (name, file_info) in enumerate(MACS_ARCHIVES_FILES.items()):
            zip_fname = file_info["fname"]
            zip_fpath = osp.join(self._dpath_archives, zip_fname)

            if self.__verbose >= 2:
                logger.debug(
                    f"Check to extract TAU Urban acoustic scenes archive {zip_fname=}..."
                )

            is_audio_archive = name.startswith("audio")
            target_dpath = (
                self._dpath_audio if is_audio_archive else self._dpath_tau_meta
            )

            with zipfile.ZipFile(zip_fpath, "r") as file:
                members_to_extract = [
                    member
                    for member in file.namelist()
                    # Extract member if file if in captions yaml file and if the audio file is not already downloaded
                    if (
                        (osp.basename(member) in macs_fnames or not is_audio_archive)
                        and not osp.isfile(osp.join(target_dpath, osp.basename(member)))
                    )
                ]

                if self.__verbose >= 1:
                    logger.info(
                        f"Extracting {len(members_to_extract)}/{len(file.namelist())} audio files from ZIP file '{zip_fname}'... ({i+1}/{len(MACS_ARCHIVES_FILES)})"
                    )

                if len(members_to_extract) > 0:
                    file.extractall(self._dpath_archives, members_to_extract)
                    for member in members_to_extract:
                        extracted_fpath = osp.join(self._dpath_archives, member)
                        target_fpath = osp.join(target_dpath, osp.basename(member))
                        shutil.move(extracted_fpath, target_fpath)

        if self.CLEAN_ARCHIVES:
            if self.__verbose >= 1:
                logger.info(f"Removing archives files in {self._dpath_archives}...")
            shutil.rmtree(self._dpath_archives, ignore_errors=True)

        audio_fnames = [
            name for name in os.listdir(self._dpath_audio) if name.endswith(".wav")
        ]
        audio_fpaths = [osp.join(self._dpath_audio, name) for name in audio_fnames]
        assert len(audio_fpaths) == len(macs_fnames)

        if self.__verbose >= 1:
            logger.info(
                f"{len(audio_fpaths)} audio files has been prepared for MACS dataset."
            )

    def __getitem__(self, index: Union[int, slice]) -> Dict[str, Any]:
        keys = [field.name for field in fields(MACSItem)]
        item = {key: self.get_field(key, index) for key in keys}
        if self.__item_transform is not None:
            item = self.__item_transform(item)
        return item

    def __len__(self) -> int:
        return len(self.__all_items["captions"])

    def __repr__(self) -> str:
        return f"MACS(size={len(self)})"


# MACS-specific files links.
MACS_FILES = {
    "licence": {
        "fname": "LICENSE.txt",
        "url": "https://zenodo.org/record/5114771/files/LICENSE.txt?download=1",
        "hash_value": "d3086f4517cccc32c1bb3a081b07cfa1",
    },
    "captions": {
        "fname": "MACS.yaml",
        "url": "https://zenodo.org/record/5114771/files/MACS.yaml?download=1",
        "hash_value": "23fcb2ebd0b109094034ef9e87972256",
    },
    "annotators_competences": {
        "fname": "MACS_competence.csv",
        "url": "https://zenodo.org/record/5114771/files/MACS_competence.csv?download=1",
        "hash_value": "4dfe9f951f0af9f29cb7952ec030370a",
    },
}

# TAU_URBAN_ACOUSTIC archives files links.
TAU_URBAN_ACOUSTIC_DEV_FILES = {
    "audio.1": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.1.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.1.zip?download=1",
        "hash_value": "aca4ebfd9ed03d5f747d6ba8c24bc728",
    },
    "audio.10": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.10.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.10.zip?download=1",
        "hash_value": "0ffbf60006da520cc761fb74c878b98b",
    },
    "audio.11": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.11.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.11.zip?download=1",
        "hash_value": "599055d93b4c11057c29be2df54538d4",
    },
    "audio.12": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.12.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.12.zip?download=1",
        "hash_value": "98b8d162ff3665695c4c910e6c372cc8",
    },
    "audio.13": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.13.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.13.zip?download=1",
        "hash_value": "a356c08b1a5a21d433eba37ef87587f4",
    },
    "audio.14": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.14.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.14.zip?download=1",
        "hash_value": "f8969771e7faf7dd471d1cf78b0cf011",
    },
    "audio.15": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.15.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.15.zip?download=1",
        "hash_value": "4758c4b0fb7484faa632266e78850820",
    },
    "audio.16": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.16.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.16.zip?download=1",
        "hash_value": "a18acad9ede8ea76574216feb887f0bc",
    },
    "audio.17": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.17.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.17.zip?download=1",
        "hash_value": "1af7703484632f340da5c33662dc9632",
    },
    "audio.18": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.18.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.18.zip?download=1",
        "hash_value": "b67402bf3e08f4da394a7c18756c0fd2",
    },
    "audio.19": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.19.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.19.zip?download=1",
        "hash_value": "035db315f19106eb848b6f9b32bcc47c",
    },
    "audio.2": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.2.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.2.zip?download=1",
        "hash_value": "c4f170408ce77c8c70c532bf268d7be0",
    },
    "audio.20": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.20.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.20.zip?download=1",
        "hash_value": "9cb28c74911bf8a3eadcf53f50a5b5d6",
    },
    "audio.21": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.21.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.21.zip?download=1",
        "hash_value": "0e44ed85c88ec036a9725b4dd1dfaea0",
    },
    "audio.3": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.3.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.3.zip?download=1",
        "hash_value": "c7214a07211f10f3250290d05e72c37e",
    },
    "audio.4": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.4.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.4.zip?download=1",
        "hash_value": "a6a62110f6699cf4432072acb1dffda6",
    },
    "audio.5": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.5.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.5.zip?download=1",
        "hash_value": "091a0b6d3c84b8e60e46940aa7d4a8a0",
    },
    "audio.6": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.6.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.6.zip?download=1",
        "hash_value": "114f4ca13e074391b98a1cfd8140de65",
    },
    "audio.7": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.7.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.7.zip?download=1",
        "hash_value": "5951dd2968f7a514e2afbe279c4f060d",
    },
    "audio.8": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.8.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.8.zip?download=1",
        "hash_value": "b0b63dc95b327e1509857c8d8a663cc3",
    },
    "audio.9": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.9.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.9.zip?download=1",
        "hash_value": "3c32a693a6b111ffb957be3c1dd22e9b",
    },
    "doc": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.doc.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.doc.zip?download=1",
        "hash_value": "1f6879544e80da70099a191613e7e51f",
    },
    "meta": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.meta.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.meta.zip?download=1",
        "hash_value": "09782f2097e4735687af73c44919329c",
    },
}

# List of TAU_URBAN_ACOUSTIC archives containing at least 1 MACS audio.
MACS_ARCHIVES_FILES = {
    name: TAU_URBAN_ACOUSTIC_DEV_FILES[name]
    for name in (
        "audio.1",
        "audio.10",
        "audio.11",
        "audio.12",
        "audio.13",
        "audio.2",
        "audio.3",
        "audio.9",
        "meta",
    )
}
