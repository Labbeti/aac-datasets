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
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

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


class MACS(Dataset[Dict[str, Any]]):
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

    # Global
    AUDIO_MAX_SEC = 10.000020833333334
    AUDIO_MAX_SIZE = 480001
    AUDIO_MIN_SEC = 9.999979166666666
    AUDIO_MIN_SIZE = 479999
    AUDIO_N_CHANNELS = 2
    CLEAN_ARCHIVES: bool = False
    FORCE_PREPARE_DATA: bool = False
    MAX_CAPTIONS_PER_AUDIO = {"full": 5}
    MIN_CAPTIONS_PER_AUDIO = {"full": 2}
    SAMPLE_RATE = 48000  # in Hz
    SUBSETS = ("full",)

    # Initialization
    def __init__(
        self,
        root: str = ".",
        subset: str = "full",
        download: bool = False,
        transform: Optional[Callable] = None,
        flat_captions: bool = False,
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
        :param transform: The transform to apply to the global dict item. This transform is applied only in getitem method.
            defaults to None.
        :param flat_captions: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param verbose: Verbose level to use. Can be 0 or 1.
            defaults to 0.
        """
        if subset not in self.SUBSETS:
            raise ValueError(
                f"Invalid argument subset={subset} for MACS. (expected one of {self.SUBSETS})"
            )

        super().__init__()
        self._root = root
        self._subset = subset
        self._download = download
        self._transform = transform
        self._flat_captions = flat_captions
        self._verbose = verbose

        self._annotator_id_to_competence = {}
        self._all_items = {}
        self._loaded = False

        if self._download:
            self.__prepare_data()
        self.__load_data()

    # Properties
    @property
    def column_names(self) -> List[str]:
        """The name of each column of the dataset."""
        return [field.name for field in fields(MACSItem)]

    @property
    def info(self) -> Dict[str, Any]:
        """Return the global dataset info."""
        return {
            "dataset": "macs",
            "subset": self._subset,
        }

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the MACS dataset."""
        return len(self), len(self.column_names)

    # Public methods
    def at(
        self,
        idx: Union[int, Iterable[int], None, slice] = None,
        column: Union[str, Iterable[str], None] = None,
    ) -> Any:
        """Get a specific data field.

        :param index: The index or slice of the value in range [0, len(dataset)-1].
        :param column: The name(s) of the column. Can be any value of :meth:`~MACS.column_names`.
        :returns: The field value. The type depends of the column.
        """
        if idx is None:
            idx = slice(None)
        if column is None:
            column = self.column_names

        if not isinstance(column, str) and isinstance(column, Iterable):
            return {column_i: self.at(idx, column_i) for column_i in column}

        if isinstance(idx, (int, slice)) and column in self._all_items.keys():
            return self._all_items[column][idx]

        if isinstance(idx, slice):
            idx = range(len(self))[idx]

        if isinstance(idx, Iterable):
            idx = list(idx)
            if not all(isinstance(idx_i, int) for idx_i in idx):
                raise TypeError(
                    f"Invalid input type for idx={idx}. (expected Iterable[int], not Iterable[{idx.__class__.__name__}])"
                )
            return [self.at(idx_i, column) for idx_i in idx]

        if column == "audio":
            fpath = self.at(idx, "fpath")
            audio, sr = torchaudio.load(fpath)  # type: ignore

            # Sanity check
            if audio.nelement() == 0:
                raise RuntimeError(
                    f"Invalid audio number of elements in {fpath}. (expected audio.nelement()={audio.nelement()} > 0)"
                )
            if sr != self.SAMPLE_RATE:
                raise RuntimeError(
                    f"Invalid sample rate in {fpath}. (expected {self.SAMPLE_RATE} but found sr={sr})"
                )
            return audio

        elif column == "audio_metadata":
            fpath = self.at(idx, "fpath")
            audio_metadata = torchaudio.info(fpath)  # type: ignore
            return audio_metadata

        elif column == "competences":
            annotators_ids = self.at(idx, "annotators_ids")
            competences = [self.get_competence(id_) for id_ in annotators_ids]
            return competences

        elif column == "dataset":
            return "macs"

        elif column == "fpath":
            fname = self.at(idx, "fname")
            fpath = osp.join(self.__dpath_audio, fname)
            return fpath

        elif column == "index":
            return idx

        elif column == "num_channels":
            audio_metadata = self.at(idx, "audio_metadata")
            return audio_metadata.num_channels

        elif column == "num_frames":
            audio_metadata = self.at(idx, "audio_metadata")
            return audio_metadata.num_frames

        elif column == "sr":
            audio_metadata = self.at(idx, "audio_metadata")
            return audio_metadata.sample_rate

        elif column == "subset":
            return self._subset

        else:
            raise ValueError(
                f"Invalid argument column={column} at idx={idx}. (expected one of {tuple(self.column_names)})"
            )

    def get_annotator_id_to_competence_dict(self) -> Dict[int, float]:
        """Get annotator to competence dictionary."""
        # Note : copy to prevent any changes on this attribute
        return copy.deepcopy(self._annotator_id_to_competence)

    def get_competence(self, annotator_id: int) -> float:
        """Get competence value for a specific annotator id."""
        return self._annotator_id_to_competence[annotator_id]

    def is_loaded(self) -> bool:
        """Returns True if the dataset is loaded."""
        return self._loaded

    def set_transform(
        self,
        transform: Optional[Callable[[Dict[str, Any]], Any]],
    ) -> None:
        """Set the transform applied to each row."""
        self._transform = transform

    # Magic methods
    def __getitem__(
        self,
        idx: Any,
    ) -> Dict[str, Any]:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None

        item = self.at(idx, column)
        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return len(self._all_items["captions"])

    def __repr__(self) -> str:
        return f"MACS(size={len(self)}, subset={self._subset}, num_columns={len(self.column_names)})"

    # Private methods
    @property
    @lru_cache()
    def __dpath_archives(self) -> str:
        return osp.join(self.__dpath_data, "archives")

    @property
    @lru_cache()
    def __dpath_audio(self) -> str:
        return osp.join(self.__dpath_data, "audio")

    @property
    @lru_cache()
    def __dpath_data(self) -> str:
        return osp.join(self._root, "MACS")

    @property
    @lru_cache()
    def __dpath_tau_meta(self) -> str:
        return osp.join(self.__dpath_data, "tau_meta")

    def __is_prepared(self) -> bool:
        if not osp.isdir(self.__dpath_audio):
            return False
        captions_fpath = osp.join(self.__dpath_data, MACS_FILES["captions"]["fname"])
        if not osp.isfile(captions_fpath):
            return False

        with open(captions_fpath, "r") as file:
            data = yaml.safe_load(file)
        data = data["files"]
        fnames = os.listdir(self.__dpath_audio)
        return len(data) == len(fnames)

    def __load_data(self) -> None:
        if not self.__is_prepared():
            raise RuntimeError(
                f"Cannot load data: macs is not prepared in data root={self._root}. Please use download=True in dataset constructor."
            )

        # Read data files
        captions_fname = MACS_FILES["captions"]["fname"]
        captions_fpath = osp.join(self.__dpath_data, captions_fname)
        if self._verbose >= 2:
            logger.debug(f"Reading captions file {captions_fname}...")

        with open(captions_fpath, "r") as file:
            caps_data = yaml.safe_load(file)

        tau_meta_fname = "meta.csv"
        tau_meta_fpath = osp.join(self.__dpath_tau_meta, tau_meta_fname)
        if self._verbose >= 2:
            logger.debug(
                f"Reading Tau Urban acoustic scene meta file {tau_meta_fname}..."
            )

        with open(tau_meta_fpath, "r") as file:
            reader = csv.DictReader(file, delimiter="\t")
            tau_tags_data = list(reader)

        competence_fname = "MACS_competence.csv"
        competence_fpath = osp.join(self.__dpath_data, competence_fname)
        if self._verbose >= 2:
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
        all_items.update(
            {key: [None for _ in range(dataset_size)] for key in tau_additional_keys}
        )

        tau_meta_fpath = osp.join(self.__dpath_tau_meta, "meta.csv")
        for tau_tags in tau_tags_data:
            fname = osp.basename(tau_tags["filename"])
            if fname in fname_to_idx:
                idx = fname_to_idx[fname]
                for key in tau_additional_keys:
                    all_items[key][idx] = tau_tags[key]

        if self._flat_captions and self.MIN_CAPTIONS_PER_AUDIO[self._subset] > 1:
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
        self._all_items = all_items
        self._annotator_id_to_competence = annotator_id_to_competence
        self._loaded = True

        if self._verbose >= 1:
            logger.info(f"{repr(self)} has been loaded. (len={len(self)})")

    def __prepare_data(self) -> None:
        if not osp.isdir(self._root):
            raise RuntimeError(f"Cannot find root directory '{self._root}'.")

        os.makedirs(self.__dpath_archives, exist_ok=True)
        os.makedirs(self.__dpath_audio, exist_ok=True)
        os.makedirs(self.__dpath_tau_meta, exist_ok=True)

        # Download MACS specific files
        for file_info in MACS_FILES.values():
            dpath = self.__dpath_data
            fname = file_info["fname"]
            fpath = osp.join(dpath, fname)

            if not osp.isfile(fpath) or self.FORCE_PREPARE_DATA:
                if self._verbose >= 1:
                    logger.info(f"Downloading captions file '{fname}'...")

                url = file_info["url"]
                download_url_to_file(
                    url,
                    fpath,
                    progress=self._verbose >= 1,
                )

            hash_value = file_info["hash_value"]
            with open(fpath, "rb") as file:
                valid = validate_file(file, hash_value, hash_type="md5")
            if not valid:
                raise RuntimeError(f"Invalid checksum for file {fname}.")

        captions_fpath = osp.join(self.__dpath_data, MACS_FILES["captions"]["fname"])
        with open(captions_fpath, "r") as file:
            captions_data = yaml.safe_load(file)
        captions_data = captions_data["files"]

        # Download TAU Urban Sound audio archives files
        for i, file_info in enumerate(MACS_ARCHIVES_FILES.values()):
            dpath = self.__dpath_archives
            zip_fname = file_info["fname"]
            zip_fpath = osp.join(dpath, zip_fname)

            if not osp.isfile(zip_fpath) or self.FORCE_PREPARE_DATA:
                if self._verbose >= 1:
                    logger.info(
                        f"Downloading audio zip file '{zip_fpath}'... ({i+1}/{len(MACS_ARCHIVES_FILES)})"
                    )

                url = file_info["url"]
                download_url_to_file(
                    url,
                    zip_fpath,
                    progress=self._verbose >= 1,
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
            zip_fpath = osp.join(self.__dpath_archives, zip_fname)

            if self._verbose >= 2:
                logger.debug(
                    f"Check to extract TAU Urban acoustic scenes archive zip_fname={zip_fname}..."
                )

            is_audio_archive = name.startswith("audio")
            target_dpath = (
                self.__dpath_audio if is_audio_archive else self.__dpath_tau_meta
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

                if self._verbose >= 1:
                    logger.info(
                        f"Extracting {len(members_to_extract)}/{len(file.namelist())} audio files from ZIP file '{zip_fname}'... ({i+1}/{len(MACS_ARCHIVES_FILES)})"
                    )

                if len(members_to_extract) > 0:
                    file.extractall(self.__dpath_archives, members_to_extract)
                    for member in members_to_extract:
                        extracted_fpath = osp.join(self.__dpath_archives, member)
                        target_fpath = osp.join(target_dpath, osp.basename(member))
                        shutil.move(extracted_fpath, target_fpath)

        if self.CLEAN_ARCHIVES:
            if self._verbose >= 1:
                logger.info(f"Removing archives files in {self.__dpath_archives}...")
            shutil.rmtree(self.__dpath_archives, ignore_errors=True)

        audio_fnames = [
            name for name in os.listdir(self.__dpath_audio) if name.endswith(".wav")
        ]
        assert len(audio_fnames) == len(macs_fnames)

        if self._verbose >= 1:
            logger.info(
                f"{len(audio_fnames)} audio files has been prepared for MACS dataset."
            )


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
