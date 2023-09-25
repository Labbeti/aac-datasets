#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import csv
import logging
import os
import os.path as osp
import shutil
import zipfile

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
)

import yaml

from torch import Tensor
from torch.hub import download_url_to_file
from typing_extensions import TypedDict

from aac_datasets.datasets.base import AACDataset, DatasetCard
from aac_datasets.utils.download import validate_file
from aac_datasets.utils.paths import _get_root


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


class MACSCard(DatasetCard):
    ANNOTATIONS_CREATORS: Tuple[str, ...] = ("crowdsourced",)
    CITATION: str = r"""
    @inproceedings{Martin2021b,
        title        = {Diversity and Bias in Audio Captioning Datasets},
        author       = {Martin, Irene and Mesaros, Annamaria},
        year         = 2021,
        month        = {November},
        booktitle    = {Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021)},
        address      = {Barcelona, Spain},
        pages        = {90--94},
        isbn         = {978-84-09-36072-7},
        url          = {https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Martin_34.pdf},
        abstract     = {Describing soundscapes in sentences allows better understanding of the acoustic scene than a single label indicating the acoustic scene class or a set of audio tags indicating the sound events active in the audio clip. In addition, the richness of natural language allows a range of possible descriptions for the same acoustic scene. In this work, we address the diversity obtained when collecting descriptions of soundscapes using crowdsourcing. We study how much the collection of audio captions can be guided by the instructions given in the annotation task, by analysing the possible bias introduced by auxiliary information provided in the annotation process. Our study shows that even when given hints on the audio content, different annotators describe the same soundscape using different vocabulary. In automatic captioning, hints provided as audio tags represent grounding textual information that facilitates guiding the captioning output towards specific concepts. We also release a new dataset of audio captions and audio tags produced by multiple annotators for a subset of the TAU Urban Acoustic Scenes 2018 dataset, suitable for studying guided captioning.},
        doi.         = {10.5281/zenodo.5770113}
    }
    """
    DESCRIPTION = "Multi-Annotator Captioned Soundscapes dataset."
    HOMEPAGE: str = "https://zenodo.org/record/5114771"
    LANGUAGE: Tuple[str, ...] = ("en",)
    LANGUAGE_DETAILS: Tuple[str, ...] = ("en-US",)
    MAX_CAPTIONS_PER_AUDIO: Dict[str, int] = {"full": 5}
    MIN_CAPTIONS_PER_AUDIO: Dict[str, int] = {"full": 2}
    NAME: str = "macs"
    N_CHANNELS: int = 2
    PRETTY_NAME: str = "MACS"
    SAMPLE_RATE: int = 48_000  # Hz
    SIZE_CATEGORIES: Tuple[str, ...] = ("1K<n<10K",)
    SUBSETS: Tuple[str, ...] = ("full",)
    TASK_CATEGORIES: Tuple[str, ...] = ("audio-to-text", "text-to-audio")


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
    FORCE_PREPARE_DATA: ClassVar[bool] = False
    VERIFY_FILES: ClassVar[bool] = True

    # MACS-specific globals
    CLEAN_ARCHIVES: ClassVar[bool] = False

    # Initialization
    def __init__(
        self,
        root: str = ...,
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
        :param transform: The transform to apply to the global dict item. This transform is applied only in getitem method when argument is an integer.
            defaults to None.
        :param flat_captions: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param verbose: Verbose level to use. Can be 0 or 1.
            defaults to 0.
        """
        if subset not in MACSCard.SUBSETS:
            raise ValueError(
                f"Invalid argument subset={subset} for MACS. (expected one of {MACSCard.SUBSETS})"
            )

        root = _get_root(root)

        if download:
            _prepare_macs_dataset(
                root,
                subset,
                verbose,
                MACS.FORCE_PREPARE_DATA,
                MACS.VERIFY_FILES,
                MACS.CLEAN_ARCHIVES,
            )
        raw_data, annotator_id_to_competence = _load_macs_dataset(root, subset, verbose)

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

        self.add_post_columns(
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


def _get_macs_dpath(root: str) -> str:
    return osp.join(root, "MACS")


def _get_archives_dpath(root: str) -> str:
    return osp.join(_get_macs_dpath(root), "archives")


def _get_audio_dpath(root: str) -> str:
    return osp.join(_get_macs_dpath(root), "audio")


def _get_tau_meta_dpath(root: str) -> str:
    return osp.join(_get_macs_dpath(root), "tau_meta")


def _is_prepared(root: str) -> bool:
    audio_dpath = _get_audio_dpath(root)
    if not osp.isdir(audio_dpath):
        return False
    captions_fpath = osp.join(_get_macs_dpath(root), MACS_FILES["captions"]["fname"])
    if not osp.isfile(captions_fpath):
        return False

    with open(captions_fpath, "r") as file:
        data = yaml.safe_load(file)
    data = data["files"]
    fnames = os.listdir(audio_dpath)
    return len(data) == len(fnames)


def _load_macs_dataset(
    root: str, subset: str, verbose: int
) -> Tuple[Dict[str, List[Any]], Dict[int, float]]:
    if not _is_prepared(root):
        raise RuntimeError(
            f"Cannot load data: macs is not prepared in data root={root}. Please use download=True in dataset constructor."
        )

    macs_dpath = _get_macs_dpath(root)
    tau_meta_dpath = _get_tau_meta_dpath(root)

    # Read data files
    captions_fname = MACS_FILES["captions"]["fname"]
    captions_fpath = osp.join(macs_dpath, captions_fname)
    if verbose >= 2:
        pylog.debug(f"Reading captions file {captions_fname}...")

    with open(captions_fpath, "r") as file:
        caps_data = yaml.safe_load(file)

    tau_meta_fname = "meta.csv"
    tau_meta_fpath = osp.join(tau_meta_dpath, tau_meta_fname)
    if verbose >= 2:
        pylog.debug(f"Reading Tau Urban acoustic scene meta file {tau_meta_fname}...")

    with open(tau_meta_fpath, "r") as file:
        reader = csv.DictReader(file, delimiter="\t")
        tau_tags_data = list(reader)

    competence_fname = "MACS_competence.csv"
    competence_fpath = osp.join(macs_dpath, competence_fname)
    if verbose >= 2:
        pylog.debug(f"Reading file {competence_fname}...")

    with open(competence_fpath, "r") as file:
        reader = csv.DictReader(file, delimiter="\t")
        competences_data = list(reader)

    # Store MACS data
    raw_data: Dict[str, List[Any]] = {
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
    dataset_size = len(raw_data["fname"])

    # Build global mappings
    fname_to_idx = {fname: i for i, fname in enumerate(raw_data["fname"])}
    annotator_id_to_competence = {
        int(annotator["annotator_id"]): float(annotator["competence"])
        for annotator in competences_data
    }

    # Store TAU Urban acoustic scenes data
    tau_additional_keys = ("scene_label", "identifier")
    raw_data.update(
        {key: [None for _ in range(dataset_size)] for key in tau_additional_keys}
    )

    tau_meta_fpath = osp.join(tau_meta_dpath, "meta.csv")
    for tau_tags in tau_tags_data:
        fname = osp.basename(tau_tags["filename"])
        if fname in fname_to_idx:
            idx = fname_to_idx[fname]
            for key in tau_additional_keys:
                raw_data[key][idx] = tau_tags[key]

    # Sanity checks
    assert all(
        all(value is not None for value in raw_data[key]) for key in tau_additional_keys
    )
    assert all(len(values) == dataset_size for values in raw_data.values())

    if verbose >= 1:
        pylog.info(
            f"Dataset {MACSCard.PRETTY_NAME} ({subset}) has been loaded. (len={len(next(iter(raw_data.values())))})"
        )

    return raw_data, annotator_id_to_competence


def _prepare_macs_dataset(
    root: str,
    subset: str,
    verbose: int,
    force: bool,
    verify_files: bool,
    clean_archives: bool,
) -> None:
    if not osp.isdir(root):
        raise RuntimeError(f"Cannot find root directory '{root}'.")

    macs_dpath = _get_macs_dpath(root)
    archives_dpath = _get_archives_dpath(root)
    audio_dpath = _get_audio_dpath(root)
    tau_meta_dpath = _get_tau_meta_dpath(root)

    for dpath in (archives_dpath, audio_dpath, tau_meta_dpath):
        os.makedirs(dpath, exist_ok=True)

    # Download MACS specific files
    for file_info in MACS_FILES.values():
        fname = file_info["fname"]
        fpath = osp.join(macs_dpath, fname)

        if not osp.isfile(fpath) or force:
            if verbose >= 1:
                pylog.info(f"Downloading captions file '{fname}'...")

            url = file_info["url"]
            download_url_to_file(
                url,
                fpath,
                progress=verbose >= 1,
            )

        if verify_files:
            hash_value = file_info["hash_value"]
            valid = validate_file(fpath, hash_value, hash_type="md5")
            if not valid:
                raise RuntimeError(f"Invalid checksum for file {fname}.")
            elif verbose >= 2:
                pylog.debug(f"File '{fname}' has a valid checksum.")

    captions_fpath = osp.join(macs_dpath, MACS_FILES["captions"]["fname"])
    with open(captions_fpath, "r") as file:
        captions_data = yaml.safe_load(file)
    captions_data = captions_data["files"]

    # Download TAU Urban Sound audio archives files
    for i, file_info in enumerate(MACS_ARCHIVES_FILES.values()):
        zip_fname = file_info["fname"]
        zip_fpath = osp.join(archives_dpath, zip_fname)

        if not osp.isfile(zip_fpath) or force:
            if verbose >= 1:
                pylog.info(
                    f"Downloading audio zip file '{zip_fpath}'... ({i+1}/{len(MACS_ARCHIVES_FILES)})"
                )

            url = file_info["url"]
            download_url_to_file(
                url,
                zip_fpath,
                progress=verbose >= 1,
            )

        if verify_files:
            hash_value = file_info["hash_value"]
            valid = validate_file(zip_fpath, hash_value, hash_type="md5")
            if not valid:
                raise RuntimeError(f"Invalid checksum for file {zip_fname}.")
            elif verbose >= 2:
                pylog.debug(f"File '{zip_fname}' has a valid checksum.")

    # Extract files from TAU Urban Sound archives
    macs_fnames = dict.fromkeys(data["filename"] for data in captions_data)
    for i, (name, file_info) in enumerate(MACS_ARCHIVES_FILES.items()):
        zip_fname = file_info["fname"]
        zip_fpath = osp.join(archives_dpath, zip_fname)

        if verbose >= 2:
            pylog.debug(
                f"Check to extract TAU Urban acoustic scenes archive zip_fname={zip_fname}..."
            )

        is_audio_archive = name.startswith("audio")
        if is_audio_archive:
            target_dpath = audio_dpath
        else:
            target_dpath = tau_meta_dpath

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

            if verbose >= 1:
                pylog.info(
                    f"Extracting {len(members_to_extract)}/{len(file.namelist())} audio files from ZIP file '{zip_fname}'... ({i+1}/{len(MACS_ARCHIVES_FILES)})"
                )

            if len(members_to_extract) > 0:
                file.extractall(archives_dpath, members_to_extract)
                for member in members_to_extract:
                    extracted_fpath = osp.join(archives_dpath, member)
                    target_fpath = osp.join(target_dpath, osp.basename(member))
                    shutil.move(extracted_fpath, target_fpath)

    if clean_archives:
        if verbose >= 1:
            pylog.info(f"Removing archives files in {archives_dpath}...")
        shutil.rmtree(archives_dpath, ignore_errors=True)

    audio_fnames = [name for name in os.listdir(audio_dpath) if name.endswith(".wav")]
    assert len(audio_fnames) == len(macs_fnames)

    if verbose >= 2:
        pylog.debug(f"Dataset {MACSCard.PRETTY_NAME} ({subset}) has been prepared.")


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

# List of TAU_URBAN_ACOUSTIC archives containing at least 1 MACS audio file.
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
