#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
import os.path as osp
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import yaml
from typing_extensions import Literal

from aac_datasets.datasets.functional.common import DatasetCard, LinkInfoHash
from aac_datasets.utils.download import download_file, hash_file
from aac_datasets.utils.globals import _get_root

pylog = logging.getLogger(__name__)


MACSSubset = Literal["full"]


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
    DEFAULT_SUBSET: MACSSubset = "full"
    DESCRIPTION: str = "Multi-Annotator Captioned Soundscapes dataset."
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
    SUBSETS: Tuple[MACSSubset, ...] = ("full",)
    TASK_CATEGORIES: Tuple[str, ...] = ("audio-to-text", "text-to-audio")


def load_macs_dataset(
    # Common args
    root: Union[str, Path, None] = None,
    subset: MACSSubset = MACSCard.DEFAULT_SUBSET,
    verbose: int = 0,
) -> Tuple[Dict[str, List[Any]], Dict[int, float]]:
    """Load MACS metadata.

    :param root: Dataset root directory.
        defaults to ".".
    :param subset: The subset of MACS to use. Can be one of :attr:`~MACSCard.SUBSETS`.
        defaults to "full".
    :param verbose: Verbose level.
        defaults to 0.
    :returns: A dictionnary of lists containing each metadata.
    """

    root = _get_root(root)
    if not _is_prepared_macs(root):
        raise RuntimeError(
            f"Cannot load data: macs is not prepared in data {root=}. Please use download=True in dataset constructor."
        )

    macs_dpath = _get_macs_root(root)
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

    competence_fname = MACS_FILES["annotators_competences"]["fname"]
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
            index = fname_to_idx[fname]
            for key in tau_additional_keys:
                raw_data[key][index] = tau_tags[key]

    # Sanity checks
    assert all(
        all(value is not None for value in raw_data[key]) for key in tau_additional_keys
    )
    assert all(len(values) == dataset_size for values in raw_data.values())

    if verbose >= 1:
        msg = f"Dataset {MACSCard.PRETTY_NAME} ({subset}) has been loaded. {len(next(iter(raw_data.values())))=})"
        pylog.info(msg)

    return raw_data, annotator_id_to_competence


def download_macs_dataset(
    # Common args
    root: Union[str, Path, None] = None,
    subset: MACSSubset = MACSCard.DEFAULT_SUBSET,
    force: bool = False,
    verbose: int = 0,
    verify_files: bool = True,
    *,
    # MACS-specific args
    clean_archives: bool = True,
) -> None:
    """Prepare MACS data.

    :param root: Dataset root directory.
        defaults to ".".
    :param subset: The subset of MACS to use. Can be one of :attr:`~MACSCard.SUBSETS`.
        defaults to "full".
    :param force: If True, force to download again all files.
        defaults to False.
    :param verbose: Verbose level.
        defaults to 0.
    :param verify_files: If True, check all file already downloaded are valid.
        defaults to False.

    :param clean_archives: If True, remove the compressed archives from disk to save space.
        defaults to True.
    """

    root = _get_root(root)
    if not osp.isdir(root):
        raise RuntimeError(f"Cannot find root directory '{root}'.")

    macs_dpath = _get_macs_root(root)
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
            download_file(url, fpath, verbose=verbose)

        if verify_files:
            hash_value = file_info["hash_value"]
            file_hash_value = hash_file(fpath, hash_type="md5")
            if file_hash_value != hash_value:
                raise RuntimeError(
                    f"Invalid checksum for file '{fname}'. (expected md5 checksum '{hash_value}' but found '{file_hash_value}')\n"
                    f"Please try to remove manually the file '{fpath}' and rerun {MACSCard.PRETTY_NAME} download."
                )
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
            download_file(url, zip_fpath, verbose=verbose)

        if verify_files:
            hash_value = file_info["hash_value"]
            file_hash_value = hash_file(zip_fpath, hash_type="md5")
            if file_hash_value != hash_value:
                raise RuntimeError(
                    f"Invalid checksum for file '{zip_fname}'. (expected md5 checksum '{hash_value}' but found '{file_hash_value}')\n"
                    f"Please try to remove manually the file '{zip_fpath}' and rerun {MACSCard.PRETTY_NAME} download."
                )
            elif verbose >= 2:
                pylog.debug(f"File '{zip_fname}' has a valid checksum.")

    # Extract files from TAU Urban Sound archives
    macs_fnames = dict.fromkeys(data["filename"] for data in captions_data)
    for i, (name, file_info) in enumerate(MACS_ARCHIVES_FILES.items()):
        zip_fname = file_info["fname"]
        zip_fpath = osp.join(archives_dpath, zip_fname)

        if verbose >= 2:
            pylog.debug(
                f"Check to extract TAU Urban acoustic scenes archive {zip_fname=}..."
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


def download_macs_datasets(
    # Common args
    root: Union[str, Path, None] = None,
    subsets: Union[MACSSubset, Iterable[MACSSubset]] = MACSCard.DEFAULT_SUBSET,
    force: bool = False,
    verbose: int = 0,
    *,
    # MACS-specific args
    clean_archives: bool = True,
    verify_files: bool = True,
) -> None:
    """Function helper to download a list of subsets. See :func:`~aac_datasets.datasets.functional.macs.download_macs_dataset` for details."""
    if isinstance(subsets, str):
        subsets = [subsets]
    else:
        subsets = list(subsets)

    kwargs: Dict[str, Any] = dict(
        root=root,
        force=force,
        verbose=verbose,
        clean_archives=clean_archives,
        verify_files=verify_files,
    )
    for subset in subsets:
        download_macs_dataset(
            subset=subset,
            **kwargs,
        )


def _get_macs_root(root: str) -> str:
    return osp.join(root, "MACS")


def _get_archives_dpath(root: str) -> str:
    return osp.join(_get_macs_root(root), "archives")


def _get_audio_dpath(root: str) -> str:
    return osp.join(_get_macs_root(root), "audio")


def _get_tau_meta_dpath(root: str) -> str:
    return osp.join(_get_macs_root(root), "tau_meta")


def _is_prepared_macs(root: str) -> bool:
    audio_dpath = _get_audio_dpath(root)
    if not osp.isdir(audio_dpath):
        return False
    captions_fpath = osp.join(_get_macs_root(root), MACS_FILES["captions"]["fname"])
    if not osp.isfile(captions_fpath):
        return False

    with open(captions_fpath, "r") as file:
        data = yaml.safe_load(file)
    data = data["files"]
    fnames = os.listdir(audio_dpath)
    return len(data) == len(fnames)


# Internal typing to make easier to add new links without error
MACSLinkType = Literal["licence", "captions", "annotators_competences"]

# MACS-specific files links.
MACS_FILES: Dict[MACSLinkType, LinkInfoHash] = {
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
