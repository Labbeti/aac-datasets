#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import csv
import logging
import os
import os.path as osp
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from zipfile import ZipFile

from py7zr import SevenZipFile
from typing_extensions import Literal

from aac_datasets.datasets.functional.common import DatasetCard, LinkInfoHash
from aac_datasets.utils.collections import union_dicts
from aac_datasets.utils.download import download_file, hash_file
from aac_datasets.utils.globals import _get_root

pylog = logging.getLogger(__name__)


ClothoSubset = Literal[
    "dev",
    "val",
    "eval",
    "dcase_aac_test",
    "dcase_aac_analysis",
    "dcase_t2a_audio",
    "dcase_t2a_captions",
]
ClothoVersion = Literal["v1", "v2", "v2.1"]


class ClothoCard(DatasetCard):
    ANNOTATIONS_CREATORS: Tuple[str, ...] = ("crowdsourced",)
    CAPTIONS_PER_AUDIO: Dict[ClothoSubset, int] = {
        "dev": 5,
        "val": 5,
        "eval": 5,
        "dcase_aac_test": 0,
        "dcase_aac_analysis": 0,
        "dcase_t2a_audio": 0,
        "dcase_t2a_captions": 1,
    }
    CITATION: str = r"""
    @inproceedings{Drossos_2020_icassp,
        title        = {Clotho: an Audio Captioning Dataset},
        author       = {Drossos, Konstantinos and Lipping, Samuel and Virtanen, Tuomas},
        year         = 2020,
        booktitle    = {Proc. IEEE Int. Conf. Acoustic., Speech and Signal Process. (ICASSP)},
        pages        = {736--740},
        abstract     = {Audio captioning is the novel task of general audio content description using free text. It is an intermodal translation task (not speech-to-text), where a system accepts as an input an audio signal and outputs the textual description (i.e. the caption) of that signal. In this paper we present Clotho, a dataset for audio captioning consisting of 4981 audio samples of 15 to 30 seconds duration and 24 905 captions of eight to 20 words length, and a baseline method to provide initial results. Clotho is built with focus on audio content and caption diversity, and the splits of the data are not hampering the training or evaluation of methods. All sounds are from the Freesound platform, and captions are crowdsourced using Amazon Mechanical Turk and annotators from English speaking countries. Unique words, named entities, and speech transcription are removed with post-processing. Clotho is freely available online (https://zenodo.org/record/3490684).}
    }
    """
    DEFAULT_SUBSET: ClothoSubset = "dev"
    DEFAULT_VERSION: ClothoVersion = "v2.1"
    HOMEPAGE: str = "https://zenodo.org/record/3490684"
    LANGUAGE: Tuple[str, ...] = ("en",)
    LANGUAGE_DETAILS: Tuple[str, ...] = ("en-US",)
    NAME: str = "clotho"
    N_CHANNELS: int = 1
    PRETTY_NAME: str = "Clotho"
    SAMPLE_RATE: int = 44_100  # Hz
    SIZE_CATEGORIES: Tuple[str, ...] = ("1K<n<10K",)
    SUBSETS: Tuple[ClothoSubset, ...] = tuple(CAPTIONS_PER_AUDIO.keys())
    TASK_CATEGORIES: Tuple[str, ...] = ("audio-to-text", "text-to-audio")
    VERSIONS: Tuple[ClothoVersion, ...] = ("v1", "v2", "v2.1")


def load_clotho_dataset(
    # Common args
    root: Union[str, Path, None] = None,
    subset: ClothoSubset = ClothoCard.DEFAULT_SUBSET,
    verbose: int = 0,
    *,
    # Clotho-specific args
    version: ClothoVersion = ClothoCard.DEFAULT_VERSION,
) -> Dict[str, List[Any]]:
    """Load Clotho metadata.

    :param root: Dataset root directory.
        defaults to ".".
    :param subset: The subset of Clotho to use. Can be one of :attr:`~ClothoCard.SUBSETS`.
        defaults to "dev".
    :param verbose: Verbose level.
        defaults to 0.

    :param version: The version of the dataset. Can be one of :attr:`~ClothoCard.VERSIONS`.
        defaults to 'v2.1'.
    :returns: A dictionnary of lists containing each metadata.
    """
    root = _get_root(root)
    if not _is_prepared_clotho(root, version, subset):
        raise RuntimeError(
            f"Cannot load data: clotho_{subset} is not prepared in data {root=}. Please use download=True in dataset constructor."
        )

    # Read fpath of .wav audio files
    links = _CLOTHO_LINKS[version][subset]
    csv_dpath = _get_csv_dpath(root, version)

    # Read Clotho files
    if "captions" in links.keys():
        captions_fname = links["captions"]["fname"]
        captions_fpath = osp.join(csv_dpath, captions_fname)

        # Keys: file_name, caption_1, caption_2, caption_3, caption_4, caption_5
        with open(captions_fpath, "r") as file:
            reader = csv.DictReader(file)
            captions_data = list(reader)

        if subset == "dcase_t2a_captions":
            captions_data = [
                union_dicts(data, {"file_name": f"no_fname_{i}"})
                for i, data in enumerate(captions_data)
            ]

    else:
        captions_data = []

    if "metadata" in links.keys():
        metadata_fname = links["metadata"]["fname"]
        metadata_fpath = osp.join(csv_dpath, metadata_fname)

        # Keys: file_name, keywords, sound_id, sound_link, start_end_samples, manufacturer, license
        if version in ("v2", "v2.1"):
            encoding = "ISO-8859-1"
        else:
            encoding = None

        with open(metadata_fpath, "r", encoding=encoding) as file:
            delimiter = ";" if subset == "dcase_aac_test" else ","
            reader = csv.DictReader(file, delimiter=delimiter)
            metadata = list(reader)
    else:
        metadata = []

    if "captions" in links.keys():
        # note: "dev", "val", "eval"
        fnames_lst = [line["file_name"] for line in captions_data]
    elif "metadata" in links.keys():
        # note: for "dcase_aac_test" subset which do not have captions CSV file
        fnames_lst = [line["file_name"] for line in metadata]
    else:
        # note 1: for "dcase_aac_analysis" subset which do not have any CSV file
        # note 2: force sorted list to have the same order on all OS
        audio_subset_dpath = _get_audio_subset_dpath(root, version, subset)
        if audio_subset_dpath is None:
            raise RuntimeError(
                f"INTERNAL ERROR: Invalid audio subset dirpath. (found audio_subset_dpath={audio_subset_dpath}, with {subset=})"
            )
        fnames_lst = list(sorted(os.listdir(audio_subset_dpath)))

    idx_to_fname = {i: fname for i, fname in enumerate(fnames_lst)}
    fname_to_idx = {fname: i for i, fname in idx_to_fname.items()}
    dataset_size = len(fnames_lst)

    # Process each item field
    if len(metadata) > 0:
        subset_metadata_keys = [key for key in _METADATA_KEYS if key in metadata[0]]
    else:
        subset_metadata_keys = []

    all_captions_lst = [[] for _ in range(dataset_size)]

    if subset != "dcase_t2a_captions":
        captions_keys = _CAPTIONS_KEYS
    else:
        captions_keys = ("caption",)

    for line in captions_data:
        fname = line["file_name"]
        index = fname_to_idx[fname]
        all_captions_lst[index] = [line[caption_key] for caption_key in captions_keys]

    all_metadata_dic: Dict[str, List[Any]] = {
        key: [None for _ in range(dataset_size)] for key in subset_metadata_keys
    }
    for line in metadata:
        fname = line["file_name"]
        if fname not in fname_to_idx:
            raise KeyError(
                f"Cannot find metadata fname={fname} in captions file. {subset=})"
            )
        index = fname_to_idx[fname]
        for key in subset_metadata_keys:
            # The test subset does not have keywords in metadata, but has sound_id, sound_link, etc.
            if key in line:
                all_metadata_dic[key][index] = line[key]

    raw_data = {
        "fname": fnames_lst,
        "captions": all_captions_lst,
    }
    raw_data.update(all_metadata_dic)

    if "keywords" in raw_data:
        # Split keywords into List[str]
        raw_data["keywords"] = [
            keywords.split(";") if keywords is not None else []
            for keywords in raw_data["keywords"]
        ]

    if subset == "dcase_t2a_audio":
        # Temporary patch to avoid file loading errors
        # indexes: 53, 521, 677
        replaces = {
            "raindrops on metal: police background.wav": "raindrops on metal_police background.wav",
            "Intersection Wet : Metro Pass.wav": "Intersection Wet_Metro Pass.wav",
            "Kitchen Roomtone w: Dripping Faucet_1-2.wav": "Kitchen Roomtone w_Dripping Faucet_1-2.wav",
        }
        raw_data["fname"] = [replaces.get(fname, fname) for fname in raw_data["fname"]]

    if verbose >= 1:
        pylog.info(
            f"Dataset {ClothoCard.PRETTY_NAME} ({subset}) has been loaded. {len(next(iter(raw_data.values())))=})"
        )
    return raw_data


def download_clotho_dataset(
    # Common args
    root: Union[str, Path, None] = None,
    subset: ClothoSubset = ClothoCard.DEFAULT_SUBSET,
    force: bool = False,
    verbose: int = 0,
    verify_files: bool = True,
    *,
    # Clotho-specific args
    clean_archives: bool = True,
    version: ClothoVersion = ClothoCard.DEFAULT_VERSION,
) -> None:
    """Prepare Clotho data.

    :param root: Dataset root directory.
        defaults to ".".
    :param subset: The subset of Clotho to use. Can be one of :attr:`~ClothoCard.SUBSETS`.
        defaults to "dev".
    :param force: If True, force to download again all files.
        defaults to False.
    :param verbose: Verbose level.
        defaults to 0.
    :param verify_files: If True, check all file already downloaded are valid.
        defaults to False.

    :param clean_archives: If True, remove the compressed archives from disk to save space.
        defaults to True.
    :param version: The version of the dataset. Can be one of :attr:`~ClothoCard.VERSIONS`.
        defaults to 'v2.1'.
    """
    if subset == "val" and version == "v1":
        pylog.error(
            f"Clotho version '{version}' does not have '{subset}' subset. It will be ignored."
        )
        return None

    root = _get_root(root)
    if not osp.isdir(root):
        raise RuntimeError(f"Cannot find root directory '{root}'.")

    archives_dpath = _get_archives_dpath(root, version)
    audio_dpath = _get_audio_dpath(root, version)
    csv_dpath = _get_csv_dpath(root, version)

    for dpath in (archives_dpath, audio_dpath, csv_dpath):
        os.makedirs(dpath, exist_ok=True)

    if verbose >= 1:
        pylog.info(f"Start to download files for clotho_{subset}...")

    links = copy.deepcopy(_CLOTHO_LINKS[version][subset])
    EXTENSIONS = ("7z", "csv", "zip")

    # Download csv and 7z files
    for file_info in links.values():
        fname, url, hash_value = (
            file_info["fname"],
            file_info["url"],
            file_info["hash_value"],
        )
        extension = fname.split(".")[-1]

        if extension in ("7z", "zip"):
            dpath = archives_dpath
        elif extension == "csv":
            dpath = csv_dpath
        else:
            raise RuntimeError(
                f"Found invalid {extension=}. Must be one of {EXTENSIONS}."
            )

        fpath = osp.join(dpath, fname)
        if not osp.isfile(fpath) or force:
            if verbose >= 1:
                pylog.info(f"Download and check file '{fname}' from {url=}...")

            download_file(url, fpath, verbose=verbose)

        elif verbose >= 1:
            pylog.info(f"File '{fname}' is already downloaded.")

        if verify_files:
            hash_value = file_info["hash_value"]
            file_hash_value = hash_file(fpath, hash_type="md5")
            if file_hash_value != hash_value:
                raise RuntimeError(
                    f"Invalid checksum for file '{fname}'. (expected md5 checksum '{hash_value}' but found '{file_hash_value}')\n"
                    f"Please try to remove manually the file '{fpath}' and rerun {ClothoCard.PRETTY_NAME} download."
                )
            elif verbose >= 2:
                pylog.debug(f"File '{fname}' has a valid checksum.")

    # Extract audio files from archives
    audio_subset_dpath = _get_audio_subset_dpath(root, version, subset)
    if audio_subset_dpath is not None:
        for file_info in links.values():
            fname = file_info["fname"]
            extension = fname.split(".")[-1]

            if extension == "csv":
                continue

            if extension not in ("7z", "zip"):
                pylog.error(
                    f"Found unexpected {extension=} for downloaded file '{fname}'. Expected one of {EXTENSIONS}."
                )
                continue

            fpath = osp.join(archives_dpath, fname)

            if verbose >= 1:
                pylog.info(f"Extract archive file {fname=}...")

            if extension == "7z":
                archive_file = SevenZipFile(fpath)
                compressed_fnames = [
                    osp.basename(fname) for fname in archive_file.getnames()
                ]
            elif extension == "zip":
                archive_file = ZipFile(fpath)
                compressed_fnames = [
                    osp.basename(file.filename) for file in archive_file.filelist
                ]
            else:
                raise RuntimeError(f"Invalid extension '{extension}'.")

            # Ignore dir name from archive file
            compressed_fnames = [
                fname for fname in compressed_fnames if fname.endswith(".wav")
            ]
            extracted_fnames = (
                os.listdir(audio_subset_dpath) if osp.isdir(audio_subset_dpath) else []
            )

            if set(extracted_fnames) != set(compressed_fnames):
                # For dcase_t2a_audio subset, the name of the audio dname is also "test", so we need to move the audio files to another folder named "test_retrieval_audio".
                if subset == "dcase_t2a_audio":
                    target_dpath = audio_subset_dpath
                    os.makedirs(target_dpath, exist_ok=True)
                else:
                    target_dpath = audio_dpath

                archive_file.extractall(target_dpath)

                if subset == "dcase_t2a_audio":
                    extracted_dpath = osp.join(target_dpath, "test")
                    for fname in os.listdir(extracted_dpath):
                        os.rename(
                            osp.join(extracted_dpath, fname),
                            osp.join(target_dpath, fname),
                        )
                    os.rmdir(extracted_dpath)

                # Check if files is good now
                extracted_fnames = os.listdir(audio_subset_dpath)
                if set(extracted_fnames) != set(compressed_fnames):
                    found_but_not_expected = len(
                        set(extracted_fnames).difference(set(compressed_fnames))
                    )
                    expected_but_not_found = len(
                        set(compressed_fnames).difference(set(extracted_fnames))
                    )

                    raise RuntimeError(
                        f"Invalid number of audios extracted, found {len(extracted_fnames)} files but expected the same {len(compressed_fnames)} files. "
                        f"(with found_but_not_expected={found_but_not_expected} and {expected_but_not_found=})"
                    )

            archive_file.close()

    if clean_archives:
        for file_info in links.values():
            fname = file_info["fname"]
            extension = fname.split(".")[-1]
            if extension not in ("7z", "zip"):
                continue

            fpath = osp.join(archives_dpath, fname)
            if verbose >= 1:
                pylog.info(f"Removing archive file {osp.basename(fpath)}...")
            os.remove(fpath)

    if verbose >= 2:
        pylog.debug(f"Dataset {ClothoCard.PRETTY_NAME} ({subset}) has been prepared.")


def download_clotho_datasets(
    # Common args
    root: Union[str, Path, None] = None,
    subsets: Union[ClothoSubset, Iterable[ClothoSubset]] = ClothoCard.DEFAULT_SUBSET,
    force: bool = False,
    verbose: int = 0,
    *,
    # Clotho-specific args
    clean_archives: bool = True,
    verify_files: bool = True,
    version: ClothoVersion = ClothoCard.DEFAULT_VERSION,
) -> None:
    """Function helper to download a list of subsets. See :func:`~aac_datasets.datasets.functional.clotho.download_clotho_dataset` for details."""
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
        version=version,
    )
    for subset in subsets:
        download_clotho_dataset(
            subset=subset,
            **kwargs,
        )


def _get_clotho_root(root: str, version: ClothoVersion) -> str:
    return osp.join(root, f"CLOTHO_{version}")


def _get_archives_dpath(root: str, version: ClothoVersion) -> str:
    return osp.join(_get_clotho_root(root, version), "archives")


def _get_audio_dpath(root: str, version: ClothoVersion) -> str:
    return osp.join(_get_clotho_root(root, version), "clotho_audio_files")


def _get_csv_dpath(root: str, version: ClothoVersion) -> str:
    return osp.join(_get_clotho_root(root, version), "clotho_csv_files")


def _get_audio_subset_dpath(
    root: str, version: ClothoVersion, subset: ClothoSubset
) -> Optional[str]:
    dname = _CLOTHO_AUDIO_DNAMES[subset]
    if dname is None:
        return None

    return osp.join(
        _get_clotho_root(root, version),
        "clotho_audio_files",
        dname,
    )


def _is_prepared_clotho(
    root: str,
    version: ClothoVersion,
    subset: ClothoSubset,
) -> bool:
    audio_dpath = _get_audio_dpath(root, version)
    csv_dpath = _get_csv_dpath(root, version)
    if not all(map(osp.isdir, (audio_dpath, csv_dpath))):
        return False

    links = _CLOTHO_LINKS[version][subset]

    if "captions" in links:
        captions_fname = links["captions"]["fname"]
        captions_fpath = osp.join(csv_dpath, captions_fname)

        if not osp.isfile(captions_fpath):
            return False

    if "metadata" in links:
        metadata_fname = links["metadata"]["fname"]
        metadata_fpath = osp.join(csv_dpath, metadata_fname)
        if not osp.isfile(metadata_fpath):
            return False

    if "audio_archive" in links:
        audio_subset_dpath = _get_audio_subset_dpath(root, version, subset)
        if audio_subset_dpath is None:
            raise RuntimeError(
                f"INTERNAL ERROR: Invalid audio subset dirpath. (found audio_subset_dpath={audio_subset_dpath}, with {subset=})"
            )
        if not osp.isdir(audio_subset_dpath):
            return False

        audio_fnames = os.listdir(audio_subset_dpath)
        if "captions" in links:
            captions_fname = links["captions"]["fname"]
            captions_fpath = osp.join(csv_dpath, captions_fname)
            with open(captions_fpath, "r") as file:
                reader = csv.DictReader(file)
                lines = list(reader)
            return len(audio_fnames) == len(lines)
        else:
            return len(audio_fnames) > 0

    else:
        return True


# Audio directory names per subset
_CLOTHO_AUDIO_DNAMES: Dict[ClothoSubset, Optional[str]] = {
    "dev": "development",
    "val": "validation",
    "eval": "evaluation",
    "dcase_aac_test": "test",
    "dcase_aac_analysis": "clotho_analysis",
    "dcase_t2a_audio": "test_retrieval_audio",
    "dcase_t2a_captions": None,
}


# Internal typing to make easier to add new links without error
_ClothoLinkType = Literal["audio_archive", "captions", "metadata"]


# Archives and file links used to download Clotho
_CLOTHO_LINKS: Dict[
    ClothoVersion, Dict[ClothoSubset, Dict[_ClothoLinkType, LinkInfoHash]]
] = {
    "v1": {
        "dev": {
            "audio_archive": {
                "fname": "clotho_audio_development.7z",
                "url": "https://zenodo.org/record/3490684/files/clotho_audio_development.7z?download=1",
                "hash_value": "e3ce88561b317cc3825e8c861cae1ec6",
            },
            "captions": {
                "fname": "clotho_captions_development.csv",
                "url": "https://zenodo.org/record/3490684/files/clotho_captions_development.csv?download=1",
                "hash_value": "dd568352389f413d832add5cf604529f",
            },
            "metadata": {
                "fname": "clotho_metadata_development.csv",
                "url": "https://zenodo.org/record/3490684/files/clotho_metadata_development.csv?download=1",
                "hash_value": "582c18ee47cebdbe33dce1feeab53a56",
            },
        },
        "eval": {
            "audio_archive": {
                "fname": "clotho_audio_evaluation.7z",
                "url": "https://zenodo.org/record/3490684/files/clotho_audio_evaluation.7z?download=1",
                "hash_value": "4569624ccadf96223f19cb59fe4f849f",
            },
            "captions": {
                "fname": "clotho_captions_evaluation.csv",
                "url": "https://zenodo.org/record/3490684/files/clotho_captions_evaluation.csv?download=1",
                "hash_value": "1b16b9e57cf7bdb7f13a13802aeb57e2",
            },
            "metadata": {
                "fname": "clotho_metadata_evaluation.csv",
                "url": "https://zenodo.org/record/3490684/files/clotho_metadata_evaluation.csv?download=1",
                "hash_value": "13946f054d4e1bf48079813aac61bf77",
            },
        },
        "dcase_aac_test": {
            "audio_archive": {
                "fname": "clotho_audio_test.7z",
                "url": "https://zenodo.org/record/3865658/files/clotho_audio_test.7z?download=1",
                "hash_value": "9b3fe72560a621641ff4351ba1154349",
            },
            "metadata": {
                "fname": "clotho_metadata_test.csv",
                "url": "https://zenodo.org/record/3865658/files/clotho_metadata_test.csv?download=1",
                "hash_value": "52f8ad01c229a310a0ff8043df480e21",
            },
        },
    },
    "v2": {
        "dev": {
            "audio_archive": {
                "fname": "clotho_audio_development.7z",
                "url": "https://zenodo.org/record/4743815/files/clotho_audio_development.7z?download=1",
                "hash_value": "eda144a5e05a60e6d2e37a65fc4720a9",
            },
            "captions": {
                "fname": "clotho_captions_development.csv",
                "url": "https://zenodo.org/record/4743815/files/clotho_captions_development.csv?download=1",
                "hash_value": "800633304e73d3daed364a2ba6069827",
            },
            "metadata": {
                "fname": "clotho_metadata_development.csv",
                "url": "https://zenodo.org/record/4743815/files/clotho_metadata_development.csv?download=1",
                "hash_value": "5fdc51b4c4f3468ff7d251ea563588c9",
            },
        },
        "val": {
            "audio_archive": {
                "fname": "clotho_audio_validation.7z",
                "url": "https://zenodo.org/record/4743815/files/clotho_audio_validation.7z?download=1",
                "hash_value": "0475bfa5793e80f748d32525018ebada",
            },
            "captions": {
                "fname": "clotho_captions_validation.csv",
                "url": "https://zenodo.org/record/4743815/files/clotho_captions_validation.csv?download=1",
                "hash_value": "3109c353138a089c7ba724f27d71595d",
            },
            "metadata": {
                "fname": "clotho_metadata_validation.csv",
                "url": "https://zenodo.org/record/4743815/files/clotho_metadata_validation.csv?download=1",
                "hash_value": "f69cfacebcd47c4d8d30d968f9865475",
            },
        },
        "eval": {
            "audio_archive": {
                "fname": "clotho_audio_evaluation.7z",
                "url": "https://zenodo.org/record/4743815/files/clotho_audio_evaluation.7z?download=1",
                "hash_value": "4569624ccadf96223f19cb59fe4f849f",
            },
            "captions": {
                "fname": "clotho_captions_evaluation.csv",
                "url": "https://zenodo.org/record/4743815/files/clotho_captions_evaluation.csv?download=1",
                "hash_value": "1b16b9e57cf7bdb7f13a13802aeb57e2",
            },
            "metadata": {
                "fname": "clotho_metadata_evaluation.csv",
                "url": "https://zenodo.org/record/4743815/files/clotho_metadata_evaluation.csv?download=1",
                "hash_value": "13946f054d4e1bf48079813aac61bf77",
            },
        },
        "dcase_aac_test": {
            "audio_archive": {
                "fname": "clotho_audio_test.7z",
                "url": "https://zenodo.org/record/3865658/files/clotho_audio_test.7z?download=1",
                "hash_value": "9b3fe72560a621641ff4351ba1154349",
            },
            "metadata": {
                "fname": "clotho_metadata_test.csv",
                "url": "https://zenodo.org/record/3865658/files/clotho_metadata_test.csv?download=1",
                "hash_value": "52f8ad01c229a310a0ff8043df480e21",
            },
        },
    },
    "v2.1": {
        "dev": {
            "audio_archive": {
                "fname": "clotho_audio_development.7z",
                "url": "https://zenodo.org/record/4783391/files/clotho_audio_development.7z?download=1",
                "hash_value": "c8b05bc7acdb13895bb3c6a29608667e",
            },
            "captions": {
                "fname": "clotho_captions_development.csv",
                "url": "https://zenodo.org/record/4783391/files/clotho_captions_development.csv?download=1",
                "hash_value": "d4090b39ce9f2491908eebf4d5b09bae",
            },
            "metadata": {
                "fname": "clotho_metadata_development.csv",
                "url": "https://zenodo.org/record/4783391/files/clotho_metadata_development.csv?download=1",
                "hash_value": "170d20935ecfdf161ce1bb154118cda5",
            },
        },
        "val": {
            "audio_archive": {
                "fname": "clotho_audio_validation.7z",
                "url": "https://zenodo.org/record/4783391/files/clotho_audio_validation.7z?download=1",
                "hash_value": "7dba730be08bada48bd15dc4e668df59",
            },
            "captions": {
                "fname": "clotho_captions_validation.csv",
                "url": "https://zenodo.org/record/4783391/files/clotho_captions_validation.csv?download=1",
                "hash_value": "5879e023032b22a2c930aaa0528bead4",
            },
            "metadata": {
                "fname": "clotho_metadata_validation.csv",
                "url": "https://zenodo.org/record/4783391/files/clotho_metadata_validation.csv?download=1",
                "hash_value": "2e010427c56b1ce6008b0f03f41048ce",
            },
        },
        "eval": {
            "audio_archive": {
                "fname": "clotho_audio_evaluation.7z",
                "url": "https://zenodo.org/record/4783391/files/clotho_audio_evaluation.7z?download=1",
                "hash_value": "4569624ccadf96223f19cb59fe4f849f",
            },
            "captions": {
                "fname": "clotho_captions_evaluation.csv",
                "url": "https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv?download=1",
                "hash_value": "1b16b9e57cf7bdb7f13a13802aeb57e2",
            },
            "metadata": {
                "fname": "clotho_metadata_evaluation.csv",
                "url": "https://zenodo.org/record/4783391/files/clotho_metadata_evaluation.csv?download=1",
                "hash_value": "13946f054d4e1bf48079813aac61bf77",
            },
        },
        "dcase_aac_test": {
            "audio_archive": {
                "fname": "clotho_audio_test.7z",
                "url": "https://zenodo.org/record/3865658/files/clotho_audio_test.7z?download=1",
                "hash_value": "9b3fe72560a621641ff4351ba1154349",
            },
            "metadata": {
                "fname": "clotho_metadata_test.csv",
                "url": "https://zenodo.org/record/3865658/files/clotho_metadata_test.csv?download=1",
                "hash_value": "52f8ad01c229a310a0ff8043df480e21",
            },
        },
        "dcase_aac_analysis": {
            "audio_archive": {
                "fname": "clotho_analysis_2022.zip",
                "url": "https://zenodo.org/record/6610709/files/clotho_analysis_2022.zip?download=1",
                "hash_value": "7e8fa4762cc3a7c5546606680b958d08",
            },
        },
        "dcase_t2a_audio": {
            "audio_archive": {
                "fname": "retrieval_audio.7z",
                "url": "https://zenodo.org/record/6590983/files/retrieval_audio.7z?download=1",
                "hash_value": "24102395fd757c462421a483fba5c407",
            },
            "metadata": {
                "fname": "retrieval_audio_metadata.csv",
                "url": "https://zenodo.org/record/6590983/files/retrieval_audio_metadata.csv?download=1",
                "hash_value": "1301db07acbf1e4fabc467eb54e0d353",
            },
        },
        "dcase_t2a_captions": {
            "captions": {
                "fname": "retrieval_captions.csv",
                "url": "https://zenodo.org/record/6590983/files/retrieval_captions.csv?download=1",
                "hash_value": "f9e810118be00c64ea8cd7557816d4fe",
            },
        },
    },
}

# Captions column names in CSV files
_CAPTIONS_KEYS = (
    "caption_1",
    "caption_2",
    "caption_3",
    "caption_4",
    "caption_5",
)

# Metadata column names in CSV files
_METADATA_KEYS = (
    "keywords",
    "sound_id",
    "sound_link",
    "start_end_samples",
    "manufacturer",
    "license",
)
