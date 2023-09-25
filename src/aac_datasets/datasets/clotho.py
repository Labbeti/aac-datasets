#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import csv
import logging
import os
import os.path as osp

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
)
from zipfile import ZipFile

from py7zr import SevenZipFile
from torch import Tensor
from torch.hub import download_url_to_file
from typing_extensions import TypedDict, NotRequired

from aac_datasets.datasets.base import AACDataset, DatasetCard
from aac_datasets.utils.download import validate_file
from aac_datasets.utils.paths import _get_root


pylog = logging.getLogger(__name__)


class ClothoItem(TypedDict, total=True):
    r"""Class representing a single Clotho item."""

    # Common attributes
    audio: NotRequired[Tensor]
    captions: NotRequired[List[str]]
    dataset: str
    fname: NotRequired[str]
    index: int
    subset: str
    sr: NotRequired[int]
    # Clotho-specific attributes
    keywords: NotRequired[List[str]]
    sound_id: NotRequired[str]  # warning: some files contains "Not found"
    sound_link: NotRequired[str]  # warning: some files contains "NA"
    start_end_samples: NotRequired[str]  # warning: some files contains ""
    manufacturer: NotRequired[str]
    license: NotRequired[str]


class ClothoCard(DatasetCard):
    ANNOTATIONS_CREATORS: Tuple[str, ...] = ("crowdsourced",)
    CAPTIONS_PER_AUDIO = {
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
        author = "Drossos, Konstantinos and Lipping, Samuel and Virtanen, Tuomas",
        title = "Clotho: an Audio Captioning Dataset",
        booktitle = "Proc. IEEE Int. Conf. Acoustic., Speech and Signal Process. (ICASSP)",
        year = "2020",
        pages = "736-740",
        abstract = "Audio captioning is the novel task of general audio content description using free text. It is an intermodal translation task (not speech-to-text), where a system accepts as an input an audio signal and outputs the textual description (i.e. the caption) of that signal. In this paper we present Clotho, a dataset for audio captioning consisting of 4981 audio samples of 15 to 30 seconds duration and 24 905 captions of eight to 20 words length, and a baseline method to provide initial results. Clotho is built with focus on audio content and caption diversity, and the splits of the data are not hampering the training or evaluation of methods. All sounds are from the Freesound platform, and captions are crowdsourced using Amazon Mechanical Turk and annotators from English speaking countries. Unique words, named entities, and speech transcription are removed with post-processing. Clotho is freely available online (https://zenodo.org/record/3490684)."
    }
    """
    HOMEPAGE: str = "https://zenodo.org/record/3490684"
    LANGUAGE: Tuple[str, ...] = ("en",)
    LANGUAGE_DETAILS: Tuple[str, ...] = ("en-US",)
    DEFAULT_VERSION: str = "v2.1"
    NAME: str = "clotho"
    N_CHANNELS: int = 1
    PRETTY_NAME: str = "Clotho"
    SAMPLE_RATE: int = 44_100  # Hz
    SIZE_CATEGORIES: Tuple[str, ...] = ("1K<n<10K",)
    SUBSETS: Tuple[str, ...] = (
        "dev",
        "val",
        "eval",
        "dcase_aac_test",
        "dcase_aac_analysis",
        "dcase_t2a_audio",
        "dcase_t2a_captions",
    )
    TASK_CATEGORIES: Tuple[str, ...] = ("audio-to-text", "text-to-audio")
    VERSIONS: Tuple[str, ...] = ("v1", "v2", "v2.1")


class Clotho(AACDataset[ClothoItem]):
    r"""Unofficial Clotho PyTorch dataset.

    Subsets available are 'train', 'val', 'eval', 'test' and 'analysis'.

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
    FORCE_PREPARE_DATA: ClassVar[bool] = False
    VERIFY_FILES: ClassVar[bool] = True

    # Clotho-specific globals
    CLEAN_ARCHIVES: ClassVar[bool] = True
    INVALID_SOUND_ID: ClassVar[str] = "Not found"
    INVALID_SOUND_LINK: ClassVar[str] = "NA"
    INVALID_START_END_SAMPLES: ClassVar[str] = ""

    # Initialization
    def __init__(
        self,
        root: str = ...,
        subset: str = "dev",
        download: bool = False,
        transform: Optional[Callable] = None,
        flat_captions: bool = False,
        verbose: int = 0,
        version: str = ClothoCard.DEFAULT_VERSION,
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
        :param flat_captions: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param verbose: Verbose level to use. Can be 0 or 1.
            defaults to 0.
        :param version: The version of the dataset. Can be one of :attr:`~ClothoCard.versions`.
            defaults to 'v2.1'.
        """
        if version not in ClothoCard.VERSIONS:
            raise ValueError(
                f"Invalid Clotho argument version={version}. Must be one of {ClothoCard.VERSIONS}."
            )

        if version == "v2":
            pylog.warning(
                f"The version '{version}' of the Clotho dataset contains minor some errors in file names and few corrupted files."
                f"Please consider using the fixed version 'v2.1'."
            )

        subsets = tuple(_CLOTHO_LINKS[version].keys())
        if subset not in subsets:
            raise ValueError(
                f"Invalid Clotho argument subset={subset} for version={version}. Must be one of {subsets}."
            )

        root = _get_root(root)

        if download:
            _prepare_clotho_dataset(
                root,
                version,
                subset,
                verbose,
                Clotho.FORCE_PREPARE_DATA,
                Clotho.VERIFY_FILES,
                Clotho.CLEAN_ARCHIVES,
            )

        # Exclude some columns containing empty values for several subsets
        column_names = list(ClothoItem.__required_keys__) + list(
            ClothoItem.__optional_keys__
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

        raw_data = _load_clotho_dataset(root, version, subset, verbose)

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
        self._subset = subset
        self._download = download
        self._version = version

        if "audio" not in removed_columns:
            self.add_post_columns(
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
    def subset(self) -> str:
        return self._subset

    @property
    def version(self) -> str:
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


def _get_clotho_dpath(root: str, version: str) -> str:
    return osp.join(root, f"CLOTHO_{version}")


def _get_archives_dpath(root: str, version: str) -> str:
    return osp.join(_get_clotho_dpath(root, version), "archives")


def _get_audio_dpath(root: str, version: str) -> str:
    return osp.join(_get_clotho_dpath(root, version), "clotho_audio_files")


def _get_csv_dpath(root: str, version: str) -> str:
    return osp.join(_get_clotho_dpath(root, version), "clotho_csv_files")


def _get_audio_subset_dpath(root: str, version: str, subset: str) -> Optional[str]:
    dname = _CLOTHO_AUDIO_DNAMES[subset]
    if dname is None:
        return None

    return osp.join(
        _get_clotho_dpath(root, version),
        "clotho_audio_files",
        dname,
    )


def _is_prepared(root: str, version: str, subset: str) -> bool:
    audio_dpath = _get_audio_dpath(root, version)
    csv_dpath = _get_csv_dpath(root, version)
    if not all(map(osp.isdir, (audio_dpath, csv_dpath))):
        return False

    if ClothoCard.CAPTIONS_PER_AUDIO[subset] == 0:
        return True
    if _CLOTHO_AUDIO_DNAMES[subset] is None:
        return True

    links = _CLOTHO_LINKS[version][subset]
    captions_fname = links["captions"]["fname"]
    captions_fpath = osp.join(csv_dpath, captions_fname)
    with open(captions_fpath, "r") as file:
        reader = csv.DictReader(file)
        lines = list(reader)

    audio_subset_dpath = _get_audio_subset_dpath(root, version, subset)
    return len(lines) == len(os.listdir(audio_subset_dpath))


def _load_clotho_dataset(
    root: str,
    version: str,
    subset: str,
    verbose: int,
) -> Dict[str, List[Any]]:
    if not _is_prepared(root, version, subset):
        raise RuntimeError(
            f"Cannot load data: clotho_{subset} is not prepared in data root={root}. Please use download=True in dataset constructor."
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
                data | {"file_name": f"no_fname_{i}"}
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
        idx = fname_to_idx[fname]
        all_captions_lst[idx] = [line[caption_key] for caption_key in captions_keys]

    all_metadata_dic: Dict[str, List[Any]] = {
        key: [None for _ in range(dataset_size)] for key in subset_metadata_keys
    }
    for line in metadata:
        fname = line["file_name"]
        if fname not in fname_to_idx:
            raise KeyError(
                f"Cannot find metadata fname={fname} in captions file. (subset={subset})"
            )
        idx = fname_to_idx[fname]
        for key in subset_metadata_keys:
            # The test subset does not have keywords in metadata, but has sound_id, sound_link, etc.
            if key in line:
                all_metadata_dic[key][idx] = line[key]

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
            f"Dataset {ClothoCard.PRETTY_NAME} ({subset}) has been loaded. (size={len(next(iter(raw_data.values())))})"
        )
    return raw_data


def _prepare_clotho_dataset(
    root: str,
    version: str,
    subset: str,
    verbose: int,
    force: bool,
    verify_files: bool,
    clean_archives: bool,
) -> None:
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
                f"Found invalid extension={extension}. Must be one of {EXTENSIONS}."
            )

        fpath = osp.join(dpath, fname)
        if not osp.isfile(fpath) or force:
            if verbose >= 1:
                pylog.info(f"Download and check file '{fname}' from url={url}...")

            download_url_to_file(
                url,
                fpath,
                progress=verbose >= 1,
            )

        elif verbose >= 1:
            pylog.info(f"File '{fname}' is already downloaded.")

        if verify_files:
            valid = validate_file(fpath, hash_value, hash_type="md5")
            if not valid:
                raise RuntimeError(f"Invalid checksum for file {fname}.")
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
                    f"Found unexpected extension={extension} for downloaded file '{fname}'. Expected one of {EXTENSIONS}."
                )
                continue

            fpath = osp.join(archives_dpath, fname)

            if verbose >= 1:
                pylog.info(f"Extract archive file fname={fname}...")

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
                        f"(with found_but_not_expected={found_but_not_expected} and expected_but_not_found={expected_but_not_found})"
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


# Audio directory names per subset
_CLOTHO_AUDIO_DNAMES = {
    "dev": "development",
    "val": "validation",
    "eval": "evaluation",
    "dcase_aac_test": "test",
    "dcase_aac_analysis": "clotho_analysis",
    "dcase_t2a_audio": "test_retrieval_audio",
    "dcase_t2a_captions": None,
}

# Archives and file links used to download Clotho
_CLOTHO_LINKS = {
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
        "test": {
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
