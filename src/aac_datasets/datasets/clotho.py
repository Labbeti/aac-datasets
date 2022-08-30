#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import csv
import logging
import os
import os.path as osp

from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from zipfile import ZipFile

import torch
import torchaudio

from py7zr import SevenZipFile
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data.dataset import Dataset
from torchaudio.datasets.utils import validate_file


logger = logging.getLogger(__name__)


@dataclass
class ClothoItem:
    """Dataclass representing a single Clotho item."""

    # Common attributes
    audio: Tensor = torch.empty((0,))
    captions: List[str] = field(default_factory=list)
    dataset: str = "clotho"
    fname: str = "unknown"
    index: int = -1
    subset: str = "unknown"
    sr: int = -1
    # Clotho-specific attributes
    keywords: List[str] = field(default_factory=list)
    sound_id: str = "unknown"
    sound_link: str = "unknown"
    start_end_samples: str = "unknown"
    manufacturer: str = "unknown"
    license: str = "unknown"


CLOTHO_LINKS = {
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
        "analysis": {
            "audio_archive": {
                "fname": "clotho_analysis_2022.zip",
                "url": "https://zenodo.org/record/6610709/files/clotho_analysis_2022.zip?download=1",
                "hash_value": "7e8fa4762cc3a7c5546606680b958d08",
            },
        },
    },
}
CLOTHO_LAST_VERSION = "v2.1"


class Clotho(Dataset):
    r"""
    Unofficial Clotho pytorch dataset.
    Subsets available are 'train', 'val', 'eval', 'test' and 'analysis'.

    Audio are waveform sounds of 15 to 30 seconds, sampled at 44100 Hz.
    Target is a list of 5 different sentences strings describing an audio sample.
    The maximal number of words in captions is 20.

    Clotho V1 Paper : https://arxiv.org/pdf/1910.09387.pdf

    .. code-block:: text
        :caption:  Dataset folder tree for version 'v2.1'

        {root}
        └── CLOTHO_v2.1
            ├── clotho_audio_files
            │   ├── clotho_analysis
            │   │    └── (8360 wav files, ~19GB)
            │   ├── development
            │   │    └── (3840 wav files, ~7.1GB)
            │   ├── evaluation
            │   │    └── (1045 wav files, ~2.0GB)
            │   ├── validation
            │   │    └── (1046 wav files, ~2.0GB)
            │   └── test
            │        └── (1044 wav files, ~2.0GB)
            └── clotho_csv_files
                ├── clotho_captions_development.csv
                ├── clotho_captions_evaluation.csv
                ├── clotho_captions_validation.csv
                ├── clotho_metadata_development.csv
                ├── clotho_metadata_evaluation.csv
                ├── clotho_metadata_test.csv
                └── clotho_metadata_validation.csv

    """

    # Global
    AUDIO_MAX_SEC = 30.0
    AUDIO_MIN_SEC = 15.0
    AUDIO_N_CHANNELS = 1
    CAPTION_MAX_LENGTH = 20
    CAPTION_MIN_LENGTH = 8
    CAPTIONS_PER_AUDIO = {"dev": 5, "val": 5, "eval": 5, "test": 0, "analysis": 0}
    CLEAN_ARCHIVES: bool = False
    FORCE_PREPARE_DATA: bool = False
    SAMPLE_RATE = 44100
    SUBSETS_DICT = {
        version: tuple(links.keys()) for version, links in CLOTHO_LINKS.items()
    }
    SUBSETS = SUBSETS_DICT[CLOTHO_LAST_VERSION]
    VERSIONS = tuple(CLOTHO_LINKS.keys())

    def __init__(
        self,
        root: str = ".",
        subset: str = "dev",
        download: bool = False,
        item_transform: Optional[Callable] = None,
        unfold: bool = False,
        verbose: int = 0,
        version: str = "v2.1",
    ) -> None:
        """
        :param root: The parent of the dataset root directory.
            Note: The data is stored in the 'CLOTHO_{version}' subdirectory.
            defaults to ".".
        :param subset: The subset of Clotho to use. Can be one of :attr:`~Clotho.SUBSETS`.
            defaults to "dev".
        :param download: Download the dataset if download=True and if the dataset is not already downloaded.
            defaults to False.
        :param item_transform: The transform to apply to the global dict item. This transform is applied AFTER each field transform.
            defaults to None.
        :param unfold: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param verbose: Verbose level to use. Can be 0 or 1.
            defaults to 0.
        :param version: The version of the dataset. Can be one of :attr:`~Clotho.VERSIONS`.
            defaults to 'v2.1'.
        """
        if version not in Clotho.VERSIONS:
            raise ValueError(
                f"Invalid Clotho argument {version=}. Must be one of {Clotho.VERSIONS}."
            )

        if subset not in CLOTHO_LINKS[version].keys():
            raise ValueError(
                f"Invalid Clotho argument {subset=} for {version=}. Must be one of {tuple(CLOTHO_LINKS[version].keys())}."
            )

        super().__init__()
        self.__root = root
        self.__subset = subset
        self.__download = download
        self.__item_transform = item_transform
        self.__unfold = unfold
        self.__version = version
        self.__verbose = verbose

        self.__all_items = {}
        self.__is_loaded = False

        if self.__download:
            self._prepare_data()
        self._load_data()

    def get_field(
        self, key: str, index: Union[int, slice, Iterable[int]] = slice(None)
    ) -> Any:
        """Get a specific data field.

        :param key: The name of the field. Can be any attribute name of :class:`~ClothoItem`.
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

        elif key == "dataset":
            return "clotho"

        elif key == "fpath":
            fname = self.get_field("fname", index)
            fpath = osp.join(self._dpath_audio_subset, fname)
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
            keys = [field.name for field in fields(ClothoItem)]
            raise ValueError(
                f"Invalid argument {key=} at {index=}. (expected one of {tuple(keys)})"
            )

    @cached_property
    def _dpath_archives(self) -> str:
        return osp.join(self._dpath_data, "archives")

    @cached_property
    def _dpath_audio(self) -> str:
        return osp.join(self._dpath_data, "clotho_audio_files")

    @cached_property
    def _dpath_audio_subset(self) -> str:
        return osp.join(
            self._dpath_data,
            "clotho_audio_files",
            CLOTHO_AUDIO_DNAMES[self.__subset],
        )

    @cached_property
    def _dpath_csv(self) -> str:
        return osp.join(self._dpath_data, "clotho_csv_files")

    @cached_property
    def _dpath_data(self) -> str:
        return osp.join(self.__root, f"CLOTHO_{self.__version}")

    def _is_loaded(self) -> bool:
        return self.__is_loaded

    def _is_prepared(self) -> bool:
        if not all(map(osp.isdir, (self._dpath_audio_subset, self._dpath_csv))):
            return False

        if Clotho.CAPTIONS_PER_AUDIO[self.__subset] == 0:
            return True

        links = CLOTHO_LINKS[self.__version][self.__subset]
        captions_fname = links["captions"]["fname"]
        captions_fpath = osp.join(self._dpath_csv, captions_fname)
        with open(captions_fpath, "r") as file:
            reader = csv.DictReader(file)
            lines = list(reader)
        return len(lines) == len(os.listdir(self._dpath_audio_subset))

    def _load_data(self) -> None:
        if not self._is_prepared():
            raise RuntimeError(
                f"Cannot load data: clotho_{self.__subset} is not prepared in data root={self.__root}. Please use download=True in dataset constructor."
            )

        # Read fpath of .wav audio files
        links = CLOTHO_LINKS[self.__version][self.__subset]
        dpath_audio_subset = self._dpath_audio_subset

        if not osp.isdir(dpath_audio_subset):
            raise RuntimeError(f'Cannot find directory "{dpath_audio_subset}".')

        all_fnames_lst = os.listdir(dpath_audio_subset)
        idx_to_fname = {i: fname for i, fname in enumerate(all_fnames_lst)}
        fname_to_idx = {fname: i for i, fname in idx_to_fname.items()}
        dataset_size = len(all_fnames_lst)

        # Read Clotho files
        if "captions" in links.keys():
            captions_fname = links["captions"]["fname"]
            captions_fpath = osp.join(self._dpath_csv, captions_fname)

            with open(captions_fpath, "r") as file:
                reader = csv.DictReader(file)
                captions_data = list(reader)
        else:
            captions_data = []

        METADATA_KEYS = (
            "keywords",
            "sound_id",
            "sound_link",
            "start_end_samples",
            "manufacturer",
            "license",
        )
        if "metadata" in links.keys():
            metadata_fname = links["metadata"]["fname"]
            metadata_fpath = osp.join(self._dpath_csv, metadata_fname)

            # Keys: file_name, keywords, sound_id, sound_link, start_end_samples, manufacturer, license
            if self.__version in ("v2", "v2.1"):
                encoding = "ISO-8859-1"
            else:
                encoding = None

            with open(metadata_fpath, "r", encoding=encoding) as file:
                delimiter = ";" if self.__subset == "test" else ","
                reader = csv.DictReader(file, delimiter=delimiter)
                metadata = list(reader)
        else:
            metadata = []

        # Process each item field
        all_captions_lst = [[] for _ in range(dataset_size)]
        captions_keys = (
            "caption_1",
            "caption_2",
            "caption_3",
            "caption_4",
            "caption_5",
        )
        for line in captions_data:
            fname = line["file_name"]
            idx = fname_to_idx[fname]
            all_captions_lst[idx] = [line[caption_key] for caption_key in captions_keys]

        all_metadata_dic: Dict[str, List[Any]] = {
            key: [None for _ in range(dataset_size)] for key in METADATA_KEYS
        }
        for line in metadata:
            fname = line["file_name"]
            idx = fname_to_idx[fname]
            for key in METADATA_KEYS:
                # The test subset does not have keywords in metadata, but has sound_id, sound_link, etc.
                if key in line:
                    all_metadata_dic[key][idx] = line[key]

        all_items = {
            "fname": all_fnames_lst,
            "captions": all_captions_lst,
        } | all_metadata_dic

        # Split keywords into list[str]
        all_items["keywords"] = [
            keywords.split(";") if keywords is not None else []
            for keywords in all_items["keywords"]
        ]

        if self.__unfold and self.CAPTIONS_PER_AUDIO[self.__subset] > 1:
            all_infos_unfolded = {key: [] for key in all_items.keys()}

            for i, captions in enumerate(all_items["captions"]):
                for caption in captions:
                    for key in all_items.keys():
                        all_infos_unfolded[key].append(all_items[key][i])
                    all_infos_unfolded["captions"] = [caption]

            all_items = all_infos_unfolded

        self.__all_items = all_items
        self.__is_loaded = True

        if self.__verbose >= 1:
            logger.info(f"{repr(self)} has been loaded. (len={len(self)})")

    def _prepare_data(self) -> None:
        if not osp.isdir(self.__root):
            raise RuntimeError(f"Cannot find root directory '{self.__root}'.")

        os.makedirs(self._dpath_archives, exist_ok=True)
        os.makedirs(self._dpath_audio, exist_ok=True)
        os.makedirs(self._dpath_csv, exist_ok=True)

        if self.__verbose >= 1:
            logger.info(f"Start to download files for clotho_{self.__subset}...")

        links = copy.deepcopy(CLOTHO_LINKS[self.__version][self.__subset])
        extensions = ("7z", "csv", "zip")

        # Download csv and 7z files
        for file_info in links.values():
            fname, url, hash_value = (
                file_info["fname"],
                file_info["url"],
                file_info["hash_value"],
            )
            extension = fname.split(".")[-1]

            if extension in ("7z", "zip"):
                dpath = self._dpath_archives
            elif extension == "csv":
                dpath = self._dpath_csv
            else:
                raise RuntimeError(
                    f"Found invalid {extension=}. Must be one of {extensions}."
                )

            fpath = osp.join(dpath, fname)
            if not osp.isfile(fpath) or self.FORCE_PREPARE_DATA:
                if self.__verbose >= 1:
                    logger.info(f"Download and check file '{fname}' from {url=}...")

                download_url_to_file(
                    url,
                    fpath,
                    fname,
                    progress=self.__verbose >= 1,
                )

            elif self.__verbose >= 1:
                logger.info(f"File {fname=} is already extracted.")

            with open(fpath, "rb") as file:
                valid = validate_file(file, hash_value, hash_type="md5")
            if not valid:
                raise RuntimeError(f"Invalid checksum for file {fname}.")

        # Extract audio files from archives
        for file_info in links.values():
            fname = file_info["fname"]
            extension = fname.split(".")[-1]

            if extension in ("7z", "zip"):
                fpath = osp.join(self._dpath_archives, fname)

                if self.__verbose >= 1:
                    logger.info(f"Extract archive file {fname=}...")

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
                    fname
                    for fname in compressed_fnames
                    if fname not in ("", CLOTHO_AUDIO_DNAMES[self.__subset])
                ]
                extracted_fnames = (
                    os.listdir(self._dpath_audio_subset)
                    if osp.isdir(self._dpath_audio_subset)
                    else []
                )

                if set(extracted_fnames) != set(compressed_fnames):
                    archive_file.extractall(self._dpath_audio)

                    # Check if files is good now
                    extracted_fnames = os.listdir(self._dpath_audio_subset)
                    if set(extracted_fnames) != set(compressed_fnames):
                        raise RuntimeError(
                            f"Invalid number of audios extracted. (found {len(extracted_fnames)} files but expected the same {len(compressed_fnames)} files)"
                        )

                archive_file.close()

            elif extension == "csv":
                pass

            else:
                logger.error(
                    f"Found unexpected {extension=} for downloaded file '{fname}'. Expected one of {extensions}."
                )

        if self.CLEAN_ARCHIVES:
            for file_info in links.values():
                fname = file_info["fname"]
                extension = fname.split(".")[-1]

                if extension in ("7z", "zip"):
                    fpath = osp.join(self._dpath_audio, fname)
                    if self.__verbose >= 1:
                        logger.info(f"Removing archive file {osp.basename(fpath)}...")
                    os.remove(fpath)

    def __getitem__(self, index: Union[int, slice]) -> Dict[str, Any]:
        keys = [field.name for field in fields(ClothoItem)]
        item = {key: self.get_field(key, index) for key in keys}
        if self.__item_transform is not None:
            item = self.__item_transform(item)
        return item

    def __len__(self) -> int:
        """
        :return: The number of items in the dataset.
        """
        return len(self.__all_items["captions"])

    def __repr__(self) -> str:
        return f"Clotho(subset={self.__subset})"


CLOTHO_AUDIO_DNAMES = {
    "dev": "development",
    "eval": "evaluation",
    "test": "test",
    "val": "validation",
    "analysis": "clotho_analysis",
}
