#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import csv
import logging
import os
import os.path as osp

from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
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

CAPTIONS_KEYS = (
    "caption_1",
    "caption_2",
    "caption_3",
    "caption_4",
    "caption_5",
)
METADATA_KEYS = (
    "keywords",
    "sound_id",
    "sound_link",
    "start_end_samples",
    "manufacturer",
    "license",
)


class Clotho(Dataset[Dict[str, Any]]):
    r"""
    Unofficial Clotho PyTorch dataset.
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

    # Initialization
    def __init__(
        self,
        root: str = ".",
        subset: str = "dev",
        download: bool = False,
        transform: Optional[Callable] = None,
        flat_captions: bool = False,
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
        :param transform: The transform to apply to the global dict item. This transform is applied only in getitem method.
            defaults to None.
        :param flat_captions: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param verbose: Verbose level to use. Can be 0 or 1.
            defaults to 0.
        :param version: The version of the dataset. Can be one of :attr:`~Clotho.VERSIONS`.
            defaults to 'v2.1'.
        """
        if version not in Clotho.VERSIONS:
            raise ValueError(
                f"Invalid Clotho argument version={version}. Must be one of {Clotho.VERSIONS}."
            )

        subsets = tuple(CLOTHO_LINKS[version].keys())
        if subset not in subsets:
            raise ValueError(
                f"Invalid Clotho argument subset={subset} for version={version}. Must be one of {subsets}."
            )

        super().__init__()
        self._root = root
        self._subset = subset
        self._download = download
        self._transform = transform
        self._flat_captions = flat_captions
        self._version = version
        self._verbose = verbose

        self._all_items = {}
        self._loaded = False

        if self._download:
            self.__prepare_data()
        self.__load_data()

    # Properties
    @property
    def column_names(self) -> List[str]:
        """The name of each column of the dataset."""
        return [
            field_.name
            for field_ in fields(ClothoItem)
            if field_.name in self._all_items or field_.name not in METADATA_KEYS
        ]

    @property
    def info(self) -> Dict[str, Any]:
        """Return the global dataset info."""
        return {
            "dataset": "clotho",
            "subset": self._subset,
            "version": self._version,
        }

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the Clotho dataset."""
        return len(self), len(self.column_names)

    # Public methods
    def at(
        self,
        idx: Union[int, Iterable[int], None, slice] = None,
        column: Union[str, Iterable[str], None] = None,
    ) -> Any:
        """Get a specific data field.

        :param index: The index or slice of the value in range [0, len(dataset)-1].
        :param column: The name(s) of the column. Can be any value of :meth:`~Clotho.column_names`.
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

        elif column == "dataset":
            return "clotho"

        elif column == "fpath":
            fname = self.at(idx, "fname")
            fpath = osp.join(self.__dpath_audio_subset, fname)
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
        """
        :return: The number of items in the dataset.
        """
        return len(self._all_items["captions"])

    def __repr__(self) -> str:
        return f"Clotho(size={len(self)}, subset={self._subset}, num_columns={len(self.column_names)}, version={self._version})"

    # Private methods
    @property
    @lru_cache()
    def __dpath_archives(self) -> str:
        return osp.join(self.__dpath_data, "archives")

    @property
    @lru_cache()
    def __dpath_audio(self) -> str:
        return osp.join(self.__dpath_data, "clotho_audio_files")

    @property
    @lru_cache()
    def __dpath_audio_subset(self) -> str:
        return osp.join(
            self.__dpath_data,
            "clotho_audio_files",
            CLOTHO_AUDIO_DNAMES[self._subset],
        )

    @property
    @lru_cache()
    def __dpath_csv(self) -> str:
        return osp.join(self.__dpath_data, "clotho_csv_files")

    @property
    @lru_cache()
    def __dpath_data(self) -> str:
        return osp.join(self._root, f"CLOTHO_{self._version}")

    def __is_prepared(self) -> bool:
        if not all(map(osp.isdir, (self.__dpath_audio_subset, self.__dpath_csv))):
            return False

        if Clotho.CAPTIONS_PER_AUDIO[self._subset] == 0:
            return True

        links = CLOTHO_LINKS[self._version][self._subset]
        captions_fname = links["captions"]["fname"]
        captions_fpath = osp.join(self.__dpath_csv, captions_fname)
        with open(captions_fpath, "r") as file:
            reader = csv.DictReader(file)
            lines = list(reader)
        return len(lines) == len(os.listdir(self.__dpath_audio_subset))

    def __load_data(self) -> None:
        if not self.__is_prepared():
            raise RuntimeError(
                f"Cannot load data: clotho_{self._subset} is not prepared in data root={self._root}. Please use download=True in dataset constructor."
            )

        # Read fpath of .wav audio files
        links = CLOTHO_LINKS[self._version][self._subset]
        dpath_audio_subset = self.__dpath_audio_subset

        if not osp.isdir(dpath_audio_subset):
            raise RuntimeError(f'Cannot find directory "{dpath_audio_subset}".')

        # Read Clotho files
        if "captions" in links.keys():
            captions_fname = links["captions"]["fname"]
            captions_fpath = osp.join(self.__dpath_csv, captions_fname)

            # Keys: file_name, caption_1, caption_2, caption_3, caption_4, caption_5
            with open(captions_fpath, "r") as file:
                reader = csv.DictReader(file)
                captions_data = list(reader)
        else:
            captions_data = []

        if "metadata" in links.keys():
            metadata_fname = links["metadata"]["fname"]
            metadata_fpath = osp.join(self.__dpath_csv, metadata_fname)

            # Keys: file_name, keywords, sound_id, sound_link, start_end_samples, manufacturer, license
            if self._version in ("v2", "v2.1"):
                encoding = "ISO-8859-1"
            else:
                encoding = None

            with open(metadata_fpath, "r", encoding=encoding) as file:
                delimiter = ";" if self._subset == "test" else ","
                reader = csv.DictReader(file, delimiter=delimiter)
                metadata = list(reader)
        else:
            metadata = []

        if "captions" in links.keys():
            # note: "dev", "val", "eval"
            fnames_lst = [line["file_name"] for line in captions_data]
        elif "metadata" in links.keys():
            # note: for "test" subset which do not have captions CSV file
            fnames_lst = [line["file_name"] for line in metadata]
        else:
            # note 1: for "analysis" subset which do not have any CSV file
            # note 2: force sorted list to have the same order on all OS
            fnames_lst = list(sorted(os.listdir(dpath_audio_subset)))

        idx_to_fname = {i: fname for i, fname in enumerate(fnames_lst)}
        fname_to_idx = {fname: i for i, fname in idx_to_fname.items()}
        dataset_size = len(fnames_lst)

        # Process each item field
        if len(metadata) > 0:
            subset_metadata_keys = [key for key in METADATA_KEYS if key in metadata[0]]
        else:
            subset_metadata_keys = []

        all_captions_lst = [[] for _ in range(dataset_size)]
        for line in captions_data:
            fname = line["file_name"]
            idx = fname_to_idx[fname]
            all_captions_lst[idx] = [line[caption_key] for caption_key in CAPTIONS_KEYS]

        all_metadata_dic: Dict[str, List[Any]] = {
            key: [None for _ in range(dataset_size)] for key in subset_metadata_keys
        }
        for line in metadata:
            fname = line["file_name"]
            if fname not in fname_to_idx:
                raise KeyError(
                    f"Cannot find metadata fname={fname} in captions file. (subset={self._subset})"
                )
            idx = fname_to_idx[fname]
            for key in subset_metadata_keys:
                # The test subset does not have keywords in metadata, but has sound_id, sound_link, etc.
                if key in line:
                    all_metadata_dic[key][idx] = line[key]

        all_items = {
            "fname": fnames_lst,
            "captions": all_captions_lst,
        }
        all_items.update(all_metadata_dic)

        if "keywords" in all_items:
            # Split keywords into list[str]
            all_items["keywords"] = [
                keywords.split(";") if keywords is not None else []
                for keywords in all_items["keywords"]
            ]

        if self._flat_captions and self.CAPTIONS_PER_AUDIO[self._subset] > 1:
            all_infos_unfolded = {key: [] for key in all_items.keys()}

            for i, captions in enumerate(all_items["captions"]):
                for caption in captions:
                    for key in all_items.keys():
                        all_infos_unfolded[key].append(all_items[key][i])
                    all_infos_unfolded["captions"] = [caption]

            all_items = all_infos_unfolded

        self._all_items = all_items
        self._loaded = True

        if self._verbose >= 1:
            logger.info(f"{repr(self)} has been loaded. (len={len(self)})")

    def __prepare_data(self) -> None:
        if not osp.isdir(self._root):
            raise RuntimeError(f"Cannot find root directory '{self._root}'.")

        os.makedirs(self.__dpath_archives, exist_ok=True)
        os.makedirs(self.__dpath_audio, exist_ok=True)
        os.makedirs(self.__dpath_csv, exist_ok=True)

        if self._verbose >= 1:
            logger.info(f"Start to download files for clotho_{self._subset}...")

        links = copy.deepcopy(CLOTHO_LINKS[self._version][self._subset])
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
                dpath = self.__dpath_archives
            elif extension == "csv":
                dpath = self.__dpath_csv
            else:
                raise RuntimeError(
                    f"Found invalid extension={extension}. Must be one of {extensions}."
                )

            fpath = osp.join(dpath, fname)
            if not osp.isfile(fpath) or self.FORCE_PREPARE_DATA:
                if self._verbose >= 1:
                    logger.info(f"Download and check file '{fname}' from url={url}...")

                download_url_to_file(
                    url,
                    fpath,
                    progress=self._verbose >= 1,
                )

            elif self._verbose >= 1:
                logger.info(f"File fname={fname} is already extracted.")

            with open(fpath, "rb") as file:
                valid = validate_file(file, hash_value, hash_type="md5")
            if not valid:
                raise RuntimeError(f"Invalid checksum for file {fname}.")

        # Extract audio files from archives
        for file_info in links.values():
            fname = file_info["fname"]
            extension = fname.split(".")[-1]

            if extension in ("7z", "zip"):
                fpath = osp.join(self.__dpath_archives, fname)

                if self._verbose >= 1:
                    logger.info(f"Extract archive file fname={fname}...")

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
                    if fname not in ("", CLOTHO_AUDIO_DNAMES[self._subset])
                ]
                extracted_fnames = (
                    os.listdir(self.__dpath_audio_subset)
                    if osp.isdir(self.__dpath_audio_subset)
                    else []
                )

                if set(extracted_fnames) != set(compressed_fnames):
                    archive_file.extractall(self.__dpath_audio)

                    # Check if files is good now
                    extracted_fnames = os.listdir(self.__dpath_audio_subset)
                    if set(extracted_fnames) != set(compressed_fnames):
                        raise RuntimeError(
                            f"Invalid number of audios extracted. (found {len(extracted_fnames)} files but expected the same {len(compressed_fnames)} files)"
                        )

                archive_file.close()

            elif extension == "csv":
                pass

            else:
                logger.error(
                    f"Found unexpected extension={extension} for downloaded file '{fname}'. Expected one of {extensions}."
                )

        if self.CLEAN_ARCHIVES:
            for file_info in links.values():
                fname = file_info["fname"]
                extension = fname.split(".")[-1]

                if extension in ("7z", "zip"):
                    fpath = osp.join(self.__dpath_audio, fname)
                    if self._verbose >= 1:
                        logger.info(f"Removing archive file {osp.basename(fpath)}...")
                    os.remove(fpath)


CLOTHO_AUDIO_DNAMES = {
    "dev": "development",
    "eval": "evaluation",
    "test": "test",
    "val": "validation",
    "analysis": "clotho_analysis",
}
