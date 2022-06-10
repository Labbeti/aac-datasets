#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import csv
import logging
import os
import os.path as osp

from dataclasses import asdict, astuple, dataclass, field, fields
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import torch
import torchaudio

from py7zr import SevenZipFile
from torch import nn, Tensor
from torch.utils.data.dataset import Dataset
from torchaudio.backend.common import AudioMetaData
from torchaudio.datasets.utils import download_url


logger = logging.getLogger(__name__)


@dataclass
class ClothoItem:
    audio: Tensor = torch.empty((0,))
    captions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    fname: Optional[str] = None
    index: Optional[int] = None
    dataset: str = "clotho"
    subset: Optional[str] = None
    sr: int = -1


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
    """
    Unofficial Clotho pytorch dataset for DCASE 2021 Task 6.
    Subsets available are 'train', 'val', 'eval' and 'test' (for v2 and v2.1).

    Audio are waveform sounds of 15 to 30 seconds, sampled at 44100 Hz.
    Target is a list of 5 different sentences strings describing an audio sample.
    Max number of words in captions is 20.

    Clotho V1 Paper : https://arxiv.org/pdf/1910.09387.pdf

    Dataset folder tree for version 'v2.1':

    ```
    {root}
    └── CLOTHO_v2.1
        ├── clotho_audio_files
        │   ├── clotho_analysis
        │   │    └── (8360 files, ~19GB)
        │   ├── development
        │   │    └── (3840 files, ~7.1GB)
        │   ├── evaluation
        │   │    └── (1045 files, ~2.0GB)
        │   ├── validation
        │   │    └── (1046 files, ~2.0GB)
        │   └── test
        │        └── (1044 files, ~2.0GB)
        └── clotho_csv_files
            ├── clotho_captions_development.csv
            ├── clotho_captions_evaluation.csv
            ├── clotho_captions_validation.csv
            ├── clotho_metadata_development.csv
            ├── clotho_metadata_evaluation.csv
            ├── clotho_metadata_test.csv
            └── clotho_metadata_validation.csv
    ```
    """

    AUDIO_MAX_LENGTH = 30.0  # in seconds
    AUDIO_MIN_LENGTH = 15.0  # in seconds
    AUDIO_N_CHANNELS = 1
    CAPTION_MAX_LENGTH = 20
    CAPTION_MIN_LENGTH = 8
    CAPTIONS_PER_AUDIO = {"dev": 5, "val": 5, "eval": 5, "test": 0}
    CLEAN_ARCHIVES: bool = False
    FORCE_PREPARE_DATA: bool = False
    ITEM_TYPES = ("tuple", "dict", "dataclass")
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
        transforms: Optional[Dict[str, Optional[nn.Module]]] = None,
        unfold: bool = False,
        item_type: str = "tuple",
        version: str = "v2.1",
        verbose: int = 0,
    ) -> None:
        """
        :param root: The parent of the dataset root directory.
            Note: The data is stored in the 'CLOTHO_{version}' subdirectory.
            defaults to ".".
        :param subset: The subset of Clotho to use.
            Can be 'dev', 'val', 'eval', 'test' or 'analysis' for clotho v2.1.
        :param download: Download the dataset if download=True and if the dataset is not already downloaded.
            default to False.
        :param transforms: The transform to apply values (Tensor).
            Keys can be 'audio', 'captions' or 'keywords'.
            defaults to None.
        :param unfold: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param item_type: The type of the value returned by __getitem__.
            Can be 'tuple', 'dict' or 'dataclass'.
            defaults to 'tuple'.
        :param version: The version of the dataset. Can be 'v1', 'v2' or 'v2.1'.
            defaults to 'v2.1'.
        :param verbose: Verbose level to use. Can be 0 or 1.
            defaults to 0.
        """
        if version not in Clotho.VERSIONS:
            raise ValueError(
                f"Invalid Clotho argument {version=}. Must be one of {Clotho.VERSIONS}."
            )

        if subset not in CLOTHO_LINKS[version].keys():
            raise ValueError(
                f"Invalid Clotho argument {subset=} for {version=}. Must be one of {tuple(CLOTHO_LINKS[version].keys())}."
            )

        if item_type not in self.ITEM_TYPES:
            raise ValueError(
                f"Invalid argument {item_type=} for Clotho. (expected one of {self.ITEM_TYPES})"
            )

        if transforms is None:
            transforms = {}

        accepted_keys = [field_.name for field_ in fields(ClothoItem)]
        for key in transforms.keys():
            if key not in accepted_keys:
                raise ValueError(
                    f"Invalid transform {key=}. (expected one of {accepted_keys})"
                )

        super().__init__()
        self._root = root
        self._subset = subset
        self._download = download
        self._transforms = transforms
        self._unfold = unfold
        self._item_type = item_type
        self._version = version
        self._verbose = verbose

        self._all_infos = []
        self._metadata_keys = []

        if self._download:
            self._prepare_data()
        self._load_data()

    def __getitem__(self, index: int) -> Union[Tuple, Dict, ClothoItem]:
        """
        Get the audio data as 1D tensor and the matching captions as 5 sentences.

        :param index: The index of the item.
        :return: Audio data of shape (size,) and the 5 matching captions as tuple or dict.
        """
        kwargs = {
            field.name: self.get(field.name, index) for field in fields(ClothoItem)
        }
        item = ClothoItem(**kwargs)

        if self._item_type == "tuple":
            return astuple(item)
        elif self._item_type == "dict":
            return asdict(item)
        elif self._item_type == "dataclass":
            return item
        else:
            raise ValueError(
                f"Invalid item_type={self._item_type} for Clotho_{self._subset}. (expected one of {self.ITEM_TYPES})"
            )

    def __len__(self) -> int:
        """
        :return: The number of items in the dataset.
        """
        return len(self._all_infos)

    def get_valid_item_keys(self) -> Tuple[str, ...]:
        return ("audio", "fpath", "dataset", "subset", "index") + tuple(
            self._all_infos[0].keys()
        )

    def get_raw(self, name: str, index: int) -> Any:
        """Read the processed raw data. (without transform)

        :param name: The name of the value.
                Can be 'audio', 'fpath', 'fname', 'captions', 'keywords', 'sound_id', 'sound_link', 'start_end_samples', 'manufacturer', 'license'.
        :param index: The index of the value in range [0, len(dataset)[.
        """
        if not (0 <= index < len(self)):
            raise IndexError(
                f"Invalid argument {index=} for {self} (expected in range [0, {len(self)}-1])"
            )

        if name == "audio":
            fpath = self.get("fpath", index)
            value, sr = torchaudio.load(fpath)  # type: ignore

            # Sanity check
            if sr != self.SAMPLE_RATE:
                raise RuntimeError(
                    f"Invalid sample rate {sr}Hz of audio file {fpath} with Clotho {self.SAMPLE_RATE}Hz."
                )

        elif name == "fpath":
            fname = self.get("fname", index)
            value = osp.join(self._dpath_audio_subset, fname)

        elif name == "dataset":
            value = "clotho"

        elif name == "subset":
            value = self._subset

        elif name == "index":
            value = index

        elif name == "sr":
            fpath = self.get("fpath", index)
            audio_info: AudioMetaData = torchaudio.info(fpath)  # type: ignore
            return audio_info.sample_rate

        elif name in self._all_infos[index].keys():
            value = self._all_infos[index][name]

        else:
            raise ValueError(
                f"Invalid value {name=} at {index=}. (dataset=clotho, subset={self._subset}, len={len(self)})"
            )
        return value

    def get(self, name: str, index: int) -> Any:
        """Read the processed data. (with transform)

        :param name: The name of the value.
                Can be 'audio', 'fpath', 'fname', 'captions', 'keywords', 'sound_id', 'sound_link', 'start_end_samples', 'manufacturer', 'license'.
        :param index: The index of the value in range [0, len(dataset)[.
        """
        value = self.get_raw(name, index)
        transform = self._transforms.get(name, None)
        if transform is not None:
            value = transform(value)
        return value

    @cached_property
    def _dpath_data(self) -> str:
        return osp.join(self._root, f"CLOTHO_{self._version}")

    @cached_property
    def _dpath_audio(self) -> str:
        return osp.join(self._dpath_data, "clotho_audio_files")

    @cached_property
    def _dpath_audio_subset(self) -> str:
        return osp.join(
            self._dpath_data,
            "clotho_audio_files",
            CLOTHO_AUDIO_DNAMES[self._subset],
        )

    @cached_property
    def _dpath_csv(self) -> str:
        return osp.join(self._dpath_data, "clotho_csv_files")

    def _is_prepared(self) -> bool:
        if not all(map(osp.isdir, (self._dpath_audio_subset, self._dpath_csv))):
            return False

        if self._subset in ("test", "analysis"):
            return True

        links = CLOTHO_LINKS[self._version][self._subset]
        captions_fname = links["captions"]["fname"]
        captions_fpath = osp.join(self._dpath_csv, captions_fname)
        with open(captions_fpath, "r") as file:
            reader = csv.DictReader(file)
            lines = list(reader)
        return len(lines) == len(os.listdir(self._dpath_audio_subset))

    def _prepare_data(self) -> None:
        if not osp.isdir(self._root):
            raise RuntimeError(f"Cannot find root directory '{self._root}'.")

        os.makedirs(self._dpath_audio, exist_ok=True)
        os.makedirs(self._dpath_csv, exist_ok=True)

        if self._verbose >= 1:
            logger.info(f"Start to download files for clotho_{self._subset}...")

        links = copy.deepcopy(CLOTHO_LINKS[self._version][self._subset])
        extensions = ("7z", "csv", "zip")

        # Download csv and 7z files
        for info in links.values():
            fname, url, hash_value = info["fname"], info["url"], info["hash_value"]
            extension = fname.split(".")[-1]

            if extension in ("7z", "zip"):
                dpath = self._dpath_audio
            elif extension == "csv":
                dpath = self._dpath_csv
            else:
                raise RuntimeError(
                    f"Found invalid {extension=}. Must be one of {extensions}."
                )

            fpath = osp.join(dpath, fname)
            if not osp.isfile(fpath) or self.FORCE_PREPARE_DATA:
                if self._verbose >= 1:
                    logger.info(f"Download and check file '{fname}' from {url=}...")

                download_url(
                    url,
                    dpath,
                    fname,
                    hash_value=hash_value,
                    hash_type="md5",
                    progress_bar=self._verbose >= 1,
                )

            elif self._verbose >= 1:
                logger.info(f"File {fname=} is already extracted.")

        # Extract audio files from archives
        for info in links.values():
            fname = info["fname"]
            extension = fname.split(".")[-1]

            if extension in ("7z", "zip"):
                fpath = osp.join(self._dpath_audio, fname)

                if self._verbose >= 1:
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
                    raise RuntimeError

                # Ignore dir name from archive file
                compressed_fnames = [
                    fname
                    for fname in compressed_fnames
                    if fname not in ("", CLOTHO_AUDIO_DNAMES[self._subset])
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
            for info in links.values():
                fname = info["fname"]
                extension = fname.split(".")[-1]

                if extension in ("7z", "zip"):
                    fpath = osp.join(self._dpath_audio, fname)
                    if self._verbose >= 1:
                        logger.info(f"Removing archive file {osp.basename(fpath)}...")
                    os.remove(fpath)

    def _load_data(self) -> None:
        if not self._is_prepared():
            raise RuntimeError(
                f"Cannot load data: clotho_{self._subset} is not prepared in data root={self._root}. Please use download=True in dataset constructor."
            )

        # Read fpath of .wav audio files
        links = CLOTHO_LINKS[self._version][self._subset]
        dpath_audio_subset = self._dpath_audio_subset

        if not osp.isdir(dpath_audio_subset):
            raise RuntimeError(f'Cannot find directory "{dpath_audio_subset}".')

        self._all_infos = [
            {
                "fname": fname,
                "captions": [],
                "keywords": [],
            }
            for fname in os.listdir(dpath_audio_subset)
        ]
        idx_to_fname = {i: infos["fname"] for i, infos in enumerate(self._all_infos)}
        fname_to_idx = {fname: i for i, fname in idx_to_fname.items()}

        # --- Read captions info
        if "captions" in links.keys():
            captions_fname = links["captions"]["fname"]
            captions_fpath = osp.join(self._dpath_csv, captions_fname)

            with open(captions_fpath, "r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Keys: file_name, caption_1, caption_2, caption_3, caption_4, caption_5
                    fname = row["file_name"]

                    if fname not in fname_to_idx.keys():
                        raise RuntimeError(
                            f"Found '{fname=}' in CSV '{captions_fname=}' but not the corresponding audio file."
                        )
                    else:
                        idx = fname_to_idx[fname]
                        self._all_infos[idx]["captions"] = list(
                            row[caption_key]
                            for caption_key in (
                                "caption_1",
                                "caption_2",
                                "caption_3",
                                "caption_4",
                                "caption_5",
                            )
                        )

        # --- Read metadata info
        if "metadata" in links.keys():
            metadata_fname = links["metadata"]["fname"]
            metadata_fpath = osp.join(self._dpath_csv, metadata_fname)

            # Keys: file_name, keywords, sound_id, sound_link, start_end_samples, manufacturer, license
            if self._version in ("v2", "v2.1"):
                encoding = "ISO-8859-1"
            else:
                encoding = None

            with open(metadata_fpath, "r", encoding=encoding) as file:
                delimiter = ";" if self._subset == "test" else ","
                reader = csv.DictReader(file, delimiter=delimiter)
                self._metadata_keys = (
                    reader.fieldnames if reader.fieldnames is not None else ()
                )

                for row in reader:
                    # file_name,keywords,sound_id,sound_link,start_end_samples,manufacturer,license
                    fname = row["file_name"]
                    row_copy: Any = copy.deepcopy(row)
                    row_copy.pop("file_name")

                    if "keywords" in row_copy.keys():
                        row_copy["keywords"] = row_copy["keywords"].split(";")
                    else:
                        row_copy["keywords"] = []

                    if fname in fname_to_idx.keys():
                        idx = fname_to_idx[fname]
                        self._all_infos[idx].update(row_copy)
                    else:
                        raise RuntimeError(
                            f"Found value {fname=} in CSV metadata file '{metadata_fname}' but not the corresponding audio file."
                        )

        if self._unfold and self.CAPTIONS_PER_AUDIO[self._subset] > 1:
            data_info_unfolded = []
            for infos in self._all_infos:
                captions = infos["captions"]
                for caption in captions:
                    new_infos = dict(infos)
                    new_infos["captions"] = [caption]
                    data_info_unfolded.append(new_infos)

            self._all_infos = data_info_unfolded

        self._all_infos = self._all_infos
        if self._verbose >= 1:
            logger.info(f"clotho_{self._subset} has been loaded. (len={len(self)})")

    def __repr__(self) -> str:
        return f"Clotho(subset={self._subset})"


CLOTHO_AUDIO_DNAMES = {
    "dev": "development",
    "eval": "evaluation",
    "test": "test",
    "val": "validation",
    "analysis": "clotho_analysis",
}
