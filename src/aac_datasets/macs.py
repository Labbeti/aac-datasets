#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import tqdm
import os
import os.path as osp
import shutil
import zipfile

from dataclasses import asdict, astuple, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torchaudio
import yaml

from torch import nn, Tensor
from torch.utils.data.dataset import Dataset
from torchaudio.datasets.utils import download_url


@dataclass
class MACSItem:
    audio: Tensor = torch.empty((0,))
    captions: List[str] = field(default_factory=list)
    tags: List[List[str]] = field(default_factory=list)
    annotators_ids: List[str] = field(default_factory=list)
    fname: Optional[str] = None
    index: Optional[int] = None
    dataset: str = "macs"


class MACS(Dataset):
    SAMPLE_RATE = 48000  # in Hz
    MIN_CAPTIONS_PER_AUDIO = {"full": 2}
    MAX_CAPTIONS_PER_AUDIO = {"full": 5}
    AUDIO_MAX_LENGTH = 10.0  # in seconds
    AUDIO_N_CHANNELS = 2
    SUBSETS = ("full",)
    ITEM_TYPES = ("tuple", "dict", "dataclass", MACSItem.__name__.lower())

    def __init__(
        self,
        root: str = ".",
        download: bool = False,
        transforms: Optional[Dict[str, Optional[nn.Module]]] = None,
        unfold: bool = False,
        item_type: Union[str, Type] = "tuple",
        multihot_tags: bool = False,
        verbose: int = 0,
    ) -> None:
        """
        :param root: The parent of the dataset root directory.
            The data will be stored in the 'MACS' subdirectory.
        :param download: Download the dataset if download=True and if the dataset is not already downloaded.
            defaults to False.
        :param transforms: The transform to apply to values.
            Keys can be 'audio', 'captions' or 'tags'.
            defaults to None.
        :param unfold: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param item_type: The type of the value returned by __getitem__.
            Can be 'tuple', 'dict' 'dataclass', or 'MACSItem'. Case insensitive.
            defaults to 'tuple'.
        :param verbose: Verbose level to use. Can be 0 or 1.
            defaults to 0.
        """
        if isinstance(item_type, type):
            item_type = item_type.__name__
        item_type = item_type.lower()

        if item_type not in self.ITEM_TYPES:
            raise ValueError(
                f"Invalid argument {item_type=} for {self.__class__.__name__}. (expected one of {self.ITEM_TYPES})"
            )

        if transforms is None:
            transforms = {}
        transforms = {
            name: transform
            for name, transform in transforms.items()
            if transform is not None
        }

        super().__init__()
        self._root = root
        self._download = download
        self._transforms = transforms
        self._unfold = unfold
        self._item_type = item_type
        self._multihot_tags = multihot_tags
        self._verbose = verbose

        self._all_infos = []
        self._tag_to_idx = {}
        self._idx_to_tag = {}

        if self._download:
            self._prepare_data()
        self._load_data()

    def __len__(self) -> int:
        return len(self._all_infos)

    def __getitem__(self, index: int) -> Union[Tuple, Dict, MACSItem]:
        kwargs = {field.name: self.get(field.name, index) for field in fields(MACSItem)}
        item = MACSItem(**kwargs)

        if self._item_type == "tuple":
            return astuple(item)
        elif self._item_type == "dict":
            return asdict(item)
        elif self._item_type in ("dataclass", MACSItem.__name__.lower()):
            return item
        else:
            raise ValueError(
                f"Invalid item_type={self._item_type} for {self.__class__.__name__}. (expected one of {self.ITEM_TYPES})"
            )

    def get_raw(self, name: str, index: int) -> Any:
        """Read the processed raw data. (without transform)

        :param name: The name of the value.
            Can be 'audio', 'fpath', 'fname', 'captions', 'tags', 'annotators_ids'.
        :param index: The index of the value in range [0, len(dataset)[.
        """
        if name == "audio":
            fpath = self.get("fpath", index)
            value, sr = torchaudio.load(fpath)  # type: ignore

            if value.nelement() == 0:
                raise RuntimeError(f'Invalid audio number of elements in "{fpath}".')
            if sr != self.SAMPLE_RATE:
                raise RuntimeError(f'Invalid sample rate in "{fpath}".')

        elif name == "fpath":
            fname = self.get("fname", index)
            value = osp.join(self._dpath_audio, fname)

        elif name == "dataset":
            value = "macs"

        elif name == "index":
            value = index

        elif (
            0 <= index < len(self._all_infos) and name in self._all_infos[index].keys()
        ):
            value = self._all_infos[index][name]

        else:
            raise ValueError(f"Invalid value {name=} at {index=}.")
        return value

    def get(self, name: str, index: int) -> Any:
        """Read the processed raw data. (with transform)

        :param name: The name of the value.
                Can be 'audio', 'fpath', 'fname', 'captions', 'tags', 'annotators_ids'.
        :param index: The index of the value in range [0, len(dataset)[.
        """
        value = self.get_raw(name, index)
        transform = self._transforms.get(name, None)
        if transform is not None:
            value = transform(value)
        return value

    def tag_to_idx(self, tag: str) -> int:
        return self._tag_to_idx[tag]

    def idx_to_tag(self, idx: int) -> str:
        return self._idx_to_tag[idx]

    def _prepare_data(self) -> None:
        if not osp.isdir(self._root):
            raise RuntimeError(f'Cannot find root directory "{self._root}".')

        os.makedirs(self._dpath_audio, exist_ok=True)
        os.makedirs(self._dpath_captions, exist_ok=True)

        fname = MACS_FILES["captions"]["fname"]
        fpath = osp.join(self._dpath_captions, fname)
        if not osp.isfile(fpath):
            if self._verbose >= 1:
                logging.info(f'Downloading captions file "{fname}"...')
            url = MACS_FILES["captions"]["url"]
            hash_ = MACS_FILES["captions"]["hash"]
            download_url(
                url,
                self._dpath_captions,
                fname,
                hash_,
                "md5",
                progress_bar=self._verbose >= 1,
            )

        captions_fpath = osp.join(self._dpath_captions, MACS_FILES["captions"]["fname"])
        with open(captions_fpath, "r") as file:
            captions_data = yaml.safe_load(file)
            captions_data = captions_data["files"]

        for infos in TAU_URBAN_ACOUSTIC_DEV_FILES.values():
            fname = infos["fname"]
            fpath = osp.join(self._dpath_audio, fname)

            if not osp.isfile(fpath):
                if self._verbose >= 1:
                    logging.info(f'Downloading audio zip file "{fname}"...')
                url = infos["url"]
                hash_ = infos["hash"]

                download_url(
                    url,
                    self._dpath_audio,
                    fname,
                    hash_,
                    "md5",
                    progress_bar=self._verbose >= 1,
                )

        for infos in TAU_URBAN_ACOUSTIC_DEV_FILES.values():
            fname = infos["fname"]
            fpath = osp.join(self._dpath_audio, fname)

            macs_fnames = [data["filename"] for data in captions_data]
            with zipfile.ZipFile(fpath, "r") as file:
                fmembers_to_extract = [
                    fpath
                    for fpath in file.namelist()
                    if osp.basename(fpath) in macs_fnames
                ]
                fmembers_to_extract = [
                    member
                    for member in fmembers_to_extract
                    if not osp.isfile(osp.join(self._dpath_audio, osp.basename(member)))
                ]
                if self._verbose >= 1 and len(fmembers_to_extract) > 0:
                    logging.info(
                        f"Extracting {len(fmembers_to_extract)}/{len(file.namelist())} audio files from ZIP file..."
                    )

                file.extractall(self._dpath_audio, fmembers_to_extract)
                for fname in fmembers_to_extract:
                    full_fpath = osp.join(self._dpath_audio, fname)
                    tgt_fpath = osp.join(self._dpath_audio, osp.basename(fname))
                    shutil.move(full_fpath, tgt_fpath)

        audio_fnames = [
            name for name in os.listdir(self._dpath_audio) if name.endswith(".wav")
        ]
        audio_fpaths = [osp.join(self._dpath_audio, name) for name in audio_fnames]
        if self._verbose >= 1:
            logging.info(
                f"{len(audio_fpaths)} audio files has been prepared for MACS dataset."
            )

    def _load_data(self) -> None:
        captions_fpath = osp.join(self._dpath_captions, MACS_FILES["captions"]["fname"])
        if self._verbose >= 1:
            logging.debug(f'Loading captions file "{captions_fpath}"...')

        with open(captions_fpath, "r") as file:
            data = yaml.safe_load(file)
            self._all_infos = [
                {
                    "fname": item["filename"],
                    "captions": [
                        subitem["sentence"] for subitem in item["annotations"]
                    ],
                    "tags": [subitem["tags"] for subitem in item["annotations"]],
                    "annotators_ids": [
                        subitem["annotator_id"] for subitem in item["annotations"]
                    ],
                }
                for item in data["files"]
            ]

        all_tags = [item["tags"] for item in self._all_infos]
        all_tags_flat = (
            tag
            for tags in all_tags
            for tags_by_annotator in tags
            for tag in tags_by_annotator
        )
        all_tags_uniq = dict.fromkeys(all_tags_flat)
        self._tag_to_idx = {tag: i for i, tag in enumerate(all_tags_uniq)}
        self._idx_to_tag = {i: tag for i, tag in enumerate(all_tags_uniq)}

        if self._multihot_tags:
            n_items = len(all_tags)
            max_annotators = max(map(len, all_tags))
            max_tag_by_annotator = max(
                map(
                    len,
                    (
                        tags_by_annotator
                        for tags in all_tags
                        for tags_by_annotator in tags
                    ),
                )
            )
            n_tags_uniq = len(all_tags_uniq)

            multihots_tags_shape = (
                n_items,
                max_annotators,
                max_tag_by_annotator,
                n_tags_uniq,
            )
            multihots_tags = torch.zeros(multihots_tags_shape, dtype=torch.bool)

            for i, tags in enumerate(tqdm.tqdm(all_tags, disable=self._verbose == 0)):
                for j, tags_by_annotator in enumerate(tags):
                    for k, tag in enumerate(tags_by_annotator):
                        idx = self._tag_to_idx[tag]
                        multihots_tags[i, j, k, idx] = True

            for i, multihots in enumerate(multihots_tags):
                self._all_infos[i]["tags"] = multihots

        if self._unfold and self.MIN_CAPTIONS_PER_AUDIO["full"] > 1:
            data_info_unfolded = []
            for infos in self._all_infos:
                captions = infos["captions"]
                for caption in captions:
                    new_infos = dict(infos)
                    new_infos["captions"] = (caption,)
                    data_info_unfolded.append(new_infos)

            self._all_infos = data_info_unfolded

        if self._verbose >= 1:
            logging.info(
                f"{self.__class__.__name__} has been loaded. (len={len(self)})"
            )

    @property
    def _dpath_audio(self) -> str:
        return osp.join(self._root, "MACS", "audio")

    @property
    def _dpath_captions(self) -> str:
        return osp.join(self._root, "MACS", "captions")

    def __repr__(self) -> str:
        return "MACS()"


MACS_FILES = {
    "licence": {
        "fname": "LICENSE.txt",
        "url": "https://zenodo.org/record/5114771/files/LICENSE.txt?download=1",
        "hash": "d3086f4517cccc32c1bb3a081b07cfa1",
    },
    "captions": {
        "fname": "MACS.yaml",
        "url": "https://zenodo.org/record/5114771/files/MACS.yaml?download=1",
        "hash": "23fcb2ebd0b109094034ef9e87972256",
    },
    "annotators_competences": {
        "fname": "MACS_competence.csv",
        "url": "https://zenodo.org/record/5114771/files/MACS_competence.csv?download=1",
        "hash": "4dfe9f951f0af9f29cb7952ec030370a",
    },
}

TAU_URBAN_ACOUSTIC_DEV_FILES = {
    "audio.1": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.1.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.1.zip?download=1",
        "hash": "aca4ebfd9ed03d5f747d6ba8c24bc728",
    },
    "audio.10": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.10.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.10.zip?download=1",
        "hash": "0ffbf60006da520cc761fb74c878b98b",
    },
    "audio.11": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.11.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.11.zip?download=1",
        "hash": "599055d93b4c11057c29be2df54538d4",
    },
    "audio.12": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.12.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.12.zip?download=1",
        "hash": "98b8d162ff3665695c4c910e6c372cc8",
    },
    "audio.13": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.13.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.13.zip?download=1",
        "hash": "a356c08b1a5a21d433eba37ef87587f4",
    },
    "audio.14": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.14.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.14.zip?download=1",
        "hash": "f8969771e7faf7dd471d1cf78b0cf011",
    },
    "audio.15": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.15.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.15.zip?download=1",
        "hash": "4758c4b0fb7484faa632266e78850820",
    },
    "audio.16": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.16.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.16.zip?download=1",
        "hash": "a18acad9ede8ea76574216feb887f0bc",
    },
    "audio.17": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.17.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.17.zip?download=1",
        "hash": "1af7703484632f340da5c33662dc9632",
    },
    "audio.18": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.18.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.18.zip?download=1",
        "hash": "b67402bf3e08f4da394a7c18756c0fd2",
    },
    "audio.19": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.19.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.19.zip?download=1",
        "hash": "035db315f19106eb848b6f9b32bcc47c",
    },
    "audio.2": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.2.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.2.zip?download=1",
        "hash": "c4f170408ce77c8c70c532bf268d7be0",
    },
    "audio.20": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.20.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.20.zip?download=1",
        "hash": "9cb28c74911bf8a3eadcf53f50a5b5d6",
    },
    "audio.21": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.21.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.21.zip?download=1",
        "hash": "0e44ed85c88ec036a9725b4dd1dfaea0",
    },
    "audio.3": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.3.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.3.zip?download=1",
        "hash": "c7214a07211f10f3250290d05e72c37e",
    },
    "audio.4": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.4.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.4.zip?download=1",
        "hash": "a6a62110f6699cf4432072acb1dffda6",
    },
    "audio.5": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.5.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.5.zip?download=1",
        "hash": "091a0b6d3c84b8e60e46940aa7d4a8a0",
    },
    "audio.6": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.6.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.6.zip?download=1",
        "hash": "114f4ca13e074391b98a1cfd8140de65",
    },
    "audio.7": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.7.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.7.zip?download=1",
        "hash": "5951dd2968f7a514e2afbe279c4f060d",
    },
    "audio.8": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.8.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.8.zip?download=1",
        "hash": "b0b63dc95b327e1509857c8d8a663cc3",
    },
    "audio.9": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.audio.9.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.9.zip?download=1",
        "hash": "3c32a693a6b111ffb957be3c1dd22e9b",
    },
    "doc": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.doc.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.doc.zip?download=1",
        "hash": "1f6879544e80da70099a191613e7e51f",
    },
    "meta": {
        "fname": "TAU-urban-acoustic-scenes-2019-development.meta.zip",
        "url": "https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.meta.zip?download=1",
        "hash": "09782f2097e4735687af73c44919329c",
    },
}
