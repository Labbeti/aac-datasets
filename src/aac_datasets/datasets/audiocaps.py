#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
import os.path as osp
import subprocess
import sys
import time
import tqdm

from dataclasses import asdict, astuple, dataclass, field, fields
from subprocess import CalledProcessError
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchaudio

from torch import nn, Tensor
from torch.utils.data.dataset import Dataset
from torchaudio.backend.common import AudioMetaData
from torchaudio.datasets.utils import download_url


logger = logging.getLogger(__name__)


@dataclass
class AudioCapsItem:
    audio: Tensor = torch.empty((0,), dtype=torch.float)
    captions: List[str] = field(default_factory=list)
    tags: Optional[List[str]] = None
    fname: Optional[str] = None
    index: int = -1
    dataset: str = "audiocaps"
    subset: Optional[str] = None
    sr: int = -1


class AudioCaps(Dataset):
    """
    Unofficial AudioCaps pytorch dataset.
    Subsets available are 'train', 'val' and 'test'.

    Audio is a waveform tensor of shape (1, n_times) of 10 seconds max, sampled at 16 KHz.
    Target is a list of strings containing the captions.
    The 'train' subset has only 1 caption per sample and 'val' and 'test' have 5 captions.

    Download requires 'youtube-dl' and 'ffmpeg' commands.
    You can change the default path with 'AudioCaps.YOUTUBE_DL_PATH' or 'AudioCaps.FFMPEG_PATH' global variables.

    AudioCaps paper : https://www.aclweb.org/anthology/N19-1011.pdf

    Dataset folder tree :

    ```
    root/
    └── AUDIOCAPS_32000Hz
        ├── train.csv
        ├── val.csv
        ├── test.csv
        └── audio
            ├── train
            │    └── (46231/49838 files, ~42G for 32KHz)
            ├── val
            │    └── (465/495 files, ~425M for 32KHz)
            └── test
                └── (913/975 files, ~832M for 32KHz)
    ```
    """

    AUDIO_FILE_EXTENSION = "flac"
    AUDIO_N_CHANNELS = 1
    AUDIO_MAX_LENGTH = 10.00096876  # in seconds
    AUDIO_MIN_LENGTH = 0.6501874  # in seconds
    CAPTION_MIN_LENGTH = 1
    CAPTION_MAX_LENGTH = 52
    CAPTIONS_PER_AUDIO = {"train": 1, "val": 5, "test": 5}
    DNAME_AUDIO = "audio"
    DNAME_LOG = "logs"
    FFMPEG_PATH: str = "ffmpeg"
    FORCE_PREPARE_DATA: bool = False
    ITEM_TYPES = ("tuple", "dict", "dataclass")
    REDIRECT_LOG = False
    SAMPLE_RATE = 32000
    SUBSETS = ("train", "val", "test")
    VERIFY_FILES = False
    YOUTUBE_DL_PATH: str = "youtube-dl"

    def __init__(
        self,
        root: str = ".",
        subset: str = "train",
        download: bool = False,
        transforms: Optional[Dict[str, Optional[nn.Module]]] = None,
        accept_invalid_fpath: bool = False,
        unfold: bool = False,
        item_type: str = "tuple",
        load_tags: bool = True,
        verbose: int = 0,
    ) -> None:
        """
        :param root: Dataset root directory.
            The data will be stored in the 'AUDIOCAPS_{SAMPLE_RATE}' subdirectory.
            defaults to ".".
        :param subset: The AudioCaps subset.
            Can be 'train', 'val' or 'test'.
        :param download: If True, starts to download the dataset from Youtube. Can requires a lot of time depending of your machine.
            defaults to False.
        :param transforms: The dictionary of transforms to apply.
            All keys must be a valid item attribute, i.e. one of "audio", "captions", "tags", "fname", "index", "dataset", "subset", "fpath", "youtube_id", "start_time", "sr".
            default to None.
        :param accept_invalid_fpath: If True, the dataset will return an empty Tensor when the audio is not present on the disk.
            defaults to False.
        :param unfold: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param item_type: The type of the value returned by __getitem__.
            Can be 'tuple', 'dict' or 'dataclass'.
            defaults to 'tuple'.
        :param load_tags: If True, load the tags from AudioSet dataset.
            Note: tags needs to be downloaded with download=True & load_tags=True before being used.
            defaults to False.
        :param verbose: Verbose level.
            defaults to 0.
        """
        if subset not in self.SUBSETS:
            raise ValueError(
                f"Invalid argument {subset=} for AudioCaps. (expected one of {self.SUBSETS})"
            )

        if item_type not in self.ITEM_TYPES:
            raise ValueError(
                f"Invalid argument {item_type=} for AudioCaps. (expected one of {self.ITEM_TYPES})"
            )

        if transforms is None:
            transforms = {}

        accepted_keys = [field_.name for field_ in fields(AudioCapsItem)]
        for key in transforms.keys():
            if key not in accepted_keys:
                raise ValueError(
                    f"Invalid argument transforms {key=}. (expected one of {accepted_keys})"
                )

        super().__init__()
        self._root = root
        self._subset = subset
        self._download = download
        self._transforms = transforms
        self._accept_invalid_fpath = accept_invalid_fpath
        self._unfold = unfold
        self._item_type = item_type
        self._load_tags = load_tags
        self._verbose = verbose

        self._all_infos = []
        self._data_dname = f"AUDIOCAPS_{self.SAMPLE_RATE}Hz"

        if self._download:
            self._prepare_data()
        self._load_data()

    def __len__(self) -> int:
        return len(self._all_infos)

    def __getitem__(self, index: int) -> Union[Tuple, Dict, AudioCapsItem]:
        """
        Get the audio data as 1D tensor and the matching captions.

        :param index: The index of the value in range [0, len(dataset)[.
        :return: Item containing the following data: "audio", "captions", "tags", "fname", "index", "dataset", "subset".
            The item type is determined by the 'item_type' attribute of the dataset, and can be a tuple, dict or AudioCapsItem.
        """
        kwargs = {
            field.name: self.get(field.name, index) for field in fields(AudioCapsItem)
        }
        item = AudioCapsItem(**kwargs)

        if self._item_type == "tuple":
            return astuple(item)
        elif self._item_type == "dict":
            return asdict(item)
        elif self._item_type == "dataclass":
            return item
        else:
            raise ValueError(
                f"Invalid item_type={self._item_type} for AudioCaps_{self._subset}. (expected one of {self.ITEM_TYPES})"
            )

    def get_raw(self, name: str, index: int) -> Any:
        """Read a raw data. (without transform applied)

        :param name: The name of the value.
            Can be "audio", "captions", "tags", "fname", "index", "dataset", "subset", "fpath", "youtube_id", "start_time", "sr".
        :param index: The index of the value in range [0, len(dataset)[.
        :return: The value of the data 'name' at specified index.
        """
        if not (0 <= index < len(self)):
            raise IndexError(
                f"Invalid argument {index=} for {self} (expected in range [0, {len(self)}-1])"
            )

        if name == "audio":
            fpath = self.get("fpath", index)

            if fpath is None and self._accept_invalid_fpath:
                return torch.empty((0,), dtype=torch.float)

            value, sr = torchaudio.load(fpath)  # type: ignore
            if sr != self.SAMPLE_RATE:
                raise RuntimeError(
                    f"Invalid sample rate {sr}Hz of audio file {fpath} with AudioCaps {self.SAMPLE_RATE}Hz."
                )

        elif name == "fpath":
            audio_subset_dpath = osp.join(
                self._root, self._data_dname, self.DNAME_AUDIO, self._subset
            )
            fname = self.get("fname", index)
            if fname is None:
                value = None
            else:
                value = osp.join(audio_subset_dpath, fname)

        elif name == "dataset":
            value = "audiocaps"

        elif name == "subset":
            value = self._subset

        elif name == "index":
            value = index

        elif name == "sr":
            fpath = self.get("fpath", index)

            if fpath is None and self._accept_invalid_fpath:
                return -1

            audio_info: AudioMetaData = torchaudio.info(fpath)  # type: ignore
            return audio_info.sample_rate

        elif name in self._all_infos[index].keys():
            # keys : "youtube_id", "fname", "start_time", "captions", "tags"
            value = self._all_infos[index][name]

        else:
            raise ValueError(f"Invalid value {name=} at {index=}.")
        return value

    def get(self, name: str, index: int) -> Any:
        """Read a processed data. (with transform)

        :param name: The name of the value.
            Can be "audio", "captions", "tags", "fname", "index", "dataset", "subset", "fpath", "youtube_id", "start_time", "sr".
        :param index: The index of the value in range [0, len(dataset)[.
        :return: The value of the data 'name' at specified index.
        """
        value = self.get_raw(name, index)
        transform = self._transforms.get(name, None)
        if transform is not None:
            value = transform(value)
        return value

    def _is_prepared(self) -> bool:
        root_dpath = osp.join(self._root, self._data_dname)
        audio_subset_dpath = osp.join(root_dpath, self.DNAME_AUDIO, self._subset)
        links = AUDIOCAPS_LINKS[self._subset]
        meta_fname = links["meta"]["fname"]
        meta_fpath = osp.join(root_dpath, meta_fname)
        return osp.isdir(audio_subset_dpath) and osp.isfile(meta_fpath)

    def _prepare_data(self) -> None:
        if not osp.isdir(self._root):
            raise RuntimeError(f"Cannot find root directory '{self._root}'.")

        try:
            subprocess.check_call(
                [self.YOUTUBE_DL_PATH, "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (CalledProcessError, PermissionError, FileNotFoundError) as err:
            logger.error(
                f"Cannot use youtube-dl path '{self.YOUTUBE_DL_PATH}'. ({err})"
            )
            raise err

        try:
            subprocess.check_call(
                [self.FFMPEG_PATH, "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (CalledProcessError, PermissionError, FileNotFoundError) as err:
            logger.error(f"Cannot use ffmpeg path '{self.FFMPEG_PATH}'. ({err})")
            raise err

        if self._is_prepared() and not self.FORCE_PREPARE_DATA:
            return None

        links = AUDIOCAPS_LINKS[self._subset]
        root_dpath = osp.join(self._root, self._data_dname)
        audio_subset_dpath = osp.join(root_dpath, self.DNAME_AUDIO, self._subset)
        meta_fname = links["meta"]["fname"]
        meta_fpath = osp.join(root_dpath, meta_fname)

        if not osp.isdir(audio_subset_dpath):
            os.makedirs(audio_subset_dpath)

        if not osp.isfile(meta_fpath):
            url_meta = links["meta"]["url"]
            download_url(url_meta, root_dpath, progress_bar=self._verbose >= 1)

        start = time.perf_counter()
        with open(meta_fpath, "r") as file:
            n_samples = len(file.readlines())

        if self._verbose >= 1:
            log_dpath = osp.join(root_dpath, self.DNAME_LOG)
            if not osp.isdir(log_dpath):
                os.makedirs(log_dpath)

            if self.REDIRECT_LOG:
                logging.basicConfig(
                    filename=osp.join(log_dpath, f"preparation_{self._subset}.txt"),
                    filemode="w",
                    level=logging.INFO,
                    force=True,
                )
            logger.info(f"Start downloading files for {self._subset} AudioCaps split.")

        with open(meta_fpath, "r") as file:
            # Download audio files
            reader = csv.DictReader(file)
            if self._verbose >= 1:
                reader = tqdm.tqdm(reader, total=n_samples)

            n_download_ok, n_download_err, n_already_ok, n_already_err = 0, 0, 0, 0
            for line in reader:
                # Keys: audiocap_id, youtube_id, start_time, caption
                audiocap_id, youtube_id, start_time = [
                    line[key] for key in ("audiocap_id", "youtube_id", "start_time")
                ]
                fpath = osp.join(
                    audio_subset_dpath,
                    f"{youtube_id}_{start_time}.{self.AUDIO_FILE_EXTENSION}",
                )
                if not start_time.isdigit():
                    raise RuntimeError(
                        f'Start time "{start_time}" is not an integer (audiocap_id={audiocap_id}, youtube_id={youtube_id}).'
                    )
                start_time = int(start_time)

                if not osp.isfile(fpath):
                    success = download_and_extract_from_youtube(
                        youtube_id,
                        fpath,
                        start_time,
                        duration=self.AUDIO_MAX_LENGTH,
                        sr=self.SAMPLE_RATE,
                        youtube_dl_path=self.YOUTUBE_DL_PATH,
                        ffmpeg_path=self.FFMPEG_PATH,
                        n_channels=self.AUDIO_N_CHANNELS,
                    )
                    if success:
                        valid_file = self._check_file(fpath)
                        if valid_file:
                            if self._verbose >= 2:
                                logger.debug(
                                    f'[{audiocap_id:6s}] File "{youtube_id}" has been downloaded and verified.'
                                )
                            n_download_ok += 1
                        else:
                            if self._verbose >= 1:
                                logger.warning(
                                    f'[{audiocap_id:6s}] File "{youtube_id}" has been downloaded but it is not valid and it will be removed.'
                                )
                            os.remove(fpath)
                            n_download_err += 1
                    else:
                        logger.error(
                            f'[{audiocap_id:6s}] Cannot extract audio from "{youtube_id}".'
                        )
                        n_download_err += 1

                elif self.VERIFY_FILES:
                    valid_file = self._check_file(fpath)
                    if valid_file:
                        if self._verbose >= 2:
                            logger.debug(
                                f'[{audiocap_id:6s}] File "{youtube_id}" is already downloaded and has been verified.'
                            )
                        n_already_ok += 1
                    else:
                        if self._verbose >= 1:
                            logger.warning(
                                f'[{audiocap_id:6s}] File "{youtube_id}" is already downloaded but it is not valid and will be removed.'
                            )
                        os.remove(fpath)
                        n_already_err += 1
                else:
                    if self._verbose >= 2:
                        logger.debug(
                            f'[{audiocap_id:6s}] File "{youtube_id}" is already downloaded but it is not verified due to {self.VERIFY_FILES=}.'
                        )
                    n_already_ok += 1

        if self._load_tags:
            for key in ("class_labels_indices", "unbalanced"):
                infos = AUDIOSET_LINKS[key]
                url = infos["url"]
                fname = infos["fname"]
                fpath = osp.join(root_dpath, fname)
                if not osp.isfile(fpath):
                    if self._verbose >= 1:
                        logger.info(f"Downloading file '{fname}'...")
                    download_url(url, root_dpath, fname, progress_bar=self._verbose > 0)

        if self._verbose >= 1:
            duration = int(time.perf_counter() - start)
            logger.info(
                f'Download and preparation of AudioCaps for subset "{self._subset}" done in {duration}s. '
            )
            logger.info(f"- {n_download_ok} downloads success,")
            logger.info(f"- {n_download_err} downloads failed,")
            logger.info(f"- {n_already_ok} already downloaded,")
            logger.info(f"- {n_already_err} already downloaded errors,")
            logger.info(f"- {n_samples} total samples.")

            if self.REDIRECT_LOG:
                logging.basicConfig(
                    stream=sys.stdout,
                    level=logging.INFO,
                    force=True,
                )

    def _check_file(self, fpath: str) -> bool:
        try:
            audio, sr = torchaudio.load(fpath)  # type: ignore
        except RuntimeError:
            message = f'Found file "{fpath}" already downloaded but it is invalid (cannot load). It will be removed.'
            logger.error(message)
            return False

        if audio.nelement() == 0:
            message = f'Found file "{fpath}" already downloaded but it is invalid (empty audio). It will be removed.'
            logger.error(message)
            return False

        if sr != self.SAMPLE_RATE:
            message = f'Found file "{fpath}" already downloaded but it is invalid (invalid sr={sr} != {self.SAMPLE_RATE}). It will be removed.'
            logger.error(message)
            return False

        return True

    def _load_data(self) -> None:
        if not self._is_prepared():
            raise RuntimeError(
                f"Cannot load data: audiocaps_{self._subset} is not prepared in data root={self._root}. Please use download=True in dataset constructor."
            )

        links = AUDIOCAPS_LINKS[self._subset]
        root_dpath = osp.join(self._root, self._data_dname)
        audio_subset_dpath = osp.join(root_dpath, self.DNAME_AUDIO, self._subset)
        meta_fname = links["meta"]["fname"]
        meta_fpath = osp.join(root_dpath, meta_fname)

        youtube_id_to_index = {}
        self._all_infos = []

        with open(meta_fpath, "r") as file:
            reader = csv.DictReader(file)

            for line_dict in tqdm.tqdm(
                reader,
                disable=self._verbose == 0,
                desc=f"audiocaps_{self._subset}: Loading captions...",
            ):
                youtube_id = line_dict["youtube_id"]
                start_time = line_dict["start_time"]
                caption = line_dict["caption"]

                fname = f"{youtube_id}_{start_time}.{self.AUDIO_FILE_EXTENSION}"
                fpath = osp.join(audio_subset_dpath, fname)
                isfile = osp.isfile(fpath)

                if isfile or self._accept_invalid_fpath:
                    if not isfile:
                        fname = None

                    if youtube_id in youtube_id_to_index.keys():
                        index = youtube_id_to_index[youtube_id]
                        captions = self._all_infos[index]["captions"]
                        assert fname == self._all_infos[index]["fname"]
                        assert start_time == self._all_infos[index]["start_time"]
                        self._all_infos[index]["captions"].append(caption)

                    else:
                        new_index = len(self._all_infos)
                        youtube_id_to_index[youtube_id] = new_index
                        info = {
                            "youtube_id": youtube_id,
                            "fname": fname,
                            "start_time": start_time,
                            "captions": [caption],
                            "tags": None,
                        }
                        self._all_infos.append(info)

        if self._load_tags:
            class_labels_indices_fpath = osp.join(
                root_dpath, AUDIOSET_LINKS["class_labels_indices"]["fname"]
            )
            unbalanced_tags_fpath = osp.join(
                root_dpath, AUDIOSET_LINKS["unbalanced"]["fname"]
            )

            mid_to_tag_name = {}
            tag_name_to_index = {}

            with open(class_labels_indices_fpath, "r") as file:
                reader = csv.DictReader(file)
                # index,mid,display_name
                for line in reader:
                    mid_to_tag_name[line["mid"]] = line["display_name"]
                    tag_name_to_index[line["display_name"]] = int(line["index"])

            with open(unbalanced_tags_fpath, "r") as file:
                fieldnames = ("YTID", "start_seconds", "end_seconds", "positive_labels")
                reader = csv.DictReader(
                    file, fieldnames, skipinitialspace=True, strict=True
                )
                # Skip the comments
                for _ in range(3):
                    next(reader)

                # YTID, start_seconds, end_seconds, positive_labels
                for line in tqdm.tqdm(
                    reader, disable=self._verbose <= 0, desc="Loading tags..."
                ):
                    youtube_id = line["YTID"]
                    if youtube_id in youtube_id_to_index.keys():
                        index = youtube_id_to_index[youtube_id]

                        # Note : In audiocaps, start_time is a string repr of an integer value
                        start_time_audiocaps = float(
                            self._all_infos[index]["start_time"]
                        )
                        # Note : In audioset, start_time is a string repr of a float value
                        start_time_audioset = float(line["start_seconds"])
                        assert start_time_audiocaps == start_time_audioset

                        tags = line["positive_labels"]
                        tags = tags.split(",")
                        tags = [mid_to_tag_name[tag] for tag in tags]
                        self._all_infos[index]["tags"] = tags

            assert all(
                "tags" in info.keys() and info["tags"] is not None
                for info in self._all_infos
            )

        if self._unfold and self.CAPTIONS_PER_AUDIO[self._subset] > 1:
            data_info_unfolded = []
            for links in self._all_infos:
                captions = links["captions"]
                for caption in captions:
                    new_infos = dict(links)
                    new_infos["captions"] = (caption,)
                    data_info_unfolded.append(new_infos)

            self._all_infos = data_info_unfolded

        self._all_infos = self._all_infos
        if self._verbose >= 1:
            logger.info(
                f"{self.__class__.__name__}/{self._subset} has been loaded. (len={len(self)})"
            )

    def __repr__(self) -> str:
        return f"AudioCaps(subset={self._subset})"


def download_and_extract_from_youtube(
    youtube_id: str,
    fpath_out: str,
    start_time: int,
    duration: float = 10.0,
    sr: int = 16000,
    n_channels: int = 1,
    target_format: str = "flac",
    acodec: str = "flac",
    youtube_dl_path: str = "youtube-dl",
    ffmpeg_path: str = "ffmpeg",
) -> bool:
    # Get audio download link with youtube-dl
    link = f"https://www.youtube.com/watch?v={youtube_id}"
    get_url_command = [
        youtube_dl_path,
        "--youtube-skip-dash-manifest",
        "-g",
        link,
    ]
    try:
        output = subprocess.check_output(get_url_command)
    except (CalledProcessError, PermissionError):
        return False

    output = output.decode()
    lines = output.split("\n")
    if len(lines) < 2:
        return False
    _video_link, audio_link = lines[:2]

    # Download and extract audio from audio_link to fpath_out with ffmpeg
    extract_command = [
        ffmpeg_path,
        # Input
        "-i",
        audio_link,
        # Remove video
        "-vn",
        # Format (flac)
        "-f",
        target_format,
        # Audio codec (flac)
        "-acodec",
        acodec,
        # Get only 10s of the clip after start_time
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        # Resample to 16 KHz
        "-ar",
        str(sr),
        # Compute mean of 2 channels
        "-ac",
        str(n_channels),
        fpath_out,
    ]
    try:
        exitcode = subprocess.check_call(extract_command)
        return exitcode == 0
    except (CalledProcessError, PermissionError):
        return False


AUDIOCAPS_LINKS = {
    "train": {
        "meta": {
            "url": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv",
            "fname": "train.csv",
        },
    },
    "val": {
        "meta": {
            "url": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/val.csv",
            "fname": "val.csv",
        },
    },
    "test": {
        "meta": {
            "url": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv",
            "fname": "test.csv",
        },
    },
}

AUDIOSET_LINKS = {
    "class_labels_indices": {
        "fname": "class_labels_indices.csv",
        "url": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
    },
    "eval": {
        "fname": "eval_segments.csv",
        "url": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
    },
    "balanced": {
        "fname": "balanced_train_segments.csv",
        "url": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
    },
    "unbalanced": {
        "fname": "unbalanced_train_segments.csv",
        "url": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv",
    },
}
