#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
import os.path as osp
import subprocess
import sys
import time

from dataclasses import dataclass, field, fields
from functools import cached_property
from subprocess import CalledProcessError
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
import torchaudio
import tqdm

from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data.dataset import Dataset


logger = logging.getLogger(__name__)


@dataclass
class AudioCapsItem:
    """Dataclass representing a single AudioCAps item."""

    # Common attributes
    audio: Tensor = torch.empty((0,))
    captions: List[str] = field(default_factory=list)
    dataset: str = "audiocaps"
    fname: str = "unknown"
    index: int = -1
    subset: str = "unknown"
    sr: int = -1
    # AudioCaps-specific attributes
    audiocaps_ids: List[int] = field(default_factory=list)
    start_time: int = -1
    tags: List[int] = field(default_factory=list)
    youtube_id: str = "unknown"


class AudioCaps(Dataset):
    r"""Unofficial AudioCaps pytorch dataset.

    Subsets available are 'train', 'val' and 'test'.

    Audio is a waveform tensor of shape (1, n_times) of 10 seconds max, sampled at 16 KHz.
    Target is a list of strings containing the captions.
    The 'train' subset has only 1 caption per sample and 'val' and 'test' have 5 captions.

    Download requires 'youtube-dl' and 'ffmpeg' commands.
    You can change the default path with :attr:`~AudioCaps.YOUTUBE_DL_PATH` or :attr:`~AudioCaps.FFMPEG_PATH` global variables.

    AudioCaps paper : https://www.aclweb.org/anthology/N19-1011.pdf

    .. code-block:: text
        :caption:  Dataset folder tree

        {root}
        └── AUDIOCAPS_32000Hz
            ├── train.csv
            ├── val.csv
            ├── test.csv
            └── audio
                ├── train
                │    └── (46231/49838 flac files, ~42G for 32KHz)
                ├── val
                │    └── (465/495 flac files, ~425M for 32KHz)
                └── test
                    └── (913/975 flac files, ~832M for 32KHz)

    """

    # Global
    AUDIO_FILE_EXTENSION = "flac"
    AUDIO_N_CHANNELS = 1
    AUDIO_MAX_SEC = 10.00096876  # in seconds
    AUDIO_MIN_SEC = 0.6501874  # in seconds
    CAPTION_MIN_LENGTH = 2
    CAPTION_MAX_LENGTH = 52
    CAPTIONS_PER_AUDIO = {"train": 1, "val": 5, "test": 5}
    DNAME_LOG = "logs"
    FFMPEG_PATH: str = "ffmpeg"
    FORCE_PREPARE_DATA: bool = False
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
        item_transform: Optional[Callable] = None,
        unfold: bool = False,
        verbose: int = 0,
        add_removed_audio: bool = False,
        load_tags: bool = False,
    ) -> None:
        """
        :param root: Dataset root directory.
            The data will be stored in the 'AUDIOCAPS_{SAMPLE_RATE}' subdirectory.
            defaults to ".".
        :param subset: The subset of Clotho to use. Can be one of :attr:`~Clotho.SUBSETS`.
            defaults to "train".
        :param download: Download the dataset if download=True and if the dataset is not already downloaded.
            defaults to False.
        :param item_transform: The transform to apply to the global dict item. This transform is applied AFTER each field transform.
            defaults to None.
        :param unfold: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param verbose: Verbose level.
            defaults to 0.
        :param add_removed_audio: If True, the dataset will return an empty Tensor when the audio has been removed from Youtube (or not present on disk).
            defaults to False.
        :param load_tags: If True, load the tags from AudioSet dataset.
            Note: tags needs to be downloaded with download=True & load_tags=True before being used.
            defaults to False.
        """
        if subset not in self.SUBSETS:
            raise ValueError(
                f"Invalid argument {subset=} for AudioCaps. (expected one of {self.SUBSETS})"
            )

        super().__init__()
        self.__root = root
        self.__subset = subset
        self.__download = download
        self.__item_transform = item_transform
        self.__unfold = unfold
        self.__verbose = verbose
        self.__add_removed_audio = add_removed_audio
        self.__load_tags = load_tags

        self.__all_items = {}
        self.__is_loaded = False
        self.__index_to_tagname = []

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
            if not self.__all_items["is_on_disk"][index]:
                return torch.empty((0,))
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
            if not self.__all_items["is_on_disk"][index]:
                return None
            audio_metadata = torchaudio.info(fpath)  # type: ignore
            return audio_metadata

        elif key == "dataset":
            return "audiocaps"

        elif key == "fpath":
            fname = self.get_field("fname", index)
            fpath = osp.join(self._dpath_audio_subset, fname)
            return fpath

        elif key == "index":
            return index

        elif key == "num_channels":
            audio_metadata = self.get_field("audio_metadata", index)
            if audio_metadata is None:
                return -1
            return audio_metadata.num_channels

        elif key == "num_frames":
            audio_metadata = self.get_field("audio_metadata", index)
            if audio_metadata is None:
                return -1
            return audio_metadata.num_frames

        elif key == "sr":
            audio_metadata = self.get_field("audio_metadata", index)
            if audio_metadata is None:
                return -1
            return audio_metadata.sample_rate

        elif key == "subset":
            return self.__subset

        else:
            keys = [field.name for field in fields(AudioCapsItem)]
            raise ValueError(
                f"Invalid argument {key=} at {index=}. (expected one of {tuple(keys)})"
            )

    def get_index_to_tagname(self) -> List[str]:
        return self.__index_to_tagname

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

    @cached_property
    def _dpath_audio_subset(self) -> str:
        return osp.join(
            self._dpath_data,
            "audio",
            self.__subset,
        )

    @cached_property
    def _dpath_data(self) -> str:
        return osp.join(self.__root, f"AUDIOCAPS_{AudioCaps.SAMPLE_RATE}Hz")

    def _is_loaded(self) -> bool:
        return self.__is_loaded

    def _is_prepared(self) -> bool:
        links = AUDIOCAPS_LINKS[self.__subset]
        captions_fname = links["captions"]["fname"]
        captions_fpath = osp.join(self._dpath_data, captions_fname)
        return osp.isdir(self._dpath_audio_subset) and osp.isfile(captions_fpath)

    def _load_data(self) -> None:
        if not self._is_prepared():
            raise RuntimeError(
                f"Cannot load data: audiocaps_{self.__subset} is not prepared in data root={self.__root}. Please use download=True in dataset constructor."
            )

        links = AUDIOCAPS_LINKS[self.__subset]

        captions_fname = links["captions"]["fname"]
        captions_fpath = osp.join(self._dpath_data, captions_fname)
        with open(captions_fpath, "r") as file:
            reader = csv.DictReader(file)
            captions_data = list(reader)

        if self.__load_tags:
            class_labels_indices_fpath = osp.join(
                self._dpath_data, AUDIOSET_LINKS["class_labels_indices"]["fname"]
            )
            unbal_tags_fpath = osp.join(
                self._dpath_data, AUDIOSET_LINKS["unbalanced"]["fname"]
            )

            if not all(map(osp.isfile, (class_labels_indices_fpath, unbal_tags_fpath))):
                raise FileNotFoundError(
                    f"Cannot load tags without tags files '{osp.basename(class_labels_indices_fpath)}' and '{osp.basename(unbal_tags_fpath)}'."
                    f"Please use download=True and load_tags=True in dataset constructor."
                )

            with open(class_labels_indices_fpath, "r") as file:
                reader = csv.DictReader(file)
                audioset_classes_data = list(reader)

            with open(unbal_tags_fpath, "r") as file:
                fieldnames = ("YTID", "start_seconds", "end_seconds", "positive_labels")
                reader = csv.DictReader(
                    file, fieldnames, skipinitialspace=True, strict=True
                )
                # Skip the comments
                for _ in range(3):
                    next(reader)
                unbal_tags_data = list(reader)
        else:
            audioset_classes_data = []
            unbal_tags_data = []

        # Build global mappings
        audio_fnames_on_disk = dict.fromkeys(os.listdir(self._dpath_audio_subset))
        if not self.__add_removed_audio:
            fnames_lst = list(audio_fnames_on_disk)
        else:
            fnames_lst = list(
                dict.fromkeys(
                    f"{line['youtube_id']}_{line['start_time']}.{self.AUDIO_FILE_EXTENSION}"
                    for line in captions_data
                )
            )
        is_on_disk_lst = [fname in audio_fnames_on_disk for fname in fnames_lst]

        dataset_size = len(fnames_lst)
        fname_to_idx = {fname: i for i, fname in enumerate(fnames_lst)}

        mid_to_tag_name = {}
        tag_name_to_index = {}

        for line in audioset_classes_data:
            # keys: index, mid, display_name
            mid_to_tag_name[line["mid"]] = line["display_name"]
            tag_name_to_index[line["display_name"]] = int(line["index"])

        classes_indexes = list(tag_name_to_index.values())
        assert classes_indexes == list(range(classes_indexes[-1] + 1))
        self.__index_to_tagname = list(tag_name_to_index.keys())

        # Process each field into a single structure
        all_caps_dic: Dict[str, List[Any]] = {
            key: [None for _ in range(dataset_size)]
            for key in ("audiocaps_ids", "youtube_id", "start_time", "captions")
        }
        for line in tqdm.tqdm(
            captions_data,
            disable=self.__verbose <= 0,
            desc=f"Loading AudioCaps ({self.__subset}) captions...",
        ):
            # audiocap_id, youtube_id, start_time, caption
            audiocap_id = line["audiocap_id"]
            youtube_id = line["youtube_id"]
            start_time = line["start_time"]
            caption = line["caption"]

            fname = f"{youtube_id}_{start_time}.{self.AUDIO_FILE_EXTENSION}"
            if fname in fname_to_idx:
                idx = fname_to_idx[fname]

                if all_caps_dic["start_time"][idx] is None:
                    all_caps_dic["start_time"][idx] = start_time
                    all_caps_dic["youtube_id"][idx] = youtube_id
                    all_caps_dic["audiocaps_ids"][idx] = [audiocap_id]
                    all_caps_dic["captions"][idx] = [caption]
                else:
                    assert all_caps_dic["start_time"][idx] == start_time
                    assert all_caps_dic["youtube_id"][idx] == youtube_id

                    all_caps_dic["audiocaps_ids"][idx].append(audiocap_id)
                    all_caps_dic["captions"][idx].append(caption)

        # Load tags from audioset data
        all_tags_lst = [[] for _ in range(dataset_size)]

        for line in tqdm.tqdm(
            unbal_tags_data,
            disable=self.__verbose <= 0,
            desc="Loading AudioSet tags...",
        ):
            # keys: YTID, start_seconds, end_seconds, positive_labels
            youtube_id = line["YTID"]
            # Note : In audioset, start_time is a string repr of a float value, audiocaps it is a string repr of an integer
            start_time = int(float(line["start_seconds"]))
            fname = f"{youtube_id}_{start_time}.{self.AUDIO_FILE_EXTENSION}"
            if fname in fname_to_idx:
                tags_mid = line["positive_labels"]
                tags_mid = tags_mid.split(",")
                tags_names = [mid_to_tag_name[tag] for tag in tags_mid]
                tags_indexes = [tag_name_to_index[tag] for tag in tags_names]

                idx = fname_to_idx[fname]
                all_tags_lst[idx] = tags_indexes

        all_items = {
            "fname": fnames_lst,
            "tags": all_tags_lst,
            "is_on_disk": is_on_disk_lst,
        } | all_caps_dic

        # Convert audiocaps_ids and start_time to ints
        all_items["audiocaps_ids"] = [
            list(map(int, item)) for item in all_items["audiocaps_ids"]
        ]
        all_items["start_time"] = list(map(int, all_items["start_time"]))

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

        links = AUDIOCAPS_LINKS[self.__subset]
        captions_fname = links["captions"]["fname"]
        captions_fpath = osp.join(self._dpath_data, captions_fname)

        os.makedirs(self._dpath_audio_subset, exist_ok=True)

        if not osp.isfile(captions_fpath):
            url = links["captions"]["url"]
            download_url_to_file(url, captions_fpath, progress=self.__verbose >= 1)

        start = time.perf_counter()
        with open(captions_fpath, "r") as file:
            n_samples = len(file.readlines())

        if self.__verbose >= 1:
            log_dpath = osp.join(self._dpath_data, self.DNAME_LOG)
            if not osp.isdir(log_dpath):
                os.makedirs(log_dpath)

            if self.REDIRECT_LOG:
                logging.basicConfig(
                    filename=osp.join(log_dpath, f"preparation_{self.__subset}.txt"),
                    filemode="w",
                    level=logging.INFO,
                    force=True,
                )
            logger.info(f"Start downloading files for {self.__subset} AudioCaps split.")

        with open(captions_fpath, "r") as file:
            # Download audio files
            reader = csv.DictReader(file)
            if self.__verbose >= 1:
                reader = tqdm.tqdm(reader, total=n_samples)

            n_download_ok, n_download_err, n_already_ok, n_already_err = 0, 0, 0, 0
            for line in reader:
                # Keys: audiocap_id, youtube_id, start_time, caption
                audiocap_id, youtube_id, start_time = [
                    line[key] for key in ("audiocap_id", "youtube_id", "start_time")
                ]
                fpath = osp.join(
                    self._dpath_audio_subset,
                    f"{youtube_id}_{start_time}.{self.AUDIO_FILE_EXTENSION}",
                )
                if not start_time.isdigit():
                    raise RuntimeError(
                        f'Start time "{start_time}" is not an integer (audiocap_id={audiocap_id}, youtube_id={youtube_id}).'
                    )
                start_time = int(start_time)

                if not osp.isfile(fpath):
                    success = _download_and_extract_from_youtube(
                        youtube_id,
                        fpath,
                        start_time,
                        duration=self.AUDIO_MAX_SEC,
                        sr=self.SAMPLE_RATE,
                        youtube_dl_path=self.YOUTUBE_DL_PATH,
                        ffmpeg_path=self.FFMPEG_PATH,
                        n_channels=self.AUDIO_N_CHANNELS,
                    )
                    if success:
                        valid_file = self._check_file(fpath)
                        if valid_file:
                            if self.__verbose >= 2:
                                logger.debug(
                                    f'[{audiocap_id:6s}] File "{youtube_id}" has been downloaded and verified.'
                                )
                            n_download_ok += 1
                        else:
                            if self.__verbose >= 1:
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
                        if self.__verbose >= 2:
                            logger.debug(
                                f'[{audiocap_id:6s}] File "{youtube_id}" is already downloaded and has been verified.'
                            )
                        n_already_ok += 1
                    else:
                        if self.__verbose >= 1:
                            logger.warning(
                                f'[{audiocap_id:6s}] File "{youtube_id}" is already downloaded but it is not valid and will be removed.'
                            )
                        os.remove(fpath)
                        n_already_err += 1
                else:
                    if self.__verbose >= 2:
                        logger.debug(
                            f'[{audiocap_id:6s}] File "{youtube_id}" is already downloaded but it is not verified due to {self.VERIFY_FILES=}.'
                        )
                    n_already_ok += 1

        if self.__load_tags:
            for key in ("class_labels_indices", "unbalanced"):
                infos = AUDIOSET_LINKS[key]
                url = infos["url"]
                fname = infos["fname"]
                fpath = osp.join(self._dpath_data, fname)
                if not osp.isfile(fpath):
                    if self.__verbose >= 1:
                        logger.info(f"Downloading file '{fname}'...")
                    download_url_to_file(url, fpath, progress=self.__verbose >= 1)

        if self.__verbose >= 1:
            duration = int(time.perf_counter() - start)
            logger.info(
                f'Download and preparation of AudioCaps for subset "{self.__subset}" done in {duration}s. '
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

    def __getitem__(self, index: Union[int, slice]) -> Dict[str, Any]:
        keys = [field.name for field in fields(AudioCapsItem)]
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
        return f"AudioCaps(subset={self.__subset})"


def _download_and_extract_from_youtube(
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
        "captions": {
            "url": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv",
            "fname": "train.csv",
        },
    },
    "val": {
        "captions": {
            "url": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/val.csv",
            "fname": "val.csv",
        },
    },
    "test": {
        "captions": {
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
