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
from functools import lru_cache
from subprocess import CalledProcessError
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torchaudio
import tqdm

from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data.dataset import Dataset


logger = logging.getLogger(__name__)


@dataclass
class AudioCapsItem:
    """Dataclass representing a single AudioCaps item."""

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


class AudioCaps(Dataset[Dict[str, Any]]):
    r"""Unofficial AudioCaps PyTorch dataset.

    Subsets available are 'train', 'val' and 'test'.

    Audio is a waveform tensor of shape (1, n_times) of 10 seconds max, sampled at 32KHz.
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
    AUDIO_MAX_SEC = 10.00096876  # in seconds
    AUDIO_MIN_SEC = 0.6501874  # in seconds
    AUDIO_N_CHANNELS = 1
    CAPTION_MAX_LENGTH = 52
    CAPTION_MIN_LENGTH = 2
    CAPTIONS_PER_AUDIO = {"train": 1, "val": 5, "test": 5}
    DNAME_LOG = "logs"
    FFMPEG_PATH: str = "ffmpeg"
    FORCE_PREPARE_DATA: bool = False
    N_AUDIOSET_CLASSES: int = 527
    REDIRECT_LOG = False
    SAMPLE_RATE = 32000
    SUBSETS = ("train", "val", "test")
    VERIFY_FILES = False
    YOUTUBE_DL_PATH: str = "youtube-dl"

    # Initialization
    def __init__(
        self,
        root: str = ".",
        subset: str = "train",
        download: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        flat_captions: bool = False,
        verbose: int = 0,
        exclude_removed_audio: bool = True,
        with_tags: bool = False,
    ) -> None:
        """
        :param root: Dataset root directory.
            The data will be stored in the 'AUDIOCAPS_{SAMPLE_RATE}' subdirectory.
            defaults to ".".
        :param subset: The subset of AudioCaps to use. Can be one of :attr:`~AudioCaps.SUBSETS`.
            defaults to "train".
        :param download: Download the dataset if download=True and if the dataset is not already downloaded.
            defaults to False.
        :param transform: The transform to apply to the global dict item. This transform is applied only in getitem method.
            defaults to None.
        :param flat_captions: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param verbose: Verbose level.
            defaults to 0.
        :param exclude_removed_audio: If True, the dataset will return exclude from the dataset the audio not downloaded from youtube (i.e. not present on disk).
            If False, invalid audios will return an empty tensor of shape (0,).
            defaults to True.
        :param with_tags: If True, load the tags from AudioSet dataset.
            Note: tags needs to be downloaded with download=True & with_tags=True before being used.
            defaults to False.
        """
        if subset not in AudioCaps.SUBSETS:
            raise ValueError(
                f"Invalid argument subset={subset} for AudioCaps. (expected one of {AudioCaps.SUBSETS})"
            )

        super().__init__()
        # Attributes
        self._root = root
        self._subset = subset
        self._download = download
        self._transform = transform
        self._flat_captions = flat_captions
        self._verbose = verbose
        self._exclude_removed_audio = exclude_removed_audio
        self._with_tags = with_tags

        # Data to load
        self._all_items: Dict[str, List[Any]] = {}
        self._loaded = False
        self._index_to_tagname: List[str] = []

        if self._download:
            self.__prepare_data()
        self.__load_data()

    # Properties
    @property
    def column_names(self) -> List[str]:
        """The name of each column of the dataset."""
        return [field.name for field in fields(AudioCapsItem)]

    @property
    def index_to_tagname(self) -> List[str]:
        """AudioSet ordered list of tag names. Returns an empty list if `with_tags` is False."""
        return self._index_to_tagname

    @property
    def info(self) -> Dict[str, Any]:
        """Return the global dataset info."""
        return {
            "dataset": "audiocaps",
            "subset": self._subset,
            "with_tags": self._with_tags,
        }

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the AudioCaps dataset."""
        return len(self), len(self.column_names)

    # Public methods
    def at(
        self,
        idx: Union[int, Iterable[int], None, slice] = None,
        column: Union[str, Iterable[str], None] = None,
    ) -> Any:
        """Get a specific data field.

        :param index: The index or slice of the value in range [0, len(dataset)-1].
        :param column: The name(s) of the column. Can be any value of :meth:`~AudioCaps.column_names`.
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
            if not self._all_items["is_on_disk"][idx]:
                return torch.empty((0,))
            audio, sr = torchaudio.load(fpath)  # type: ignore

            # Sanity check
            if audio.nelement() == 0:
                raise RuntimeError(
                    f"Invalid audio number of elements in '{fpath}'. (expected audio.nelements()={audio.nelement()} > 0)"
                )
            if sr != self.SAMPLE_RATE:
                raise RuntimeError(
                    f"Invalid sample rate in '{fpath}'. (expected {self.SAMPLE_RATE} but found sr={sr})"
                )
            return audio

        elif column == "audio_metadata":
            fpath = self.at(idx, "fpath")
            if not self._all_items["is_on_disk"][idx]:
                return None
            audio_metadata = torchaudio.info(fpath)  # type: ignore
            return audio_metadata

        elif column == "dataset":
            return "audiocaps"

        elif column == "fpath":
            fname = self.at(idx, "fname")
            fpath = osp.join(self.__dpath_audio_subset, fname)
            return fpath

        elif column == "index":
            return idx

        elif column == "num_channels":
            audio_metadata = self.at(idx, "audio_metadata")
            if audio_metadata is None:
                return -1
            return audio_metadata.num_channels

        elif column == "num_frames":
            audio_metadata = self.at(idx, "audio_metadata")
            if audio_metadata is None:
                return -1
            return audio_metadata.num_frames

        elif column == "sr":
            audio_metadata = self.at(idx, "audio_metadata")
            if audio_metadata is None:
                return -1
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
        return f"AudioCaps(size={len(self)}, subset={self._subset}, num_columns={len(self.column_names)}, with_tags={self._with_tags})"

    # Public class methods
    @classmethod
    def load_class_labels_indices(cls, root: str) -> List[Dict[str, str]]:
        class_labels_indices_fpath = osp.join(
            root,
            f"AUDIOCAPS_{AudioCaps.SAMPLE_RATE}Hz",
            AUDIOSET_LINKS["class_labels_indices"]["fname"],
        )
        with open(class_labels_indices_fpath, "r") as file:
            reader = csv.DictReader(file)
            audioset_classes_data = list(reader)
        return audioset_classes_data

    # Private methods
    def __check_file(self, fpath: str) -> bool:
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

    @property
    @lru_cache()
    def __dpath_audio_subset(self) -> str:
        return osp.join(
            self.__dpath_data,
            "audio",
            self._subset,
        )

    @property
    @lru_cache()
    def __dpath_data(self) -> str:
        return osp.join(self._root, f"AUDIOCAPS_{AudioCaps.SAMPLE_RATE}Hz")

    def __is_prepared(self) -> bool:
        links = AUDIOCAPS_LINKS[self._subset]
        captions_fname = links["captions"]["fname"]
        captions_fpath = osp.join(self.__dpath_data, captions_fname)
        return osp.isdir(self.__dpath_audio_subset) and osp.isfile(captions_fpath)

    def __load_data(self) -> None:
        if not self.__is_prepared():
            raise RuntimeError(
                f"Cannot load data: audiocaps_{self._subset} is not prepared in data root={self._root}. Please use download=True in dataset constructor."
            )

        links = AUDIOCAPS_LINKS[self._subset]

        captions_fname = links["captions"]["fname"]
        captions_fpath = osp.join(self.__dpath_data, captions_fname)
        with open(captions_fpath, "r") as file:
            reader = csv.DictReader(file)
            captions_data = list(reader)

        if self._with_tags:
            class_labels_indices_fpath = osp.join(
                self.__dpath_data, AUDIOSET_LINKS["class_labels_indices"]["fname"]
            )
            unbal_tags_fpath = osp.join(
                self.__dpath_data, AUDIOSET_LINKS["unbalanced"]["fname"]
            )

            if not all(map(osp.isfile, (class_labels_indices_fpath, unbal_tags_fpath))):
                raise FileNotFoundError(
                    f"Cannot load tags without tags files '{osp.basename(class_labels_indices_fpath)}' and '{osp.basename(unbal_tags_fpath)}'."
                    f"Please use download=True and with_tags=True in dataset constructor."
                )

            audioset_classes_data = AudioCaps.load_class_labels_indices(self._root)

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
        fnames_dic = dict.fromkeys(
            f"{line['youtube_id']}_{line['start_time']}.{self.AUDIO_FILE_EXTENSION}"
            for line in captions_data
        )
        audio_fnames_on_disk = dict.fromkeys(os.listdir(self.__dpath_audio_subset))
        if self._exclude_removed_audio:
            fnames_lst = [
                fname for fname in fnames_dic if fname in audio_fnames_on_disk
            ]
            is_on_disk_lst = [True for _ in range(len(fnames_lst))]
        else:
            fnames_lst = list(fnames_dic)
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
        assert len(classes_indexes) == 0 or classes_indexes == list(
            range(classes_indexes[-1] + 1)
        )
        self._index_to_tagname = list(tag_name_to_index.keys())

        # Process each field into a single structure
        all_caps_dic: Dict[str, List[Any]] = {
            key: [None for _ in range(dataset_size)]
            for key in ("audiocaps_ids", "youtube_id", "start_time", "captions")
        }
        for line in tqdm.tqdm(
            captions_data,
            disable=self._verbose <= 0,
            desc=f"Loading AudioCaps ({self._subset}) captions...",
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
            disable=self._verbose <= 0,
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
        }
        all_items.update(all_caps_dic)

        # Convert audiocaps_ids and start_time to ints
        all_items["audiocaps_ids"] = [
            list(map(int, item)) for item in all_items["audiocaps_ids"]
        ]
        all_items["start_time"] = list(map(int, all_items["start_time"]))

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

        if self.__is_prepared() and not self.FORCE_PREPARE_DATA:
            return None

        links = AUDIOCAPS_LINKS[self._subset]
        captions_fname = links["captions"]["fname"]
        captions_fpath = osp.join(self.__dpath_data, captions_fname)

        os.makedirs(self.__dpath_audio_subset, exist_ok=True)

        if not osp.isfile(captions_fpath):
            url = links["captions"]["url"]
            download_url_to_file(url, captions_fpath, progress=self._verbose >= 1)

        start = time.perf_counter()
        with open(captions_fpath, "r") as file:
            n_samples = len(file.readlines())

        if self._verbose >= 1:
            log_dpath = osp.join(self.__dpath_data, self.DNAME_LOG)
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

        with open(captions_fpath, "r") as file:
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
                    self.__dpath_audio_subset,
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
                        valid_file = self.__check_file(fpath)
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
                    valid_file = self.__check_file(fpath)
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
                            f'[{audiocap_id:6s}] File "{youtube_id}" is already downloaded but it is not verified due to self.VERIFY_FILES={self.VERIFY_FILES}.'
                        )
                    n_already_ok += 1

        if self._with_tags:
            for key in ("class_labels_indices", "unbalanced"):
                infos = AUDIOSET_LINKS[key]
                url = infos["url"]
                fname = infos["fname"]
                fpath = osp.join(self.__dpath_data, fname)
                if not osp.isfile(fpath):
                    if self._verbose >= 1:
                        logger.info(f"Downloading file '{fname}'...")
                    download_url_to_file(url, fpath, progress=self._verbose >= 1)

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
    """Download audio from youtube with youtube-dl and ffmpeg."""

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
