#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
import os.path as osp
import subprocess
import time

from subprocess import CalledProcessError
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
)

import torch
import torchaudio
import tqdm

from torch import Tensor
from torch.hub import download_url_to_file
from torchaudio.backend.common import AudioMetaData
from typing_extensions import TypedDict, NotRequired

from aac_datasets.datasets.base import AACDataset, DatasetCard
from aac_datasets.utils.paths import _get_root, _get_ffmpeg_path, _get_ytdl_path


pylog = logging.getLogger(__name__)


class AudioCapsItem(TypedDict, total=True):
    r"""Class representing a single AudioCaps item."""

    # Common attributes
    audio: Tensor
    captions: List[str]
    dataset: str
    fname: str
    index: int
    subset: str
    sr: int
    # AudioCaps-specific attributes
    audiocaps_ids: List[int]
    start_time: int
    tags: NotRequired[List[int]]
    youtube_id: str


class AudioCapsCard(DatasetCard):
    ANNOTATIONS_CREATORS: Tuple[str, ...] = ("crowdsourced",)
    CAPTIONS_PER_AUDIO: Dict[str, int] = {
        "train": 1,
        "val": 5,
        "test": 5,
        "train_v2": 1,
    }
    CITATION: str = r"""
    @inproceedings{kim_etal_2019_audiocaps,
        title        = {{A}udio{C}aps: Generating Captions for Audios in The Wild},
        author       = {Kim, Chris Dongjoo  and Kim, Byeongchang  and Lee, Hyunmin  and Kim, Gunhee},
        year         = 2019,
        month        = jun,
        booktitle    = {Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
        publisher    = {Association for Computational Linguistics},
        address      = {Minneapolis, Minnesota},
        pages        = {119--132},
        doi          = {10.18653/v1/N19-1011},
        url          = {https://aclanthology.org/N19-1011},
    }
    """
    HOMEPAGE: str = "https://audiocaps.github.io/"
    LANGUAGE: Tuple[str, ...] = ("en",)
    LANGUAGE_DETAILS: Tuple[str, ...] = ("en-US",)
    NAME: str = "audiocaps"
    PRETTY_NAME: str = "AudioCaps"
    SIZE_CATEGORIES: Tuple[str, ...] = ("10K<n<100K",)
    SUBSETS: Tuple[str, ...] = ("train", "val", "test", "train_v2")
    TASK_CATEGORIES: Tuple[str, ...] = ("audio-to-text", "text-to-audio")


class AudioCaps(AACDataset[AudioCapsItem]):
    r"""Unofficial AudioCaps PyTorch dataset.

    Subsets available are 'train', 'val' and 'test'.

    Audio is a waveform tensor of shape (1, n_times) of 10 seconds max, sampled at 32kHz by default.
    Target is a list of strings containing the captions.
    The 'train' subset has only 1 caption per sample and 'val' and 'test' have 5 captions.
    Download requires 'youtube-dl' and 'ffmpeg' commands.

    AudioCaps paper : https://www.aclweb.org/anthology/N19-1011.pdf

    .. code-block:: text
        :caption:  Dataset folder tree

        {root}
        └── AUDIOCAPS
            ├── train.csv
            ├── val.csv
            ├── test.csv
            └── audio_32000Hz
                ├── train
                │    └── (46231/49838 flac files, ~42G for 32kHz)
                ├── val
                │    └── (465/495 flac files, ~425M for 32kHz)
                └── test
                    └── (913/975 flac files, ~832M for 32kHz)

    """

    # Common globals
    CARD: ClassVar[AudioCapsCard] = AudioCapsCard()
    FORCE_PREPARE_DATA: ClassVar[bool] = False
    VERIFY_FILES: ClassVar[bool] = False

    # AudioCaps-specific globals
    AUDIO_DURATION: ClassVar[float] = 10.0
    AUDIO_FORMAT: ClassVar[str] = "flac"
    AUDIO_N_CHANNELS: ClassVar[int] = 1

    # Initialization
    def __init__(
        self,
        root: str = ...,
        subset: str = "train",
        download: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        flat_captions: bool = False,
        verbose: int = 0,
        exclude_removed_audio: bool = True,
        with_tags: bool = False,
        sr: int = 32_000,
        ffmpeg_path: str = ...,
        ytdl_path: str = ...,
    ) -> None:
        """
        :param root: Dataset root directory.
            The data will be stored in the 'AUDIOCAPS' subdirectory.
            defaults to ".".
        :param subset: The subset of AudioCaps to use. Can be one of :attr:`~AudioCapsCard.SUBSETS`.
            defaults to "train".
        :param download: Download the dataset if download=True and if the dataset is not already downloaded.
            defaults to False.
        :param transform: The transform to apply to the global dict item. This transform is applied only in getitem method when argument is an integer.
            defaults to None.
        :param flat_captions: If True, map captions to audio instead of audio to caption.
            defaults to True.
        :param verbose: Verbose level.
            defaults to 0.
        :param exclude_removed_audio: If True, the dataset will exclude from the dataset the audio not downloaded from youtube (i.e. not present on disk).
            If False, invalid audios will return an empty tensor of shape (0,).
            defaults to True.
        :param with_tags: If True, load the tags from AudioSet dataset.
            Note: tags needs to be downloaded with download=True & with_tags=True before being used.
            defaults to False.
        :param sr: The sample rate used for audio files in the dataset (in Hz).
            Since original YouTube videos are recorded in various settings, this parameter allow to download allow audio files with a specific sample rate.
            defaults to 32000.
        :param ffmpeg_path: Path to ffmpeg executable file.
            defaults to "ffmpeg".
        :param ytdl_path: Path to youtube-dl or ytdlp executable.
            defaults to "youtube-dl".
        """
        if subset not in AudioCapsCard.SUBSETS:
            raise ValueError(
                f"Invalid argument subset={subset} for AudioCaps. (expected one of {AudioCapsCard.SUBSETS})"
            )

        root = _get_root(root)
        ffmpeg_path = _get_ffmpeg_path(ffmpeg_path)
        ytdl_path = _get_ytdl_path(ytdl_path)

        if download:
            _prepare_audiocaps_dataset(
                root,
                subset,
                sr,
                with_tags,
                verbose,
                AudioCaps.FORCE_PREPARE_DATA,
                ytdl_path,
                ffmpeg_path,
                AudioCaps.AUDIO_FORMAT,
                AudioCaps.AUDIO_DURATION,
                AudioCaps.AUDIO_N_CHANNELS,
                AudioCaps.VERIFY_FILES,
            )

        raw_data, index_to_tagname = _load_audiocaps_dataset(
            root,
            subset,
            sr,
            with_tags,
            exclude_removed_audio,
            verbose,
            AudioCaps.AUDIO_FORMAT,
        )
        audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr)
        size = len(next(iter(raw_data.values())))
        raw_data["dataset"] = [AudioCapsCard.NAME] * size
        raw_data["subset"] = [subset] * size
        raw_data["fpath"] = [
            osp.join(audio_subset_dpath, fname) for fname in raw_data["fname"]
        ]
        raw_data["index"] = list(range(size))

        column_names = list(AudioCapsItem.__required_keys__) + list(
            AudioCapsItem.__optional_keys__
        )
        if not with_tags:
            column_names.remove("tags")

        super().__init__(
            raw_data=raw_data,
            transform=transform,
            column_names=column_names,
            flat_captions=flat_captions,
            sr=sr,
            verbose=verbose,
        )

        # Attributes
        self._root = root
        self._subset = subset
        self._download = download
        self._exclude_removed_audio = exclude_removed_audio
        self._with_tags = with_tags
        self._index_to_tagname = index_to_tagname

        self.add_post_columns(
            {
                "audio": AudioCaps._load_audio,
                "audio_metadata": AudioCaps._load_audio_metadata,
                "duration": AudioCaps._load_duration,
                "num_channels": AudioCaps._load_num_channels,
                "num_frames": AudioCaps._load_num_frames,
                "sr": AudioCaps._load_sr,
            }
        )

    # Properties
    @property
    def download(self) -> bool:
        return self._download

    @property
    def exclude_removed_audio(self) -> bool:
        return self._exclude_removed_audio

    @property
    def index_to_tagname(self) -> List[str]:
        return self._index_to_tagname

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
    def with_tags(self) -> bool:
        return self._with_tags

    # Public class methods
    @classmethod
    def load_class_labels_indices(
        cls,
        root: str,
        sr: int = 32_000,
    ) -> List[Dict[str, str]]:
        class_labels_indices_fpath = osp.join(
            _get_audiocaps_dpath(root, sr),
            _AUDIOSET_LINKS["class_labels_indices"]["fname"],
        )
        if not osp.isfile(class_labels_indices_fpath):
            raise ValueError(
                f"Cannot find class_labels_indices file in root='{root}'."
                f"Maybe use AudioCaps(root, download=True, with_tags=True) before or use a different root directory."
            )

        with open(class_labels_indices_fpath, "r") as file:
            reader = csv.DictReader(file)
            audioset_classes_data = list(reader)
        return audioset_classes_data

    # Magic methods
    def __repr__(self) -> str:
        repr_dic = {
            "subset": self._subset,
            "size": len(self),
            "num_columns": len(self.column_names),
            "with_tags": self._with_tags,
            "exclude_removed_audio": self._exclude_removed_audio,
        }
        repr_str = ", ".join(f"{k}={v}" for k, v in repr_dic.items())
        return f"{AudioCapsCard.PRETTY_NAME}({repr_str})"

    # Private methods
    def _load_audio(self, idx: int) -> Tensor:
        if not self._raw_data["is_on_disk"][idx]:
            return torch.empty((0,))
        fpath = self.at(idx, "fpath")
        audio, sr = torchaudio.load(fpath)  # type: ignore

        # Sanity check
        if audio.nelement() == 0:
            raise RuntimeError(
                f"Invalid audio number of elements in {fpath}. (expected audio.nelement()={audio.nelement()} > 0)"
            )

        if self._sr is not None and (self._sr != sr):
            raise RuntimeError(
                f"Invalid sample rate {sr}Hz for audio {fpath}. (expected {self._sr}Hz)"
            )
        return audio

    def _load_audio_metadata(self, idx: int) -> AudioMetaData:
        if not self._raw_data["is_on_disk"][idx]:
            return AudioMetaData(-1, -1, -1, -1, "unknown_encoding")
        fpath = self.at(idx, "fpath")
        audio_metadata = torchaudio.info(fpath)  # type: ignore
        return audio_metadata


def _get_audiocaps_dpath(root: str, sr: int) -> str:
    return osp.join(root, "AUDIOCAPS")


def _get_audio_subset_dpath(root: str, subset: str, sr: int) -> str:
    return osp.join(
        _get_audiocaps_dpath(root, sr),
        f"audio_{sr}Hz",
        _AUDIOCAPS_AUDIO_DNAMES[subset],
    )


def _is_prepared(root: str, subset: str, sr: int, verbose: int) -> bool:
    links = _AUDIOCAPS_LINKS[subset]
    captions_fname = links["captions"]["fname"]
    captions_fpath = osp.join(_get_audiocaps_dpath(root, sr), captions_fname)
    audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr)

    msgs = []

    if not osp.isdir(audio_subset_dpath):
        msgs.append(f"Cannot find directory '{audio_subset_dpath}'.")
    if not osp.isfile(captions_fpath):
        msgs.append(f"Cannot find file '{captions_fpath}'.")

    if verbose >= 0:
        for msg in msgs:
            pylog.warning(msg)

    return len(msgs) == 0


def _load_audiocaps_dataset(
    root: str,
    subset: str,
    sr: int,
    with_tags: bool,
    exclude_removed_audio: bool,
    verbose: int,
    audio_format: str = "flac",
) -> Tuple[Dict[str, List[Any]], List[str]]:
    if not _is_prepared(root, subset, sr, verbose):
        raise RuntimeError(
            f"Cannot load data: audiocaps_{subset} is not prepared in data root={root}. Please use download=True in dataset constructor."
        )

    links = _AUDIOCAPS_LINKS[subset]
    audiocaps_root = _get_audiocaps_dpath(root, sr)
    audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr)

    captions_fname = links["captions"]["fname"]
    captions_fpath = osp.join(audiocaps_root, captions_fname)
    with open(captions_fpath, "r") as file:
        reader = csv.DictReader(file)
        captions_data = list(reader)

    if with_tags:
        class_labels_indices_fpath = osp.join(
            audiocaps_root, _AUDIOSET_LINKS["class_labels_indices"]["fname"]
        )
        unbal_tags_fpath = osp.join(
            audiocaps_root, _AUDIOSET_LINKS["unbalanced"]["fname"]
        )

        if not all(map(osp.isfile, (class_labels_indices_fpath, unbal_tags_fpath))):
            raise FileNotFoundError(
                f"Cannot load tags without tags files '{osp.basename(class_labels_indices_fpath)}' and '{osp.basename(unbal_tags_fpath)}'."
                f"Please use download=True and with_tags=True in dataset constructor."
            )

        audioset_classes_data = AudioCaps.load_class_labels_indices(root, sr)

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
        f"{line['youtube_id']}_{line['start_time']}.{audio_format}"
        for line in captions_data
    )
    audio_fnames_on_disk = dict.fromkeys(os.listdir(audio_subset_dpath))
    if exclude_removed_audio:
        fnames_lst = [fname for fname in fnames_dic if fname in audio_fnames_on_disk]
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
    index_to_tagname = list(tag_name_to_index.keys())

    # Process each field into a single structure
    all_caps_dic: Dict[str, List[Any]] = {
        key: [None for _ in range(dataset_size)]
        for key in ("audiocaps_ids", "youtube_id", "start_time", "captions")
    }
    for line in tqdm.tqdm(
        captions_data,
        disable=verbose <= 0,
        desc=f"Loading AudioCaps ({subset}) captions...",
    ):
        # audiocap_id, youtube_id, start_time, caption
        audiocap_id = line["audiocap_id"]
        youtube_id = line["youtube_id"]
        start_time = line["start_time"]
        caption = line["caption"]

        fname = f"{youtube_id}_{start_time}.{audio_format}"
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
        disable=verbose <= 0,
        desc="Loading AudioSet tags for AudioCaps...",
    ):
        # keys: YTID, start_seconds, end_seconds, positive_labels
        youtube_id = line["YTID"]
        # Note : In audioset, start_time is a string repr of a float value, audiocaps it is a string repr of an integer
        start_time = int(float(line["start_seconds"]))
        fname = f"{youtube_id}_{start_time}.{audio_format}"
        if fname in fname_to_idx:
            tags_mid = line["positive_labels"]
            tags_mid = tags_mid.split(",")
            tags_names = [mid_to_tag_name[tag] for tag in tags_mid]
            tags_indexes = [tag_name_to_index[tag] for tag in tags_names]

            idx = fname_to_idx[fname]
            all_tags_lst[idx] = tags_indexes

    raw_data = {
        "fname": fnames_lst,
        "tags": all_tags_lst,
        "is_on_disk": is_on_disk_lst,
    }
    raw_data.update(all_caps_dic)

    # Convert audiocaps_ids and start_time to ints
    raw_data["audiocaps_ids"] = [
        list(map(int, item)) for item in raw_data["audiocaps_ids"]
    ]
    raw_data["start_time"] = list(map(int, raw_data["start_time"]))

    if verbose >= 1:
        pylog.info(
            f"{AudioCapsCard.PRETTY_NAME}(subset={subset}) has been loaded. (len={len(fnames_lst)})"
        )

    return raw_data, index_to_tagname


def _prepare_audiocaps_dataset(
    root: str,
    subset: str,
    sr: int,
    with_tags: bool,
    verbose: int,
    force: bool = False,
    ytdl_path: str = ...,
    ffmpeg_path: str = ...,
    audio_format: str = "flac",
    audio_duration: float = 10.0,
    n_channels: int = 1,
    verify_files: bool = False,
) -> None:
    if not osp.isdir(root):
        raise RuntimeError(f"Cannot find root directory '{root}'.")

    try:
        subprocess.check_call(
            [ytdl_path, "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (CalledProcessError, PermissionError, FileNotFoundError) as err:
        pylog.error(f"Cannot use youtube-dl path '{ytdl_path}'. ({err})")
        raise err

    try:
        subprocess.check_call(
            [ffmpeg_path, "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (CalledProcessError, PermissionError, FileNotFoundError) as err:
        pylog.error(f"Cannot use ffmpeg path '{ffmpeg_path}'. ({err})")
        raise err

    if _is_prepared(root, subset, sr, -1) and not force:
        return None

    links = _AUDIOCAPS_LINKS[subset]
    audiocaps_root = _get_audiocaps_dpath(root, sr)
    audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr)

    captions_fname = links["captions"]["fname"]
    captions_fpath = osp.join(audiocaps_root, captions_fname)

    os.makedirs(audio_subset_dpath, exist_ok=True)

    if not osp.isfile(captions_fpath):
        url = links["captions"]["url"]
        if url is None:
            raise ValueError(
                f"AudioCaps subset '{subset}' cannot be automatically downloaded. (found url={url})"
            )
        download_url_to_file(url, captions_fpath, progress=verbose >= 1)

    start = time.perf_counter()
    with open(captions_fpath, "r") as file:
        n_samples = len(file.readlines())

    if verbose >= 1:
        pylog.info(f"Start downloading files for {subset} AudioCaps split.")

    with open(captions_fpath, "r") as file:
        # Download audio files
        reader = csv.DictReader(file)
        captions_data = list(reader)

    n_download_ok = 0
    n_download_err = 0
    n_already_ok = 0
    n_already_err = 0

    for line in tqdm.tqdm(captions_data, total=n_samples, disable=verbose < 1):
        # Keys: audiocap_id, youtube_id, start_time, caption
        audiocap_id, youtube_id, start_time = [
            line[key] for key in ("audiocap_id", "youtube_id", "start_time")
        ]
        fpath = osp.join(
            audio_subset_dpath,
            f"{youtube_id}_{start_time}.{audio_format}",
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
                duration=audio_duration,
                sr=sr,
                ytdl_path=ytdl_path,
                ffmpeg_path=ffmpeg_path,
                n_channels=n_channels,
            )
            if success:
                valid_file = _check_file(fpath, sr)
                if valid_file:
                    if verbose >= 2:
                        pylog.debug(
                            f"[{audiocap_id:6s}] File '{youtube_id}' has been downloaded and verified."
                        )
                    n_download_ok += 1
                else:
                    if verbose >= 1:
                        pylog.warning(
                            f"[{audiocap_id:6s}] File '{youtube_id}' has been downloaded but it is not valid and it will be removed."
                        )
                    os.remove(fpath)
                    n_download_err += 1
            else:
                pylog.error(
                    f"[{audiocap_id:6s}] Cannot extract audio from {youtube_id}."
                )
                n_download_err += 1

        elif verify_files:
            valid_file = _check_file(fpath, sr)
            if valid_file:
                if verbose >= 2:
                    pylog.debug(
                        f"[{audiocap_id:6s}] File '{youtube_id}' is already downloaded and has been verified."
                    )
                n_already_ok += 1
            else:
                if verbose >= 1:
                    pylog.warning(
                        f"[{audiocap_id:6s}] File '{youtube_id}' is already downloaded but it is not valid and will be removed."
                    )
                os.remove(fpath)
                n_already_err += 1
        else:
            if verbose >= 2:
                pylog.debug(
                    f"[{audiocap_id:6s}] File '{youtube_id}' is already downloaded but it is not verified due to verify_files={verify_files}."
                )
            n_already_ok += 1

    if with_tags:
        for key in ("class_labels_indices", "unbalanced"):
            infos = _AUDIOSET_LINKS[key]
            url = infos["url"]
            fname = infos["fname"]
            fpath = osp.join(audiocaps_root, fname)
            if not osp.isfile(fpath):
                if verbose >= 1:
                    pylog.info(f"Downloading file '{fname}'...")
                download_url_to_file(url, fpath, progress=verbose >= 1)

    if verbose >= 1:
        duration = int(time.perf_counter() - start)
        pylog.info(
            f"Download and preparation of AudioCaps for subset '{subset}' done in {duration}s."
        )
        pylog.info(f"- {n_download_ok} downloads success,")
        pylog.info(f"- {n_download_err} downloads failed,")
        pylog.info(f"- {n_already_ok} already downloaded,")
        pylog.info(f"- {n_already_err} already downloaded errors,")
        pylog.info(f"- {n_samples} total samples.")

    if verbose >= 2:
        pylog.debug(
            f"Dataset {AudioCapsCard.PRETTY_NAME} (subset={subset}) has been prepared."
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
    ytdl_path: str = ...,
    ffmpeg_path: str = ...,
) -> bool:
    """Download audio from youtube with youtube-dl and ffmpeg."""

    # Get audio download link with youtube-dl, without start time
    link = _get_youtube_link(youtube_id, None)
    get_url_command = [
        ytdl_path,
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
        # Resample to a specific rate (default to 32 kHz)
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


def _check_file(fpath: str, expected_sr: Optional[int]) -> bool:
    try:
        audio, sr = torchaudio.load(fpath)  # type: ignore
    except RuntimeError:
        message = (
            f"Found file '{fpath}' already downloaded but it is invalid (cannot load)."
        )
        pylog.error(message)
        return False

    if audio.nelement() == 0:
        message = (
            f"Found file '{fpath}' already downloaded but it is invalid (empty audio)."
        )
        pylog.error(message)
        return False

    if expected_sr is not None and sr != expected_sr:
        message = f"Found file '{fpath}' already downloaded but it is invalid (invalid sr={sr} != {expected_sr})."
        pylog.error(message)
        return False

    return True


def _get_youtube_link(youtube_id: str, start_time: Optional[int]) -> str:
    link = f"https://www.youtube.com/watch?v={youtube_id}"
    if start_time is None:
        return link
    else:
        return f"{link}&t={start_time}s"


def _get_youtube_link_embed(
    youtube_id: str, start_time: Optional[int], duration: float = 10.0
) -> str:
    link = f"https://www.youtube.com/embed/{youtube_id}"
    if start_time is None:
        return link
    else:
        end_time = start_time + duration
        return f"{link}?start={start_time}&end={end_time}"


# Audio directory names per subset
_AUDIOCAPS_AUDIO_DNAMES = {
    "train": "train",
    "val": "val",
    "test": "test",
    "train_v2": "train",
}

# Archives and file links used to download AudioCaps labels and metadata
_AUDIOCAPS_LINKS = {
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
    "train_v2": {
        "captions": {
            "url": "https://raw.githubusercontent.com/Labbeti/aac-datasets/dev/data/train_v2.csv",
            "fname": "train_v2.csv",
        },
    },
}

# Archives and file links used to download AudioSet metadata
_AUDIOSET_LINKS = {
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
