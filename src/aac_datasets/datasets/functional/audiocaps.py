#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
import os.path as osp
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torchaudio
import tqdm
from typing_extensions import Literal

from aac_datasets.datasets.functional.common import DatasetCard, LinkInfo
from aac_datasets.utils.audioset_mapping import (
    download_audioset_mapping,
    load_audioset_mapping,
)
from aac_datasets.utils.collections import union_dicts
from aac_datasets.utils.download import download_file
from aac_datasets.utils.globals import _get_ffmpeg_path, _get_root, _get_ytdlp_path

pylog = logging.getLogger(__name__)

AudioCapsSubset = Literal["train", "val", "test", "train_fixed"]
AudioCapsVersion = Literal["v1", "v2"]


class AudioCapsCard(DatasetCard):
    ANNOTATIONS_CREATORS: Tuple[str, ...] = ("crowdsourced",)
    CAPTIONS_PER_AUDIO: Dict[AudioCapsSubset, int] = {
        "train": 1,
        "val": 5,
        "test": 5,
        "train_fixed": 1,
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
    DEFAULT_SUBSET: AudioCapsSubset = "train"
    DEFAULT_VERSION: AudioCapsVersion = "v1"
    HOMEPAGE: str = "https://audiocaps.github.io/"
    LANGUAGE: Tuple[str, ...] = ("en",)
    LANGUAGE_DETAILS: Tuple[str, ...] = ("en-US",)
    NAME: str = "audiocaps"
    PRETTY_NAME: str = "AudioCaps"
    SIZE_CATEGORIES: Tuple[str, ...] = ("10K<n<100K",)
    SUBSETS: Tuple[AudioCapsSubset, ...] = tuple(CAPTIONS_PER_AUDIO.keys())
    TASK_CATEGORIES: Tuple[str, ...] = ("audio-to-text", "text-to-audio")
    VERSIONS: Tuple[AudioCapsVersion, ...] = ("v1", "v2")


def load_audiocaps_dataset(
    # Common args
    root: Union[str, Path, None] = None,
    subset: AudioCapsSubset = AudioCapsCard.DEFAULT_SUBSET,
    verbose: int = 0,
    *,
    # AudioCaps-specific args
    audio_format: str = "flac",
    exclude_removed_audio: bool = True,
    sr: int = 32_000,
    with_tags: bool = False,
    version: AudioCapsVersion = AudioCapsCard.DEFAULT_VERSION,
) -> Tuple[Dict[str, List[Any]], Dict[int, str]]:
    """Load AudioCaps metadata.

    :param root: Dataset root directory.
        The data will be stored in the 'AUDIOCAPS' subdirectory.
        defaults to ".".
    :param subset: The subset of AudioCaps to use. Can be one of :attr:`~AudioCapsCard.SUBSETS`.
        defaults to "train".
    :param verbose: Verbose level.
        defaults to 0.

    :param audio_format: Audio format and extension name.
        defaults to "flac".
    :param exclude_removed_audio: If True, the dataset will exclude from the dataset the audio not downloaded from youtube (i.e. not present on disk).
        If False, invalid audios will return an empty tensor of shape (0,).
        defaults to True.
    :param sr: The sample rate used for audio files in the dataset (in Hz).
        Since original YouTube videos are recorded in various settings, this parameter allow to download allow audio files with a specific sample rate.
        defaults to 32000.
    :param with_tags: If True, load the tags from AudioSet dataset.
        Note: tags needs to be downloaded with download=True & with_tags=True before being used.
        defaults to False.
    :param version: The version of the dataset. Can be one of :attr:`~AudioCapsCard.VERSIONS`.
        defaults to 'v1'.

    :returns: A dictionnary of lists containing each metadata.
        Expected keys: "audiocaps_ids", "youtube_id", "start_time", "captions", "fname", "tags", "is_on_disk".
    """
    if subset in _AUDIOCAPS_OLD_SUBSETS_NAMES:
        new_subset = _AUDIOCAPS_OLD_SUBSETS_NAMES[subset]
        if verbose >= 0:
            msg = f"Deprecated subset name '{subset}', use '{new_subset}' instead."
            pylog.warning(msg)
        subset = new_subset

    root = _get_root(root)
    audiocaps_root = _get_audiocaps_root(root, sr, version)
    audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr, version)

    is_prepared = _is_prepared_audiocaps(
        root,
        subset,
        sr,
        audio_format,
        verbose=verbose,
        version=version,
    )
    if not is_prepared:
        msg = f"Cannot load data: audiocaps_{subset} is not prepared in data {root=}. Please use download=True in dataset constructor."
        raise RuntimeError(msg)

    version_links = _AUDIOCAPS_LINKS[version]
    if subset not in version_links:
        msg = f"Subset {subset} is not available for AudioCaps version {version}. (expected one of {tuple(version_links.keys())})"
        raise ValueError(msg)

    links = version_links[subset]

    captions_dpath = _get_captions_dpath(root, subset, sr, version)
    captions_fname = links["captions"]["fname"]
    captions_fpath = osp.join(captions_dpath, captions_fname)

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

        mid_to_index: Dict[str, int] = load_audioset_mapping(
            "mid",
            "index",
            offline=True,
            cache_path=audiocaps_root,
            verbose=verbose,
        )
        index_to_name: Dict[int, str] = load_audioset_mapping(
            "index",
            "display_name",
            offline=True,
            cache_path=audiocaps_root,
            verbose=verbose,
        )

        with open(unbal_tags_fpath, "r") as file:
            FIELDNAMES = ("YTID", "start_seconds", "end_seconds", "positive_labels")
            reader = csv.DictReader(
                file,
                FIELDNAMES,
                skipinitialspace=True,
                strict=True,
            )
            # Skip the comments
            for _ in range(3):
                next(reader)
            unbal_tags_data = list(reader)
    else:
        mid_to_index = {}
        index_to_name = {}
        unbal_tags_data = []

    # Build global mappings
    fnames_dic = dict.fromkeys(
        _AUDIO_FNAME_FORMAT.format(**line, audio_format=audio_format)
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

        fname = _AUDIO_FNAME_FORMAT.format(**line, audio_format=audio_format)
        if fname in fname_to_idx:
            index = fname_to_idx[fname]

            if all_caps_dic["start_time"][index] is None:
                all_caps_dic["start_time"][index] = start_time
                all_caps_dic["youtube_id"][index] = youtube_id
                all_caps_dic["audiocaps_ids"][index] = [audiocap_id]
                all_caps_dic["captions"][index] = [caption]
            else:
                # sanity check
                assert all_caps_dic["start_time"][index] == start_time
                assert all_caps_dic["youtube_id"][index] == youtube_id

                all_caps_dic["audiocaps_ids"][index].append(audiocap_id)
                all_caps_dic["captions"][index].append(caption)

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
        fname = _AUDIO_FNAME_FORMAT.format(
            youtube_id=youtube_id,
            start_time=start_time,
            audio_format=audio_format,
        )
        if fname in fname_to_idx:
            tags_mid = line["positive_labels"]
            tags_mid = tags_mid.split(",")
            tags_indexes = [mid_to_index[tag_mid] for tag_mid in tags_mid]

            index = fname_to_idx[fname]
            all_tags_lst[index] = tags_indexes

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
        msg = f"{AudioCapsCard.PRETTY_NAME}(subset={subset}) has been loaded. {len(fnames_lst)=})"
        pylog.info(msg)

    return raw_data, index_to_name


def download_audiocaps_dataset(
    # Common args
    root: Union[str, Path, None] = None,
    subset: AudioCapsSubset = AudioCapsCard.DEFAULT_SUBSET,
    force: bool = False,
    verbose: int = 0,
    verify_files: bool = False,
    *,
    # AudioCaps-specific args
    audio_duration: float = 10.0,
    audio_format: str = "flac",
    audio_n_channels: int = 1,
    download_audio: bool = True,
    ffmpeg_path: Union[str, Path, None] = None,
    max_workers: Optional[int] = 1,
    sr: int = 32_000,
    ytdlp_path: Union[str, Path, None] = None,
    with_tags: bool = False,
    version: AudioCapsVersion = AudioCapsCard.DEFAULT_VERSION,
) -> None:
    """Prepare AudioCaps data (audio, labels, metadata).

    :param root: Dataset root directory.
        The data will be stored in the 'AUDIOCAPS' subdirectory.
        defaults to ".".
    :param subset: The subset of AudioCaps to use. Can be one of :attr:`~AudioCapsCard.SUBSETS`.
        defaults to "train".
    :param force: If True, force to re-download file even if they exists on disk.
        defaults to False.
    :param verbose: Verbose level.
        defaults to 0.
    :param verify_files: If True, check hash value when possible.
        defaults to True.

    :param audio_duration: Extracted duration for each audio file in seconds.
        defaults to 10.0.
    :param audio_format: Audio format and extension name.
        defaults to "flac".
    :param audio_n_channels: Number of channels extracted for each audio file.
        defaults to 1.
    :param download_audio: If True, download audio, metadata and labels files. Otherwise it will only donwload metadata and labels files.
        defaults to True.
    :param ffmpeg_path: Path to ffmpeg executable file.
        defaults to "ffmpeg".
    :param max_workers: Number of threads to download audio files in parallel.
        Do not use a value too high to avoid "Too Many Requests" error.
        The value None will use `min(32, os.cpu_count() + 4)` workers, which is the default of ThreadPoolExecutor.
        defaults to 1.
    :param sr: The sample rate used for audio files in the dataset (in Hz).
        Since original YouTube videos are recorded in various settings, this parameter allow to download allow audio files with a specific sample rate.
        defaults to 32000.
    :param with_tags: If True, download the tags from AudioSet dataset.
        defaults to False.
    :param ytdlp_path: Path to yt-dlp or ytdlp executable.
        defaults to "yt-dlp".
    :param version: The version of the dataset. Can be one of :attr:`~AudioCapsCard.VERSIONS`.
        defaults to 'v1'.
    """

    root = _get_root(root)
    ytdlp_path = _get_ytdlp_path(ytdlp_path)
    ffmpeg_path = _get_ffmpeg_path(ffmpeg_path)

    if not osp.isdir(root):
        raise RuntimeError(f"Cannot find root directory '{root}'.")

    _check_subprog_help(ytdlp_path, "ytdlp")
    _check_subprog_help(ffmpeg_path, "ffmpeg")

    is_prepared = _is_prepared_audiocaps(
        root, subset, sr, audio_format=audio_format, verbose=-1, version=version
    )
    if is_prepared and not force:
        return None

    audiocaps_root = _get_audiocaps_root(root, sr, version)
    os.makedirs(audiocaps_root, exist_ok=True)
    if with_tags:
        _download_tags_files(root, sr, version, verbose)

    version_links = _AUDIOCAPS_LINKS[version]
    if subset not in version_links:
        msg = f"Subset {subset} is not available for AudioCaps version {version}. (expected one of {tuple(version_links.keys())})"
        raise ValueError(msg)
    links = version_links[subset]

    audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr, version)
    os.makedirs(audio_subset_dpath, exist_ok=True)

    captions_dpath = _get_captions_dpath(root, subset, sr, version)
    os.makedirs(captions_dpath, exist_ok=True)

    captions_fname = links["captions"]["fname"]
    captions_fpath = osp.join(captions_dpath, captions_fname)

    if not osp.isfile(captions_fpath):
        url = links["captions"]["url"]
        if url is None:
            msg = f"AudioCaps subset '{subset}' cannot be automatically downloaded. (found {url=})"
            raise ValueError(msg)
        download_file(url, captions_fpath, verbose=verbose)

    if download_audio:
        _download_audio_files(
            subset=subset,
            verbose=verbose,
            verify_files=verify_files,
            captions_fpath=captions_fpath,
            audio_subset_dpath=audio_subset_dpath,
            audio_duration=audio_duration,
            audio_format=audio_format,
            audio_n_channels=audio_n_channels,
            ffmpeg_path=ffmpeg_path,
            max_workers=max_workers,
            sr=sr,
            ytdlp_path=ytdlp_path,
        )

    if verbose >= 2:
        msg = f"Dataset {AudioCapsCard.PRETTY_NAME} {subset=}) has been prepared."
        pylog.debug(msg)


def download_audiocaps_datasets(
    # Common args
    root: Union[str, Path, None] = None,
    subsets: Union[
        AudioCapsSubset, Iterable[AudioCapsSubset]
    ] = AudioCapsCard.DEFAULT_SUBSET,
    force: bool = False,
    verbose: int = 0,
    verify_files: bool = False,
    *,
    # AudioCaps-specific args
    audio_duration: float = 10.0,
    audio_format: str = "flac",
    audio_n_channels: int = 1,
    download_audio: bool = True,
    ffmpeg_path: Union[str, Path, None] = None,
    max_workers: Optional[int] = 1,
    sr: int = 32_000,
    with_tags: bool = False,
    ytdlp_path: Union[str, Path, None] = None,
    version: AudioCapsVersion = AudioCapsCard.DEFAULT_VERSION,
) -> None:
    """Function helper to download a list of subsets. See :func:`~aac_datasets.datasets.functional.audiocaps.download_audiocaps_dataset` for details."""
    if isinstance(subsets, str):
        subsets = [subsets]
    else:
        subsets = list(subsets)

    kwargs: Dict[str, Any] = dict(
        root=root,
        force=force,
        verbose=verbose,
        verify_files=verify_files,
        audio_duration=audio_duration,
        audio_format=audio_format,
        audio_n_channels=audio_n_channels,
        download_audio=download_audio,
        ffmpeg_path=ffmpeg_path,
        max_workers=max_workers,
        sr=sr,
        with_tags=with_tags,
        ytdlp_path=ytdlp_path,
        version=version,
    )
    for subset in subsets:
        download_audiocaps_dataset(
            subset=subset,
            **kwargs,
        )


def _download_audio_files(
    subset: AudioCapsSubset,
    verbose: int,
    verify_files: bool,
    *,
    captions_fpath: str,
    audio_subset_dpath: str,
    audio_duration: float,
    audio_format: str,
    audio_n_channels: int,
    ffmpeg_path: str,
    max_workers: Optional[int],
    sr: int,
    ytdlp_path: str,
) -> None:
    start = time.perf_counter()
    if verbose >= 1:
        pylog.info(f"Start downloading audio files for AudioCaps {subset} split...")

    with open(captions_fpath, "r") as file:
        # Download audio files
        reader = csv.DictReader(file)
        # Keys: audiocap_id, youtube_id, start_time, caption
        captions_data = list(reader)

    def _cast_line(line: Dict[str, Any], audio_format: str) -> Dict[str, Any]:
        youtube_id = line["youtube_id"]
        start_time = line["start_time"]

        if not start_time.isdigit():
            msg = f"Start time '{start_time}' is not an integer (with {youtube_id=})."
            raise RuntimeError(msg)

        start_time = int(start_time)
        fname = _AUDIO_FNAME_FORMAT.format(
            youtube_id=youtube_id,
            start_time=start_time,
            audio_format=audio_format,
        )
        line = union_dicts(line, {"start_time": start_time, "fname": fname})
        return line

    captions_data = [_cast_line(line, audio_format) for line in captions_data]
    download_kwds = {
        line["fname"]: {k: line[k] for k in ("fname", "youtube_id", "start_time")}
        for line in captions_data
    }
    del captions_data

    present_audio_fnames = os.listdir(audio_subset_dpath)
    present_audio_fpaths = [
        osp.join(audio_subset_dpath, fname) for fname in present_audio_fnames
    ]
    present_audio_fpaths = dict.fromkeys(present_audio_fpaths)

    common_kwds: Dict[str, Any] = dict(
        audio_subset_dpath=audio_subset_dpath,
        verify_files=verify_files,
        present_audio_fpaths=present_audio_fpaths,
        audio_duration=audio_duration,
        sr=sr,
        audio_n_channels=audio_n_channels,
        ffmpeg_path=ffmpeg_path,
        ytdlp_path=ytdlp_path,
        verbose=verbose,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if verbose >= 2:
            pylog.debug(f"Using {executor._max_workers} workers.")

        submitted_dict = {
            fname: executor.submit(
                _download_from_youtube_and_verify,
                **kwds,
                **common_kwds,
            )
            for fname, kwds in download_kwds.items()
        }
        for i, (fname, submitted) in enumerate(
            tqdm.tqdm(submitted_dict.items(), disable=verbose < 1)
        ):
            file_exists, download_success, valid_file = submitted.result()

            if verbose < 2:
                continue

            if not file_exists:
                if not download_success:
                    msg_end = f"File '{fname}' cannot be downloaded. (maybe the source video has been removed?)"
                elif valid_file:
                    msg_end = f"File '{fname}' has been downloaded and verified."
                elif verify_files:
                    msg_end = f"File '{fname}' has been downloaded but it was not valid and has been removed."
                else:
                    msg_end = f"File '{fname}' has been downloaded."
            else:
                if valid_file:
                    msg_end = (
                        f"File '{fname}' is already downloaded and has been verified."
                    )
                elif verify_files:
                    msg_end = f"File '{fname}' is already downloaded but it was not valid and has been removed."
                else:
                    msg_end = f"File '{fname}' is already downloaded."

            pylog.debug(f"[{i+1:5d}/{len(download_kwds)}] {msg_end}")

    if verbose >= 1:
        duration_s = int(time.perf_counter() - start)
        msgs = (
            f"Download and preparation of AudioCaps for subset '{subset}' done in {duration_s}s.",
            f"- {len(download_kwds)} total samples.",
        )
        pylog.info("\n".join(msgs))


def _download_tags_files(
    root: Union[str, Path, None],
    sr: int,
    version: AudioCapsVersion,
    verbose: int,
) -> None:
    root = _get_root(root)
    audiocaps_root = _get_audiocaps_root(root, sr, version)

    target = "unbalanced"
    infos = _AUDIOSET_LINKS[target]
    url = infos["url"]
    fname = infos["fname"]
    fpath = osp.join(audiocaps_root, fname)
    if not osp.isfile(fpath):
        if verbose >= 1:
            pylog.info(f"Downloading file '{fname}'...")
        download_file(url, fpath, verbose=verbose)

    download_audioset_mapping(audiocaps_root, verbose=verbose)


def _get_audiocaps_root(root: str, sr: int, version: AudioCapsVersion) -> str:
    return osp.join(root, "AUDIOCAPS")


def _get_audio_subset_dpath(
    root: str,
    subset: AudioCapsSubset,
    sr: int,
    version: AudioCapsVersion,
) -> str:
    return osp.join(
        _get_audiocaps_root(root, sr, version),
        f"audio_{sr}Hz",
        _AUDIOCAPS_AUDIO_DNAMES[subset],
    )


def _get_captions_dpath(
    root: str,
    subset: AudioCapsSubset,
    sr: int,
    version: AudioCapsVersion,
) -> str:
    audiocaps_root = _get_audiocaps_root(root, sr, version)
    captions_fname = _AUDIOCAPS_LINKS[version][subset]["captions"]["fname"]
    captions_fpath = osp.join(audiocaps_root, captions_fname)
    # For backward compatibility only
    if version == "v1" and osp.isfile(captions_fpath):
        return audiocaps_root
    else:
        return osp.join(audiocaps_root, f"csv_files_{version}")


def _is_prepared_audiocaps(
    root: str,
    subset: AudioCapsSubset = AudioCapsCard.DEFAULT_SUBSET,
    sr: int = 32_000,
    audio_format: str = "flac",
    verbose: int = 0,
    version: AudioCapsVersion = AudioCapsCard.DEFAULT_VERSION,
) -> bool:
    version_links = _AUDIOCAPS_LINKS[version]
    if subset not in version_links:
        msg = f"Subset {subset} is not available for AudioCaps version {version}. (expected one of {tuple(version_links.keys())})"
        raise ValueError(msg)
    links = version_links[subset]

    captions_dpath = _get_captions_dpath(root, subset, sr, version)
    captions_fname = links["captions"]["fname"]
    captions_fpath = osp.join(captions_dpath, captions_fname)
    audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr, version)

    msgs = []

    if not osp.isdir(audio_subset_dpath):
        msg = f"Cannot find directory '{audio_subset_dpath}'."
        msgs.append(msg)
    else:
        audio_fnames = os.listdir(audio_subset_dpath)
        audio_fnames = [fname for fname in audio_fnames if fname.endswith(audio_format)]
        if len(audio_fnames) == 0:
            msg = (
                f"Cannot find any audio {audio_format} file in '{audio_subset_dpath}'."
            )
            msgs.append(msg)

    if not osp.isfile(captions_fpath):
        msg = f"Cannot find file '{captions_fpath}'."
        msgs.append(msg)

    if verbose >= 0:
        for msg in msgs:
            pylog.warning(msg)

    return len(msgs) == 0


def _download_from_youtube_and_verify(
    fname: str,
    youtube_id: str,
    start_time: int,
    audio_subset_dpath: str,
    verify_files: bool,
    present_audio_fpaths: Dict[str, None],
    audio_duration: float,
    sr: int,
    audio_n_channels: int,
    ffmpeg_path: str,
    ytdlp_path: str,
    verbose: int,
) -> Tuple[bool, bool, bool]:
    fpath = osp.join(audio_subset_dpath, fname)

    file_exists = fpath in present_audio_fpaths
    download_success = False
    valid_file = False

    if not file_exists:
        download_success = _download_from_youtube(
            youtube_id=youtube_id,
            fpath_out=fpath,
            start_time=start_time,
            audio_duration=audio_duration,
            sr=sr,
            audio_n_channels=audio_n_channels,
            ffmpeg_path=ffmpeg_path,
            ytdlp_path=ytdlp_path,
            verbose=verbose,
        )

    if verify_files and (download_success or file_exists):
        valid_file = _is_valid_audio_file(
            fpath,
            min_n_frames=1,
            sr=sr,
            n_channels=audio_n_channels,
        )

    if verify_files and not valid_file and osp.isfile(fpath):
        os.remove(fpath)

    return file_exists, download_success, valid_file


def _download_from_youtube(
    youtube_id: str,
    fpath_out: str,
    start_time: int,
    audio_duration: float = 10.0,
    sr: int = 32_000,
    audio_n_channels: int = 1,
    audio_format: str = "flac",
    acodec: str = "flac",
    ytdlp_path: Union[str, Path, None] = None,
    ffmpeg_path: Union[str, Path, None] = None,
    verbose: int = 0,
) -> bool:
    """Download audio from youtube with yt-dlp and ffmpeg."""
    ytdlp_path = _get_ytdlp_path(ytdlp_path)
    ffmpeg_path = _get_ffmpeg_path(ffmpeg_path)

    # Get audio download link with yt-dlp, without start time
    link = _get_youtube_link(youtube_id, None)
    get_url_command = [
        ytdlp_path,
        "--youtube-skip-dash-manifest",
        "-g",
        link,
    ]
    try:
        output = subprocess.check_output(get_url_command)
    except (CalledProcessError, PermissionError) as err:
        if verbose >= 2:
            pylog.debug(err)
        return False

    output = output.decode()
    lines = output.split("\n")
    if len(lines) < 2:
        return False
    _video_link, audio_link = lines[:2]

    # if yt-dlp only returns one link, it is a combined video audio
    if len(audio_link) == 0:
        if verbose >= 2:
            pylog.debug(
                f"youtube_id={youtube_id} is combined video audio only (cant download)"
            )
        # audio_link = video_link # this does not work, not sure why. probably requires changes to ffmpeg command
        return False

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
        audio_format,
        # Audio codec (flac)
        "-acodec",
        acodec,
        # Get only 10s of the clip after start_time
        "-ss",
        str(start_time),
        "-t",
        str(audio_duration),
        # Resample to a specific rate (default to 32 kHz)
        "-ar",
        str(sr),
        # Compute mean of 2 channels
        "-ac",
        str(audio_n_channels),
        fpath_out,
    ]
    try:
        if verbose < 3:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        else:
            stdout = None
            stderr = None
        exitcode = subprocess.check_call(extract_command, stdout=stdout, stderr=stderr)
        return exitcode == 0

    except (CalledProcessError, PermissionError) as err:
        if verbose >= 2:
            pylog.debug(err)
        return False


def _check_subprog_help(
    path: str,
    name: str,
    stdout: Any = subprocess.DEVNULL,
    stderr: Any = subprocess.DEVNULL,
) -> None:
    try:
        subprocess.check_call(
            [path, "--help"],
            stdout=stdout,
            stderr=stderr,
        )
    except (CalledProcessError, PermissionError, FileNotFoundError) as err:
        pylog.error(f"Invalid {name} path '{path}'. ({err})")
        raise err


def _is_valid_audio_file(
    fpath: str,
    *,
    min_n_frames: Optional[int] = None,
    max_n_frames: Optional[int] = None,
    sr: Optional[int] = None,
    n_channels: Optional[int] = None,
) -> bool:
    try:
        metadata = torchaudio.info(fpath)  # type: ignore
    except RuntimeError:
        msg = f"Found file '{fpath}' already downloaded but it is invalid (cannot load metadata)."
        pylog.error(msg)
        return False

    msgs = []
    if min_n_frames is not None and metadata.num_frames < min_n_frames:
        msg = f"Found file '{fpath}' already downloaded but it is invalid (audio is shorter than {min_n_frames=} samples)."
        msgs.append(msg)

    if max_n_frames is not None and metadata.num_frames > max_n_frames:
        msg = f"Found file '{fpath}' already downloaded but it is invalid (audio is longer than {max_n_frames=} samples)."
        msgs.append(msg)

    if sr is not None and metadata.sample_rate != sr:
        msg = f"Found file '{fpath}' already downloaded but it is invalid (invalid {metadata.sample_rate=} != {sr})."
        msgs.append(msg)

    if n_channels is not None and metadata.num_channels != n_channels:
        msg = f"Found file '{fpath}' already downloaded but it is invalid (invalid {metadata.num_channels=} != {sr})."
        msgs.append(msg)

    for msg in msgs:
        pylog.error(msg)

    return len(msgs) == 0


def _get_youtube_link(youtube_id: str, start_time: Optional[int]) -> str:
    link = f"https://www.youtube.com/watch?v={youtube_id}"
    if start_time is None:
        return link
    else:
        return f"{link}&t={start_time}s"


# Audio directory names per subset
_AUDIOCAPS_AUDIO_DNAMES: Dict[AudioCapsSubset, str] = {
    "train": "train",
    "val": "val",
    "test": "test",
    "train_fixed": "train",
}

# Internal typing to make easier to add new links without error
_AudioCapsLinkType = Literal["captions"]

# Archives and file links used to download AudioCaps labels and metadata
_AUDIOCAPS_LINKS: Dict[
    AudioCapsVersion, Dict[AudioCapsSubset, Dict[_AudioCapsLinkType, LinkInfo]]
] = {
    "v1": {
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
        "train_fixed": {
            "captions": {
                "url": "https://raw.githubusercontent.com/Labbeti/aac-datasets/dev/data/audiocaps/train_fixed.csv",
                "fname": "train_fixed.csv",
            },
        },
    },
    "v2": {
        "train": {
            "captions": {
                "url": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset2.0/train.csv",
                "fname": "train.csv",
            },
        },
        "val": {
            "captions": {
                "url": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset2.0/val.csv",
                "fname": "val.csv",
            },
        },
        "test": {
            "captions": {
                "url": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset2.0/test.csv",
                "fname": "test.csv",
            },
        },
    },
}

# Archives and file links used to download AudioSet metadata
_AUDIOSET_LINKS: Dict[str, LinkInfo] = {
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

# Audio filename format for AudioCaps
_AUDIO_FNAME_FORMAT = "{youtube_id}_{start_time}.{audio_format}"

_AUDIOCAPS_OLD_SUBSETS_NAMES: Dict[str, AudioCapsSubset] = {
    "train_v2": "train_fixed",
}
