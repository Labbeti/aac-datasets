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
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import torchaudio
import tqdm

from aac_datasets.datasets.functional.common import DatasetCard
from aac_datasets.utils.download import download_file
from aac_datasets.utils.globals import _get_root, _get_ffmpeg_path, _get_ytdlp_path


pylog = logging.getLogger(__name__)


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
    DEFAULT_SUBSET: str = "train"
    HOMEPAGE: str = "https://audiocaps.github.io/"
    LANGUAGE: Tuple[str, ...] = ("en",)
    LANGUAGE_DETAILS: Tuple[str, ...] = ("en-US",)
    NAME: str = "audiocaps"
    PRETTY_NAME: str = "AudioCaps"
    SIZE_CATEGORIES: Tuple[str, ...] = ("10K<n<100K",)
    SUBSETS: Tuple[str, ...] = tuple(CAPTIONS_PER_AUDIO.keys())
    TASK_CATEGORIES: Tuple[str, ...] = ("audio-to-text", "text-to-audio")


def load_audiocaps_dataset(
    # Common args
    root: Union[str, Path, None] = None,
    subset: str = AudioCapsCard.DEFAULT_SUBSET,
    verbose: int = 0,
    # AudioCaps-specific args
    audio_format: str = "flac",
    exclude_removed_audio: bool = True,
    sr: int = 32_000,
    with_tags: bool = False,
) -> Tuple[Dict[str, List[Any]], List[str]]:
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
    :returns: A dictionnary of lists containing each metadata.
        Expected keys: "audiocaps_ids", "youtube_id", "start_time", "captions", "fname", "tags", "is_on_disk".
    """

    root = _get_root(root)
    audiocaps_root = _get_audiocaps_root(root, sr)
    audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr)

    if not _is_prepared_audiocaps(root, subset, sr, verbose):
        raise RuntimeError(
            f"Cannot load data: audiocaps_{subset} is not prepared in data root={root}. Please use download=True in dataset constructor."
        )

    links = _AUDIOCAPS_LINKS[subset]
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

        audioset_classes_data = load_class_labels_indices(root, sr)

        with open(unbal_tags_fpath, "r") as file:
            FIELDNAMES = ("YTID", "start_seconds", "end_seconds", "positive_labels")
            reader = csv.DictReader(
                file, FIELDNAMES, skipinitialspace=True, strict=True
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


def download_audiocaps_dataset(
    # Common args
    root: Union[str, Path, None] = None,
    subset: str = AudioCapsCard.DEFAULT_SUBSET,
    force: bool = False,
    verbose: int = 0,
    verify_files: bool = False,
    # AudioCaps-specific args
    audio_duration: float = 10.0,
    audio_format: str = "flac",
    audio_n_channels: int = 1,
    download_audio: bool = True,
    ffmpeg_path: Union[str, Path, None] = None,
    max_workers: Optional[int] = None,
    sr: int = 32_000,
    ytdlp_path: Union[str, Path, None] = None,
    with_tags: bool = False,
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
    :param sr: The sample rate used for audio files in the dataset (in Hz).
        Since original YouTube videos are recorded in various settings, this parameter allow to download allow audio files with a specific sample rate.
        defaults to 32000.
    :param with_tags: If True, download the tags from AudioSet dataset.
        defaults to False.
    :param ytdlp_path: Path to yt-dlp or ytdlp executable.
        defaults to "yt-dlp".
    """

    root = _get_root(root)
    ytdlp_path = _get_ytdlp_path(ytdlp_path)
    ffmpeg_path = _get_ffmpeg_path(ffmpeg_path)

    if not osp.isdir(root):
        raise RuntimeError(f"Cannot find root directory '{root}'.")

    _check_subprog_help(ytdlp_path, "ytdlp")
    _check_subprog_help(ffmpeg_path, "ffmpeg")

    if _is_prepared_audiocaps(root, subset, sr, -1) and not force:
        return None

    audiocaps_root = _get_audiocaps_root(root, sr)
    os.makedirs(audiocaps_root, exist_ok=True)
    if with_tags:
        download_class_labels_indices(root)

    links = _AUDIOCAPS_LINKS[subset]
    audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr)
    os.makedirs(audio_subset_dpath, exist_ok=True)

    captions_fname = links["captions"]["fname"]
    captions_fpath = osp.join(audiocaps_root, captions_fname)

    if not osp.isfile(captions_fpath):
        url = links["captions"]["url"]
        if url is None:
            raise ValueError(
                f"AudioCaps subset '{subset}' cannot be automatically downloaded. (found url={url})"
            )
        download_file(url, captions_fpath, verbose=verbose)

    if download_audio:
        start = time.perf_counter()
        with open(captions_fpath, "r") as file:
            n_samples = len(file.readlines())

        if verbose >= 1:
            pylog.info(f"Start downloading audio files for {subset} AudioCaps split.")

        with open(captions_fpath, "r") as file:
            # Download audio files
            reader = csv.DictReader(file)
            captions_data = list(reader)
            # Keys: audiocap_id, youtube_id, start_time, caption

        present_audio_fnames = os.listdir(audio_subset_dpath)
        present_audio_fpaths = [
            osp.join(audio_subset_dpath, fname) for fname in present_audio_fnames
        ]
        present_audio_fpaths = dict.fromkeys(present_audio_fpaths)

        common_kwds = dict(
            audio_subset_dpath=audio_subset_dpath,
            audio_format=audio_format,
            verify_files=verify_files,
            present_audio_fpaths=present_audio_fpaths,
            audio_duration=audio_duration,
            sr=sr,
            audio_n_channels=audio_n_channels,
            ffmpeg_path=ffmpeg_path,
            ytdlp_path=ytdlp_path,
            verbose=verbose,
        )
        download_kwds = {
            f"{line['youtube_id']}-{line['start_time']}": {
                k: line[k] for k in ("youtube_id", "start_time")
            }
            for line in captions_data
        }
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            submitted_dict = {
                i: executor.submit(
                    _download_from_youtube_and_verify,
                    **kwds,
                    **common_kwds,
                )
                for i, kwds in enumerate(download_kwds.values())
            }
            for i, submitted in tqdm.tqdm(submitted_dict.items(), disable=verbose < 1):
                file_exists, download_success, valid_file = submitted.result()

                if verbose < 2:
                    continue

                line = captions_data[i]
                audiocap_id = line["audiocap_id"]
                youtube_id = line["youtube_id"]
                prefix = f"[{audiocap_id:6s};{i:5d}/{n_samples}] "

                if not file_exists:
                    if download_success:
                        if valid_file:
                            pylog.debug(
                                f"{prefix}File '{youtube_id}' has been downloaded and verified."
                            )
                        else:
                            pylog.debug(
                                f"{prefix}File '{youtube_id}' has been downloaded but it was not valid and has been removed."
                            )
                    else:
                        pylog.debug(
                            f"{prefix}Cannot extract audio '{youtube_id}'. (maybe the source video has been removed?)"
                        )
                else:
                    if valid_file:
                        pylog.info(
                            f"{prefix}File '{youtube_id}' is already downloaded and has been verified."
                        )
                    elif verify_files:
                        pylog.debug(
                            f"{prefix}File '{youtube_id}' is already downloaded but it was not valid and has been removed."
                        )
                    else:
                        pylog.debug(
                            f"{prefix}File '{youtube_id}' is already downloaded but it is not verified due to verify_files={verify_files}."
                        )

        if verbose >= 1:
            duration = int(time.perf_counter() - start)
            pylog.info(
                f"Download and preparation of AudioCaps for subset '{subset}' done in {duration}s."
            )
            pylog.info(f"- {n_samples} total samples.")

    if verbose >= 2:
        pylog.debug(
            f"Dataset {AudioCapsCard.PRETTY_NAME} (subset={subset}) has been prepared."
        )


def download_audiocaps_datasets(
    # Common args
    root: Union[str, Path, None] = None,
    subsets: Union[str, Iterable[str]] = AudioCapsCard.DEFAULT_SUBSET,
    force: bool = False,
    verbose: int = 0,
    # AudioCaps-specific args
    audio_duration: float = 10.0,
    audio_format: str = "flac",
    audio_n_channels: int = 1,
    download_audio: bool = True,
    ffmpeg_path: Union[str, Path, None] = None,
    sr: int = 32_000,
    verify_files: bool = False,
    ytdlp_path: Union[str, Path, None] = None,
    with_tags: bool = False,
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
        audio_duration=audio_duration,
        audio_format=audio_format,
        audio_n_channels=audio_n_channels,
        download_audio=download_audio,
        ffmpeg_path=ffmpeg_path,
        sr=sr,
        verify_files=verify_files,
        ytdlp_path=ytdlp_path,
        with_tags=with_tags,
    )
    for subset in subsets:
        download_audiocaps_dataset(
            subset=subset,
            **kwargs,
        )


def load_class_labels_indices(
    root: Union[str, Path, None] = None,
    sr: int = 32_000,
) -> List[Dict[str, str]]:
    root = _get_root(root)
    class_labels_indices_fname = _AUDIOSET_LINKS["class_labels_indices"]["fname"]
    class_labels_indices_fpath = osp.join(
        _get_audiocaps_root(root, sr), class_labels_indices_fname
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


def download_class_labels_indices(
    root: Union[str, Path, None] = None,
    sr: int = 32_000,
    verbose: int = 0,
) -> None:
    root = _get_root(root)
    audiocaps_root = _get_audiocaps_root(root, sr)

    for key in ("class_labels_indices", "unbalanced"):
        infos = _AUDIOSET_LINKS[key]
        url = infos["url"]
        fname = infos["fname"]
        fpath = osp.join(audiocaps_root, fname)
        if not osp.isfile(fpath):
            if verbose >= 1:
                pylog.info(f"Downloading file '{fname}'...")
            download_file(url, fpath, verbose=verbose)


def _get_audiocaps_root(root: str, sr: int) -> str:
    return osp.join(root, "AUDIOCAPS")


def _get_audio_subset_dpath(root: str, subset: str, sr: int) -> str:
    return osp.join(
        _get_audiocaps_root(root, sr),
        f"audio_{sr}Hz",
        _AUDIOCAPS_AUDIO_DNAMES[subset],
    )


def _is_prepared_audiocaps(
    root: str,
    subset: str = AudioCapsCard.DEFAULT_SUBSET,
    sr: int = 32_000,
    audio_format: str = "flac",
    verbose: int = 0,
) -> bool:
    links = _AUDIOCAPS_LINKS[subset]
    captions_fname = links["captions"]["fname"]
    captions_fpath = osp.join(_get_audiocaps_root(root, sr), captions_fname)
    audio_subset_dpath = _get_audio_subset_dpath(root, subset, sr)

    msgs = []

    if not osp.isdir(audio_subset_dpath):
        msgs.append(f"Cannot find directory '{audio_subset_dpath}'.")
    else:
        audio_fnames = os.listdir(audio_subset_dpath)
        audio_fnames = [
            fname for fname in audio_fnames if fname.endswith(f".{audio_format}")
        ]
        if len(audio_fnames) == 0:
            msgs.append(f"Cannot find any audio file in '{audio_subset_dpath}'.")

    if not osp.isfile(captions_fpath):
        msgs.append(f"Cannot find file '{captions_fpath}'.")

    if verbose >= 0:
        for msg in msgs:
            pylog.warning(msg)

    return len(msgs) == 0


def _download_from_youtube_and_verify(
    youtube_id: str,
    start_time: str,
    audio_subset_dpath: str,
    audio_format: str,
    verify_files: bool,
    present_audio_fpaths: Dict[str, None],
    audio_duration: float,
    sr: int,
    audio_n_channels: int,
    ffmpeg_path: str,
    ytdlp_path: str,
    verbose: int,
) -> Tuple[bool, bool, bool]:
    fname = f"{youtube_id}_{start_time}.{audio_format}"
    fpath = osp.join(audio_subset_dpath, fname)
    if not start_time.isdigit():
        raise RuntimeError(
            f'Start time "{start_time}" is not an integer youtube_id={youtube_id}).'
        )
    start_time = int(start_time)

    file_exists = fpath in present_audio_fpaths
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
        if download_success:
            valid_file = verify_files and _is_valid_audio_file(
                fpath, sr, audio_n_channels
            )
        else:
            valid_file = False

    elif verify_files:
        download_success = False
        valid_file = verify_files and _is_valid_audio_file(fpath, sr, audio_n_channels)
    else:
        download_success = False
        valid_file = False

    if verify_files and not valid_file:
        os.remove(fpath)

    return file_exists, download_success, valid_file


def _download_from_youtube(
    youtube_id: str,
    fpath_out: str,
    start_time: int,
    audio_duration: float = 10.0,
    sr: int = 16000,
    audio_n_channels: int = 1,
    target_format: str = "flac",
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
        exitcode = subprocess.check_call(extract_command)
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
    expected_sr: Optional[int],
    expected_n_channels: Optional[int],
) -> bool:
    try:
        metadata = torchaudio.info(fpath)  # type: ignore
    except RuntimeError:
        msg = f"Found file '{fpath}' already downloaded but it is invalid (cannot load metadata)."
        pylog.error(msg)
        return False

    if metadata.num_frames == 0:
        msg = (
            f"Found file '{fpath}' already downloaded but it is invalid (empty audio)."
        )
        pylog.error(msg)
        return False

    if expected_sr is not None and metadata.sample_rate != expected_sr:
        msg = f"Found file '{fpath}' already downloaded but it is invalid (invalid sr={metadata.sample_rate} != {expected_sr})."
        pylog.error(msg)
        return False

    if expected_n_channels is not None and metadata.num_channels != expected_n_channels:
        msg = f"Found file '{fpath}' already downloaded but it is invalid (invalid n_channels={metadata.num_channels} != {expected_sr})."
        pylog.error(msg)
        return False

    return True


def _get_youtube_link(youtube_id: str, start_time: Optional[int]) -> str:
    link = f"https://www.youtube.com/watch?v={youtube_id}"
    if start_time is None:
        return link
    else:
        return f"{link}&t={start_time}s"


def _get_youtube_link_embed(
    youtube_id: str,
    start_time: Optional[int],
    duration: float = 10.0,
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
