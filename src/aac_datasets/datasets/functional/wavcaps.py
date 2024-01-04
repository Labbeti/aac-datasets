#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
import logging
import os
import os.path as osp
import subprocess
import zipfile

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import tqdm

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils.tqdm import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
)
from typing_extensions import TypedDict

from aac_datasets.datasets.functional.common import DatasetCard
from aac_datasets.utils.collections import list_dict_to_dict_list
from aac_datasets.utils.download import safe_rmdir
from aac_datasets.utils.globals import _get_root, _get_zip_path


pylog = logging.getLogger(__name__)


class WavCapsCard(DatasetCard):
    ANNOTATIONS_CREATORS: Tuple[str, ...] = ("machine-generated",)
    CAPTIONS_PER_AUDIO: Dict[str, int] = {
        "as": 1,
        "bbc": 1,
        "fsd": 1,
        "sb": 1,
        "as_noac": 1,
        "fsd_nocl": 1,
    }
    CITATION: str = r"""
    @article{mei2023WavCaps,
        title        = {Wav{C}aps: A {ChatGPT}-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research},
        author       = {Xinhao Mei and Chutong Meng and Haohe Liu and Qiuqiang Kong and Tom Ko and Chengqi Zhao and Mark D. Plumbley and Yuexian Zou and Wenwu Wang},
        year         = 2023,
        journal      = {arXiv preprint arXiv:2303.17395},
        url          = {https://arxiv.org/pdf/2303.17395.pdf}
    }
    """
    DEFAULT_REVISION: str = "85a0c21e26fa7696a5a74ce54fada99a9b43c6de"
    DEFAULT_SUBSET: str = "as_noac"
    DESCRIPTION: str = "WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research."
    EXPECTED_SIZES: Dict[str, int] = {
        "AudioSet_SL": 108317,
        "BBC_Sound_Effects": 31201,
        "FreeSound": 262300,
        "SoundBible": 1320,  # note: 1232 according to github+hf, but found 1320 => seems that archive contains more data than in json
    }
    HOMEPAGE = "https://huggingface.co/datasets/cvssp/WavCaps"
    LANGUAGE: Tuple[str, ...] = ("en",)
    LANGUAGE_DETAILS: Tuple[str, ...] = ("en-US",)
    NAME: str = "wavcaps"
    PRETTY_NAME: str = "WavCaps"
    REPO_ID: str = "cvssp/WavCaps"
    SOURCES: Tuple[str, ...] = tuple(EXPECTED_SIZES.keys())
    SUBSETS: Tuple[str, ...] = tuple(CAPTIONS_PER_AUDIO.keys())
    SAMPLE_RATE: int = 32_000  # Hz
    SIZE_CATEGORIES: Tuple[str, ...] = ("100K<n<1M",)
    TASK_CATEGORIES: Tuple[str, ...] = ("audio-to-text", "text-to-audio")


def load_wavcaps_dataset(
    # Common args
    root: Union[str, Path, None] = None,
    subset: str = WavCapsCard.DEFAULT_SUBSET,
    verbose: int = 0,
    # WavCaps-specific args
    hf_cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> Dict[str, List[Any]]:
    """Load WavCaps metadata.

    :param root: Dataset root directory.
        defaults to ".".
    :param subset: The subset of MACS to use. Can be one of :attr:`~MACSCard.SUBSETS`.
        defaults to "as_noac".
    :param verbose: Verbose level.
        defaults to 0.

    :param hf_cache_dir: Optional override for HuggingFace cache directory path.
        defaults to None.
    :param revision: Optional override for revision commit/name for HuggingFace rapository.
        defaults to None.
    :returns: A dictionnary of lists containing each metadata.
    """
    root = _get_root(root)
    if subset not in WavCapsCard.SUBSETS:
        raise ValueError(
            f"Invalid argument subset={subset}. (expected one of {WavCapsCard.SUBSETS})"
        )

    if subset == "as":
        overlapped_ds = "AudioCaps"
        overlapped_subsets = ("val", "test")
        pylog.warning(
            f"You selected WavCaps subset '{subset}', be careful to not use these data as training when evaluating on {overlapped_ds} {overlapped_subsets} subsets. "
            "You can use as_noac subset for to avoid this bias with AudioCaps."
        )

    elif subset == "fsd":
        overlapped_ds = "Clotho"
        overlapped_subsets = (
            "val",
            "eval",
            "dcase_aac_test",
            "dcase_aac_analysis",
            "dcase_t2a_audio",
            "dcase_t2a_captions",
        )
        pylog.warning(
            f"You selected WavCaps subset '{subset}', be careful to not use these data as training when evaluating on {overlapped_ds} {overlapped_subsets} subsets. "
            f"You can use fsd_nocl subset for to avoid this bias for Clotho val, eval, dcase_t2a_audio and dcase_t2a_captions subsets. Data could still overlap with Clotho dcase_aac_test and dcase_aac_analysis subsets."
        )

    if subset in ("as_noac", "fsd_nocl"):
        if subset == "as_noac":
            target_subset = "as"
            csv_fname = "blacklist_audiocaps.full.csv"

        elif subset == "fsd_nocl":
            target_subset = "fsd"
            csv_fname = "blacklist_clotho.full.csv"

        else:
            raise ValueError(f"INTERNAL ERROR: Invalid argument subset={subset}.")

        raw_data = load_wavcaps_dataset(
            root=root,
            subset=target_subset,
            verbose=verbose,
            hf_cache_dir=hf_cache_dir,
            revision=revision,
        )
        wavcaps_ids = raw_data["id"]

        csv_fpath = (
            Path(__file__)
            .parent.parent.parent.parent.joinpath("data")
            .joinpath(csv_fname)
        )
        with open(csv_fpath, "r") as file:
            reader = csv.DictReader(file)
            data = list(reader)
        other_ids = [data_i["id"] for data_i in data]
        other_ids = dict.fromkeys(other_ids)

        indexes = [i for i, wc_id in enumerate(wavcaps_ids) if wc_id not in other_ids]

        if verbose >= 1:
            pylog.info(
                f"Getting {len(indexes)}/{len(wavcaps_ids)} items from '{target_subset}' for subset '{subset}'."
            )

        raw_data = {
            column: [column_data[idx] for idx in indexes]
            for column, column_data in raw_data.items()
        }
        return raw_data

    if not _is_prepared_wavcaps(root, hf_cache_dir, revision, subset, verbose):
        raise RuntimeError(
            f"{WavCapsCard.PRETTY_NAME} is not prepared in root={root}. Please use download=True to install it in root."
        )

    json_dpath = _get_json_dpath(root, hf_cache_dir, revision)
    json_paths = [
        ("AudioSet_SL", osp.join(json_dpath, "AudioSet_SL", "as_final.json")),
        (
            "BBC_Sound_Effects",
            osp.join(json_dpath, "BBC_Sound_Effects", "bbc_final.json"),
        ),
        ("FreeSound", osp.join(json_dpath, "FreeSound", "fsd_final.json")),
        ("SoundBible", osp.join(json_dpath, "SoundBible", "sb_final.json")),
    ]
    json_paths = [
        (source, json_path)
        for source, json_path in json_paths
        if _use_source(source, subset)
    ]

    raw_data = {k: [] for k in _WAVCAPS_RAW_COLUMNS + ("source", "fname")}
    for source, json_path in json_paths:
        if verbose >= 2:
            pylog.debug(f"Loading metadata in JSON '{json_path}'...")
        json_data, size = _load_json(json_path)

        sources = [source] * size
        json_data.pop("audio", None)

        if source == "AudioSet_SL":
            ids = json_data["id"]
            fnames = [id_.replace(".wav", ".flac") for id_ in ids]
            raw_data["fname"] += fnames

        elif source == "BBC_Sound_Effects":
            ids = json_data["id"]
            fnames = [f"{id_}.flac" for id_ in ids]
            raw_data["fname"] += fnames

        elif source == "FreeSound":
            ids = json_data["id"]
            fnames = [f"{id_}.flac" for id_ in ids]
            raw_data["fname"] += fnames

        elif source == "SoundBible":
            ids = json_data["id"]
            fnames = [f"{id_}.flac" for id_ in ids]
            raw_data["fname"] += fnames

        else:
            raise RuntimeError(f"Invalid source={source}.")

        for k in _WAVCAPS_RAW_COLUMNS:
            if k in json_data:
                raw_data[k] += json_data[k]
            elif k in _DEFAULT_VALUES:
                default_val = _DEFAULT_VALUES[k]
                default_values = [default_val] * size
                raw_data[k] += default_values
            elif k in ("audio", "file_name"):
                pass
            else:
                raise RuntimeError(f"Invalid column name {k}. (with source={source})")

        raw_data["source"] += sources

    raw_data.pop("audio")
    raw_data.pop("file_name")
    captions = raw_data.pop("caption")

    # Convert str -> List[str] for captions to match other datasets captions type
    raw_data["captions"] = [[caption] for caption in captions]

    # Force floating-point precision for duration
    raw_data["duration"] = list(map(float, raw_data["duration"]))

    return raw_data


def download_wavcaps_dataset(
    # Common args
    root: Union[str, Path, None] = None,
    subset: str = WavCapsCard.DEFAULT_SUBSET,
    force: bool = False,
    verbose: int = 0,
    verify_files: bool = False,
    # WavCaps-specific args
    clean_archives: bool = False,
    hf_cache_dir: Optional[str] = None,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    zip_path: Union[str, Path, None] = None,
) -> None:
    """Prepare WavCaps data.

    :param root: Dataset root directory.
        defaults to ".".
    :param subset: The subset of MACS to use. Can be one of :attr:`~WavCapsCard.SUBSETS`.
        defaults to "as_noac".
    :param force: If True, force to download again all files.
        defaults to False.
    :param verbose: Verbose level.
        defaults to 0.
    :param verify_files: If True, check all file already downloaded are valid.
        defaults to False.

    :param clean_archives: If True, remove the compressed archives from disk to save space.
        defaults to True.
    :param hf_cache_dir: Optional override for HuggingFace cache directory path.
        defaults to None.
    :param repo_id: Repository ID on HuggingFace.
        defaults to "cvssp/WavCaps".
    :param revision: Optional override for revision commit/name for HuggingFace rapository.
        defaults to None.
    :param zip_path: Path to zip executable path in shell.
        defaults to "zip".
    """
    if subset == "as_noac":
        return download_wavcaps_dataset(
            root=root,
            subset="as",
            revision=revision,
            hf_cache_dir=hf_cache_dir,
            force=force,
            verify_files=verify_files,
            clean_archives=clean_archives,
            zip_path=zip_path,
            verbose=verbose,
        )
    elif subset == "fsd_nocl":
        return download_wavcaps_dataset(
            root=root,
            subset="fsd",
            revision=revision,
            hf_cache_dir=hf_cache_dir,
            force=force,
            verify_files=verify_files,
            clean_archives=clean_archives,
            zip_path=zip_path,
            verbose=verbose,
        )

    root = _get_root(root)
    zip_path = _get_zip_path(zip_path)

    if subset not in WavCapsCard.SUBSETS:
        raise ValueError(
            f"Invalid argument subset={subset}. (expected one of {WavCapsCard.SUBSETS})"
        )

    # note: verbose=-1 to disable warning triggered when dset is not prepared
    if not force and _is_prepared_wavcaps(
        root, hf_cache_dir, revision, subset, verbose=-1
    ):
        return None

    if hf_cache_dir is None:
        hf_cache_dir = HUGGINGFACE_HUB_CACHE
    if repo_id is None:
        repo_id = WavCapsCard.REPO_ID

    # Download files from huggingface
    ign_sources = [
        source for source in WavCapsCard.SOURCES if not _use_source(source, subset)
    ]
    ign_patterns = [
        pattern
        for source in ign_sources
        for pattern in (f"json_files/{source}/*.json", "Zip_files/*")  # {source}/
    ]
    if verbose >= 2:
        pylog.debug(f"ign_sources={ign_sources}")
        pylog.debug(f"ign_patterns={ign_patterns}")

    pbar_enabled = are_progress_bars_disabled()
    if pbar_enabled and verbose <= 0:
        disable_progress_bars()

    snapshot_dpath = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        resume_download=not force,
        local_files_only=False,
        cache_dir=hf_cache_dir,
        allow_patterns=None,
        ignore_patterns=ign_patterns,
    )

    if pbar_enabled and verbose <= 0:
        enable_progress_bars()

    snapshot_abs_dpath = osp.abspath(snapshot_dpath)
    wavcaps_root = _get_wavcaps_root(root, hf_cache_dir, revision)
    if verbose >= 2:
        pylog.debug(f"snapshot_dpath={snapshot_dpath}")
        pylog.debug(f"snapshot_absdpath={snapshot_abs_dpath}")
        pylog.debug(f"wavcaps_dpath={wavcaps_root}")
    del snapshot_dpath

    # Build symlink to hf cache
    if osp.exists(wavcaps_root):
        if not osp.islink(wavcaps_root):
            raise RuntimeError("WavCaps root exists but it is not a symlink.")
        link_target_abspath = osp.abspath(osp.realpath(wavcaps_root))
        if link_target_abspath != snapshot_abs_dpath:
            pylog.error(
                "Target link is not pointing to current snapshot path. It will be automatically replaced."
            )
            os.remove(wavcaps_root)
            os.symlink(snapshot_abs_dpath, wavcaps_root, True)
    else:
        os.symlink(snapshot_abs_dpath, wavcaps_root, True)

    source_and_splitted = [
        ("AudioSet_SL", True),
        ("BBC_Sound_Effects", True),
        ("FreeSound", True),
        ("SoundBible", False),
    ]
    source_and_splitted = {
        source: is_splitted
        for source, is_splitted in source_and_splitted
        if _use_source(source, subset)
    }

    archives_dpath = _get_archives_dpath(root, hf_cache_dir, revision)
    for source, is_splitted in source_and_splitted.items():
        main_zip_fpath = osp.join(
            archives_dpath, _WAVCAPS_ARCHIVE_DNAMES[source], f"{source}.zip"
        )

        if is_splitted:
            merged_zip_fpath = osp.join(
                archives_dpath, _WAVCAPS_ARCHIVE_DNAMES[source], f"{source}_merged.zip"
            )
        else:
            merged_zip_fpath = main_zip_fpath

        if is_splitted and not osp.isfile(merged_zip_fpath):
            cmd = [
                zip_path,
                "-FF",
                main_zip_fpath,
                "--out",
                merged_zip_fpath,
            ]
            if verbose >= 2:
                pylog.debug(f"Merging ZIP files for {source}...")
                pylog.debug(f"Using command: {' '.join(cmd)}")

            if verbose >= 2:
                stdout = None
                stderr = None
            else:
                stdout = subprocess.DEVNULL
                stderr = subprocess.DEVNULL

            subprocess.check_call(cmd, stdout=stdout, stderr=stderr)

        audio_subset_dpath = _get_audio_subset_dpath(
            root, hf_cache_dir, revision, source
        )
        os.makedirs(audio_subset_dpath, exist_ok=True)

        with zipfile.ZipFile(merged_zip_fpath, "r") as file:
            flac_subnames = [name for name in file.namelist() if name.endswith(".flac")]
            assert len(flac_subnames) > 0
            assert all(
                osp.dirname(name) == osp.dirname(flac_subnames[0])
                for name in flac_subnames
            )

            src_root = osp.join(audio_subset_dpath, osp.dirname(flac_subnames[0]))
            src_fnames_found = (
                dict.fromkeys(name for name in os.listdir(src_root))
                if osp.isdir(src_root)
                else {}
            )
            tgt_fnames_found = dict.fromkeys(
                name for name in os.listdir(audio_subset_dpath)
            )

            missing_subnames = [
                subname
                for subname in flac_subnames
                if osp.basename(subname) not in src_fnames_found
                and osp.basename(subname) not in tgt_fnames_found
            ]
            if verbose >= 2:
                pylog.debug(
                    f"Extracting {len(missing_subnames)}/{len(flac_subnames)} audio files from {merged_zip_fpath}..."
                )
            file.extractall(audio_subset_dpath, missing_subnames)
            if verbose >= 2:
                pylog.debug("Extraction done.")

        src_fnames_found = (
            dict.fromkeys(name for name in os.listdir(src_root))
            if osp.isdir(src_root)
            else {}
        )
        src_fpaths_to_move = [
            osp.join(audio_subset_dpath, subname)
            for subname in flac_subnames
            if osp.basename(subname) in src_fnames_found
        ]
        if verbose >= 2:
            pylog.debug(f"Moving {len(src_fpaths_to_move)} files...")
        for src_fpath in tqdm.tqdm(src_fpaths_to_move):
            tgt_fpath = osp.join(audio_subset_dpath, osp.basename(src_fpath))
            os.rename(src_fpath, tgt_fpath)
        if verbose >= 2:
            pylog.debug("Move done.")

        if verify_files:
            tgt_fnames_expected = [osp.basename(subname) for subname in flac_subnames]
            tgt_fnames_found = dict.fromkeys(
                fname for fname in os.listdir(audio_subset_dpath)
            )
            if verbose >= 2:
                pylog.debug(f"Checking {len(tgt_fnames_expected)} files...")
            tgt_fnames_invalids = [
                fname for fname in tgt_fnames_expected if fname not in tgt_fnames_found
            ]
            if len(tgt_fnames_invalids) > 0:
                raise FileNotFoundError(
                    f"Found {len(tgt_fnames_invalids)}/{len(tgt_fnames_expected)} invalid files."
                )

        safe_rmdir(audio_subset_dpath, rm_root=False, error_on_non_empty_dir=True)

    if clean_archives:
        used_sources = source_and_splitted.keys()
        for source in used_sources:
            archive_source_dpath = osp.join(
                archives_dpath, _WAVCAPS_ARCHIVE_DNAMES[source]
            )
            archives_names = os.listdir(archive_source_dpath)
            for name in archives_names:
                if not name.endswith(".zip") and ".z" not in name:
                    continue
                fpath = osp.join(archive_source_dpath, name)
                if verbose >= 1:
                    pylog.info(f"Removing archive file {name} for source={source}...")
                os.remove(fpath)


def download_wavcaps_datasets(
    # Common args
    root: Union[str, Path, None] = None,
    subsets: Union[str, Iterable[str]] = WavCapsCard.DEFAULT_SUBSET,
    force: bool = False,
    verbose: int = 0,
    # WavCaps-specific args
    clean_archives: bool = False,
    hf_cache_dir: Optional[str] = None,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    verify_files: bool = False,
    zip_path: Union[str, Path, None] = None,
) -> None:
    """Function helper to download a list of subsets. See :func:`~aac_datasets.datasets.functional.wavcaps.download_wavcaps_dataset` for details."""
    if isinstance(subsets, str):
        subsets = [subsets]
    else:
        subsets = list(subsets)

    kwargs: Dict[str, Any] = dict(
        root=root,
        force=force,
        verbose=verbose,
        clean_archives=clean_archives,
        hf_cache_dir=hf_cache_dir,
        repo_id=repo_id,
        revision=revision,
        verify_files=verify_files,
        zip_path=zip_path,
    )
    for subset in subsets:
        download_wavcaps_dataset(
            subset=subset,
            **kwargs,
        )


def _get_wavcaps_root(
    root: str,
    hf_cache_dir: Optional[str],
    revision: Optional[str],
) -> str:
    return osp.join(root, "WavCaps")


def _get_json_dpath(
    root: str,
    hf_cache_dir: Optional[str],
    revision: Optional[str],
) -> str:
    return osp.join(_get_wavcaps_root(root, hf_cache_dir, revision), "json_files")


def _get_archives_dpath(
    root: str,
    hf_cache_dir: Optional[str],
    revision: Optional[str],
) -> str:
    return osp.join(_get_wavcaps_root(root, hf_cache_dir, revision), "Zip_files")


def _get_audio_dpath(
    root: str,
    hf_cache_dir: Optional[str],
    revision: Optional[str],
) -> str:
    return osp.join(_get_wavcaps_root(root, hf_cache_dir, revision), "Audio")


def _get_audio_subset_dpath(
    root: str,
    hf_cache_dir: Optional[str],
    revision: Optional[str],
    source: str,
) -> str:
    return osp.join(
        _get_audio_dpath(root, hf_cache_dir, revision), _WAVCAPS_AUDIO_DNAMES[source]
    )


def _is_prepared_wavcaps(
    root: str,
    hf_cache_dir: Optional[str],
    revision: Optional[str],
    subset: str,
    verbose: int,
) -> bool:
    sources = [source for source in WavCapsCard.SOURCES if _use_source(source, subset)]
    for source in sources:
        audio_fnames = os.listdir(
            _get_audio_subset_dpath(root, hf_cache_dir, revision, source)
        )
        expected_size = WavCapsCard.EXPECTED_SIZES[source]
        if expected_size != len(audio_fnames):
            if verbose >= 0:
                pylog.error(
                    f"Invalid number of files for source={source}. (expected {expected_size} but found {len(audio_fnames)} files)"
                )
            return False
    return True


def _use_source(source: str, subset: str) -> bool:
    return any(
        (
            source == "AudioSet_SL" and subset in ("as", "as_noac"),
            source == "BBC_Sound_Effects" and subset in ("bbc",),
            source == "FreeSound" and subset in ("fsd", "fsd_nocl"),
            source == "SoundBible" and subset in ("sb",),
        )
    )


def _load_json(fpath: str) -> Tuple[Dict[str, Any], int]:
    with open(fpath, "r") as file:
        data = json.load(file)
    data = data["data"]
    size = len(data)
    data = list_dict_to_dict_list(data, key_mode="same")
    return data, size


class _WavCapsRawItem(TypedDict):
    # Common values
    caption: str
    duration: float
    id: str
    # Source Specific values
    audio: Optional[str]
    author: Optional[str]
    description: Optional[str]
    download_link: Optional[str]
    file_name: Optional[str]
    href: Optional[str]
    tags: Optional[List[str]]


_DEFAULT_VALUES = {
    "author": "",
    "description": "",
    "download_link": "",
    "href": "",
    "tags": [],
}

_WAVCAPS_RAW_COLUMNS = tuple(
    _WavCapsRawItem.__required_keys__ | _WavCapsRawItem.__optional_keys__
)

_WAVCAPS_AUDIO_DNAMES = {
    # Source name to audio directory name
    "AudioSet_SL": "AudioSet_SL",
    "BBC_Sound_Effects": "BBC_Sound_Effects",
    "FreeSound": "FreeSound",
    "SoundBible": "SoundBible",
}

_WAVCAPS_ARCHIVE_DNAMES = {
    # Source name to audio directory name
    "AudioSet_SL": "AudioSet_SL",
    "BBC_Sound_Effects": "BBC_Sound_Effects",
    "FreeSound": "FreeSound",
    "SoundBible": "SoundBible",
}
