#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os.path as osp

from typing import Iterable


DCASE_TASK6A_FIELDNAMES = ("file_name", "caption_predicted")
DCASE_TASK6B_TOP_N = 10
DCASE_TASK6B_FIELDNAMES = ("caption",) + tuple(
    f"fname_{i}" for i in range(1, DCASE_TASK6B_TOP_N + 1)
)


def export_to_dcase_task6a_csv(
    csv_fpath: str,
    audio_fnames: Iterable[str],
    candidates: Iterable[str],
    overwrite: bool = False,
) -> None:
    """Export results to DCASE task6a CSV submission file.

    The CSV filename should be <author>_<institute>_task6a_submission_<submission_index>_<testing_output or analysis_output>.csv

    The rules are defined in https://dcase.community/challenge2023/task-automated-audio-captioning#submission

    :param csv_fpath: The path to the new CSV file.
    :param audio_fnames: The ordered list of audio filenames.
    :param candidates: The ordered captions predicted by your AAC system corresponding to the audio filenames.
    :param overwrite:
        If the CSV already exists and overwrite is True, the function will replace it.
        If the file already exists and overwrite is False, raises a FileExistsError.
        It has no effect otherwise.
        defaults to False.
    """

    audio_fnames = list(audio_fnames)
    candidates = list(candidates)

    if not overwrite and osp.isfile(csv_fpath):
        raise FileExistsError(
            f"DCASE submission file {csv_fpath=} already exists. Please delete it or use argument overwrite=True."
        )
    if len(audio_fnames) != len(candidates):
        raise ValueError(
            f"Invalid lengths for arguments audio_fnames and candidates. (found {len(audio_fnames)=} != {len(candidates)=})"
        )

    rows = [
        {DCASE_TASK6A_FIELDNAMES[0]: fname, DCASE_TASK6A_FIELDNAMES[1]: cand}
        for fname, cand in zip(audio_fnames, candidates)
    ]
    with open(csv_fpath, "w") as file:
        writer = csv.DictWriter(file, fieldnames=DCASE_TASK6A_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)  # type: ignore


def export_to_dcase_task6b_csv(
    csv_fpath: str,
    query_captions: Iterable[str],
    predicted_fnames: Iterable[Iterable[str]],
    overwrite: bool = False,
) -> None:
    """Export results to DCASE task6b CSV submission file.

    The CSV filename should be <author>_<institute>_task6b_submission_<submission_index>_output.csv

    The rules are defined in https://dcase.community/challenge2023/task-language-based-audio-retrieval#submission

    :param csv_fpath: The path to the new CSV file.
    :param query_captions: The ordered list of queries.
    :param predicted_fnames: The ordered list of top-10 filenames corresponding to the queries.
    :param overwrite:
        If the CSV already exists and overwrite is True, the function will replace it.
        If the file already exists and overwrite is False, raises a FileExistsError.
        It has no effect otherwise.
        defaults to False.
    """

    query_captions = list(query_captions)
    predicted_fnames = [list(fnames) for fnames in predicted_fnames]

    if not overwrite and osp.isfile(csv_fpath):
        raise FileExistsError(
            f"DCASE submission file {csv_fpath=} already exists. Please delete it or use argument overwrite=True."
        )
    if len(query_captions) != len(predicted_fnames):
        raise ValueError(
            f"Invalid lengths for arguments audio_fnames and candidates. (found {len(query_captions)=} != {len(predicted_fnames)=})"
        )

    invalid_lens = [
        query
        for query, fnames in zip(query_captions, predicted_fnames)
        if len(fnames) != DCASE_TASK6B_TOP_N
    ]
    if len(invalid_lens) > 0:
        raise ValueError(
            f"Invalid number of relevant audio filenames. (found {invalid_lens=} but expected only {DCASE_TASK6B_TOP_N} files per query)"
        )

    rows = [
        {DCASE_TASK6A_FIELDNAMES[0]: query}
        | dict(zip(DCASE_TASK6B_FIELDNAMES[1:], fnames))
        for query, fnames in zip(query_captions, predicted_fnames)
    ]
    with open(csv_fpath, "w") as file:
        writer = csv.DictWriter(file, fieldnames=DCASE_TASK6B_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)  # type: ignore
