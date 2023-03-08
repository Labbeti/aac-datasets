#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os.path as osp

from typing import Iterable


DCASE_FIELDNAMES = ("file_name", "caption_predicted")


def export_to_dcase_csv_file(
    csv_fpath: str,
    audio_fnames: Iterable[str],
    candidates: Iterable[str],
    overwrite: bool = False,
) -> None:
    """Export results to DCASE csv submission format into a file.

    The CSV filename should be <author>_<institute>_task6a_submission_<submission_index>_<testing_output or analysis_output>.csv

    The rules are defined in https://dcase.community/challenge2023/task-automated-audio-captioning#submission

    :param csv_fpath: The path to the new CSV file.
    :param audio_fnames: The list of audio filenames.
    :param candidates: The captions predicted by your AAC system.
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
        {DCASE_FIELDNAMES[0]: fname, DCASE_FIELDNAMES[1]: cand}
        for fname, cand in zip(audio_fnames, candidates)
    ]
    with open(csv_fpath, "w") as file:
        writer = csv.DictWriter(file, fieldnames=DCASE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)  # type: ignore
