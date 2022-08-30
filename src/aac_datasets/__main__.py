#!/usr/bin/env python
# -*- coding: utf-8 -*-


def _print_usage() -> None:
    print(
        "Command line usage :\n"
        "- Download a dataset             : python -m aac_datasets.download [--root ROOT] [--verbose VERBOSE] [--force (false|true)] (clotho|audiocaps|macs) [ARGS...]\n"
        "- Check a installation directory : python -m aac_datasets.check [--root ROOT] [--verbose VERBOSE]\n"
        "- Print package version          : python -m aac_datasets.version\n"
        "- Show this usage page           : python -m aac_datasets\n"
    )


if __name__ == "__main__":
    _print_usage()
