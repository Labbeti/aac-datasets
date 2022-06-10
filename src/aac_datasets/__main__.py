#!/usr/bin/env python
# -*- coding: utf-8 -*-


def print_usage() -> None:
    print(
        "Command line usage :\n"
        "- Download a dataset             : python -m aac_datasets.download [--root ROOT] [--verbose VERBOSE] [--force (false|true)] (clotho|audiocaps|macs) [ARGS...]\n"
        "- Check a installation directory : python -m aac_datasets.check [--root ROOT] [--verbose VERBOSE]\n"
        "- Print package version          : python -m aac_datasets.version\n"
    )


if __name__ == "__main__":
    print_usage()
