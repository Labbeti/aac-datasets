#!/usr/bin/env python
# -*- coding: utf-8 -*-


def _print_usage() -> None:
    print(
        "Command line usage :\n"
        "- Download a dataset             : aacd-download [--root ROOT] [--verbose VERBOSE] [--force (false|true)] (clotho|audiocaps|macs) [ARGS...]\n"
        "- Check a installation directory : aacd-check [--root ROOT] [--verbose VERBOSE]\n"
        "- Print package version          : aacd-version\n"
        "- Show this usage page           : aacd\n"
    )


if __name__ == "__main__":
    _print_usage()
