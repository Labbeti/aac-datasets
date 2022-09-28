#!/usr/bin/env python
# -*- coding: utf-8 -*-

import platform
import sys

from typing import Dict

import torch
import torchaudio
import yaml

import aac_datasets


def get_install_info() -> Dict[str, str]:
    """Return a dictionary containing the version python, the os name, the architecture name and the versions of the following packages: aac_datasets, torch, torchaudio."""
    return {
        "aac_datasets": aac_datasets.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": platform.system(),
        "architecture": platform.architecture()[0],
        "torch": str(torch.__version__),
        "torchaudio": torchaudio.__version__,
    }


def print_install_info() -> None:
    """Print packages versions and architecture info."""
    install_info = get_install_info()
    print(yaml.dump(install_info, sort_keys=False))


if __name__ == "__main__":
    print_install_info()
