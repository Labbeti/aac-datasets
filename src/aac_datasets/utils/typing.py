#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    # To support torchaudio >= 2.1.0
    from torchaudio import AudioMetaData  # type: ignore
except ImportError:
    from torchaudio.backend.common import AudioMetaData


__all__ = ["AudioMetaData"]
