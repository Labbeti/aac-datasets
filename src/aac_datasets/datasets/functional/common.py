#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing_extensions import TypedDict


class DatasetCard:
    pass


class LinkInfo(TypedDict):
    fname: str
    url: str


class LinkInfoHash(TypedDict):
    fname: str
    url: str
    hash_value: str
