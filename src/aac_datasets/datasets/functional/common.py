#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datasets import DatasetInfo
from typing_extensions import TypedDict


class DatasetCard:
    def to_dataset_info(self) -> DatasetInfo:
        empty_info = DatasetInfo()
        info = {
            k.lower(): v
            for k, v in self.__dict__.items()
            if k.lower() in empty_info.__dict__
        }
        return DatasetInfo(**info)


class LinkInfo(TypedDict):
    fname: str
    url: str


class LinkInfoHash(TypedDict):
    fname: str
    url: str
    hash_value: str
