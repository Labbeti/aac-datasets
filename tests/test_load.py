#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from aac_datasets.datasets import Clotho


class TestLoad(TestCase):
    def test_load_clotho(self) -> None:
        ds = Clotho(subset="val")
        item = ds[0]
        assert isinstance(item, dict)


if __name__ == "__main__":
    unittest.main()
