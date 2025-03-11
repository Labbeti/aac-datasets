#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from aac_datasets.datasets.base import _flat_raw_data, _unflat_raw_data


class TestDatasetBase(TestCase):
    def test_flat_raw_data(self) -> None:
        raw_data = {
            "captions": [["a1", "a2", "a3"], ["b1"], ["c1", "c2"], []],
            "index": list(range(1, 5)),
        }
        expected_flat = {
            "captions": [["a1"], ["a2"], ["a3"], ["b1"], ["c1"], ["c2"], []],
            "index": [1, 1, 1, 2, 3, 3, 4],
        }
        expected_sizes = [3, 1, 2, 0]

        raw_data_flat_out, sizes_out = _flat_raw_data(raw_data, "captions")
        raw_data_out = _unflat_raw_data(raw_data_flat_out, sizes_out)

        assert raw_data_flat_out == expected_flat
        assert sizes_out == expected_sizes
        assert raw_data_out == raw_data


if __name__ == "__main__":
    unittest.main()
