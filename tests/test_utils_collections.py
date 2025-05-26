#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from aac_datasets.utils.collections import intersect_lists


class TestCollections(TestCase):
    def test_intersect_lists(self) -> None:
        input_ = [["a", "b", "b", "c"], ["c", "d", "b", "a"], ["b", "a", "a", "e"]]
        expected = ["a", "b"]

        output = intersect_lists(input_)
        assert output == expected


if __name__ == "__main__":
    unittest.main()
