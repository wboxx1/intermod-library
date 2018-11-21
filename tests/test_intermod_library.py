#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `intermod_library` package."""

import pytest


import intermod_library.intermod_library as it


def test_intermod_table():
    signals = [1000, 2000]
    order = 3
    table = it.intermod_table(signals, order)
    assert table.Frequency.sum() == 34000.0
