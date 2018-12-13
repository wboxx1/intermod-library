#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `intermod_library` package."""


import intermod_library.intermod_tools as it


def test_intermod_table():
    signals = [1000, 2000]
    order = 3
    table = it.intermod_table(signals, order)
    assert table.Frequency.sum() == 34000.0


def test_harmonic_toi():
    frqs = [1000]
    order = 5
    band_of_interest = [2500, 3500]
    table = it.harmonic_toi(frqs, order, band_of_interest)
    assert table["Signal 1"].sum() == "1000.02000.0**3000.04000.05000.0"
