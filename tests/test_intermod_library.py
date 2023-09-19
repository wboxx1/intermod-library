#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `intermod_library` package."""


import intermod_library.intermod_library as it


def test_intermod_table():
    signals = [1000, 2000]
    order = 3
    lower_limit = 0
    upper_limit = 100e3
    (
        freqs,
        tx_indexes,
        coeff_tuples,
        order_tuples
    ) = it.intermod_table(signals, order, lower_limit, upper_limit)
    assert freqs.sum() == 16000.0


def test_harmonic_toi():
    frqs = [1000]
    order = 5
    band_of_interest = [2500, 3500]
    table = it.harmonic_toi(frqs, order, band_of_interest)
    assert table["Signal 1"].sum() == "1000.02000.0**3000.04000.05000.0"


def test_harmonic_toi_no_boi():
    frqs = [1000]
    order = 5
    band_of_interest = []
    table = it.harmonic_toi(frqs, order, band_of_interest)
    assert table["Signal 1"].sum() == "1000.02000.03000.04000.05000.0"


def test_intermod_locate():
    soi = 2227.75
    pivot = 2196.0
    order = 3
    table = it.intermod_locate(soi, pivot, order)
    assert round(table.Frequency.sum() - 19551.4583, 4) == 0
