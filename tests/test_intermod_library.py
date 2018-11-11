# -*- coding: utf-8 -*-

import intermod_library.intermod_tools as it

def test_intermod_table():
    signals = [1000, 2000]
    order = 3
    table = it.intermod_table(signals, order)
    assert table.Frequency.sum() == 34000.0