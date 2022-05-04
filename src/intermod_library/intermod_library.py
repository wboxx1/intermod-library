# -*- coding: utf-8 -*-

"""
Intermod Tools

@author: William Boxx
"""


import numpy as np
import pandas as pd
import itertools
# import helpers.helper_functions


def intermod_table(signals, order, maximum_single_order=None, bandwidths=None):
    """Calculates intermodulation products between the given signals

    Will calculate all intermodulation products that could be potentially
    created between the given signals.  Must specify the highest order of
    intermods.

    :param signals: list of signals to calculate intermod products on
    :type signals: list[float]
    :param order: highest order of intermod products to calculate
    :type order: integer
    :param maximum_single_order: the maximum order for a single frequency
    :type maximum_single_order: integer
    :param bandwidths: bandwidths of the provided signals.
    :type bandwidths: list[float]
    :returns: pandas dataframe containing the calculated intermod products
    :Example:

    >>> import intermod_library.intermod_library as il
    >>> signals = [1000, 2000]
    >>> order = 3
    >>> table = il.intermod_table(signals, order)
    >>> table.head()

    ========= ========= ========= ========= =========
    <index>   Frequency Signal 1  Signal 2  Order
    ========= ========= ========= ========= =========
    0         1000.0    1.0       0.0       1.0
    1         1000.0    -1.0      1.0       2.0
    2         2000.0    0.0       1.0       1.0
    3         2000.0    2.0       0.0       2.0
    4         3000.0    1.0       1.0       2.0
    ========= ========= ========= ========= =========
    """

    M = np.size(signals)   # Number of signals
    # N = order + 1

    # A = np.arange(0, N)

    # coefmat = np.zeros([N**M, M])

    # ind = 0

    # for i in range(M, 0, -1):
    #     m = N**(M-i)
    #     B = np.ones([N**(i-1), m])
    #     C = np.kron(B, A)
    #     coefmat[:, ind] = C.T.ravel()
    #     ind += 1

    # B = np.ones(2**M)
    # coefmat = np.reshape(np.kron(B, coefmat), [-1, M])

    # # Make sign array
    # A = np.array([1, -1])
    # signmat = np.zeros([2**M, M])
    # ind = 0

    # for i in range(M, 0, -1):
    #     m = 2**(M-i)
    #     B = np.ones([2**(i - 1), m])
    #     C = np.kron(B, A)
    #     signmat[:, ind] = C.T.ravel()
    #     ind += 1

    # j = np.size(coefmat)/M
    # firstblock = signmat

    # for i in range(1, int(j / 2**M)):
    #     signmat = np.vstack([signmat, firstblock])

    # finalmat = coefmat * signmat

    if maximum_single_order is None:
        mso = order
    else:
        mso = maximum_single_order

    finalmat = np.array(
        list(
            itertools.product(range(-1*mso, mso + 1, 1), repeat=M)
        )
    )

    intermods = np.dot(finalmat, signals)
    intermod_order = np.sum(abs(finalmat), 1)
    final = np.column_stack((intermods, finalmat, intermod_order))

    header = ['Frequency']
    for i in np.arange(M):
        header = header + ['Signal ' + str(i+1)]

    header = header + ['Order']

    T = pd.DataFrame(final, columns=header)

    T.query('Frequency > 0', inplace=True)
    T.query('(Order > 0) & (Order <= @order)', inplace=True)
    T.drop_duplicates(inplace=True)
    T.sort_values(by=['Frequency'], inplace=True)
    T.reset_index(drop=True, inplace=True)

    def find_bw(row, bws):
        return max(row[1:-1]*bws)

    if bandwidths is not None:
        T['bandwidth'] = T.apply(find_bw, 1, True, None, ([bandwidths]))

    return (T)


def harmonic_toi(frqs, order, band_of_interest=[]):
    """Calculates the harmonic table of interest for given frequencies.

    Will calculate the harmonics of the given frequencies and highlight
    which ones fall within the given band of interest.

    :param frqs: list of frequencies
    :param order: largest order of harmonic
    :param band_of_interest: tuple containing the lower and upper
        values of the band of interest
    :type frqs: float
    :type order: integer
    :type band_of_interest: tuple(float)
    :returns: pandas dataframe that lists harmonics
        and emphasizes those in boi with **
    :Example:

    >>> import intermod_library.intermod_library as il
    >>> frqs = [1000]
    >>> order = 5
    >>> band_of_interest = [2500, 3500]
    >>> table = il.harmonic_toi(frqs, order, band_of_interest)
    >>> table

    =========== =========
    <index>     Signal 1
    =========== =========
    Harmonic #1   1000.0
    Harmonic #2   2000.0
    Harmonic #3 \\*\\*3000.0
    Harmonic #4   4000.0
    Harmonic #5   5000.0
    =========== =========
    """

    if (len(band_of_interest) == 0) | (len(band_of_interest) == 1):
        lower = 1.0
        upper = -1.0
    else:
        lower = band_of_interest[0]
        upper = band_of_interest[1]

    n = np.size(frqs)
    m = order
    table = np.zeros((m, n))

    for i in np.arange(n):
        table[0, i] = frqs[i]
        table[1:, i] = [frqs[i]*x for x in np.arange(2, order+1)]

    index = ["Harmonic #" + str(x) for x in range(1, order+1)]
    header = ["Signal " + str(x) for x in range(1, n+1)]

    T = pd.DataFrame(table, columns=header, index=index)

    for i in np.arange(T.shape[0]):
        for j in np.arange(T.shape[1]):
            if (T.iloc[i, j] >= lower and T.iloc[i, j] <= upper):
                T.iloc[i, j] = "**" + str(T.iloc[i, j])
            else:
                T.iloc[i, j] = str(T.iloc[i, j])

    return T


def intermod_locate(soi, pivot, order):
    """Calculates a list of frequencies that would create an intermod at soi.

    Will calculate a list of frequencies that could combine with the pivot
    to create intermods at the signal of interest (soi).

    :param soi: signal of interest
    :param pivot: pivot frequency
    :param order: largest order of harmonic
    :type soi: float
    :type pivot: float
    :type order: integer
    :returns: pandas dataframe that lists possible frequencies and order
    :Example:

    >>> import intermod_library.intermod_library as il
    >>> soi = 2227.75
    >>> pivot = 2196.0
    >>> order = 3
    >>> table = il.intermod_locate(soi, pivot, order)
    >>> table

    ========= ========= ========= ========= =========
    <index>   Frequency Signal 1  Signal 2  Order
    ========= ========= ========= ========= =========
    0           15.875    1.0       2.0       3.0
    1           31.750    1.0       1.0       2.0
    2          742.583    0.0       3.0       3.0
    3         1113.875    0.0       2.0       2.0
    4         2164.250    2.0      -1.0       3.0
    ========= ========= ========= ========= =========
    """

    M = 2   # Number of signals
    # N = order + 1

    # A = np.arange(0, N)

    # coefmat = np.zeros([N**M, M])

    # ind = 0

    # for i in range(M, 0, -1):
    #     m = N**(M-i)
    #     B = np.ones([N**(i-1), m])
    #     C = np.kron(B, A)
    #     coefmat[:, ind] = C.T.ravel()
    #     ind += 1

    # B = np.ones(2**M)
    # coefmat = np.reshape(np.kron(B, coefmat), [-1, M])

    # # Make sign array
    # A = np.array([1, -1])
    # signmat = np.zeros([2**M, M])
    # ind = 0

    # for i in range(M, 0, -1):
    #     m = 2**(M-i)
    #     B = np.ones([2**(i - 1), m])
    #     C = np.kron(B, A)
    #     signmat[:, ind] = C.T.ravel()
    #     ind += 1

    # j = np.size(coefmat)/M
    # firstblock = signmat

    # for i in range(1, int(j / 2**M)):
    #     signmat = np.vstack([signmat, firstblock])

    # finalmat = coefmat * signmat

    finalmat = np.array(
        list(
            itertools.product(range(-1*order, order + 1, 1), repeat=M)
        )
    )

    y = []

    for pair in finalmat:
        if pair[1] != 0:
            y.append((soi - pair[0] * pivot) / pair[1])
        else:
            y.append(0)

    intermod_order = np.sum(abs(finalmat), 1)
    final = np.column_stack((y, finalmat, intermod_order))

    header = ['Frequency']
    for i in np.arange(M):
        header = header + ['Signal ' + str(i+1)]

    header = header + ['Order']

    T = pd.DataFrame(final, columns=header)

    T.query('Frequency > 0 & Frequency < inf', inplace=True)
    T.query('(Order > 0) & (Order <= @order)', inplace=True)
    T.drop_duplicates(inplace=True)
    T.sort_values(by=['Frequency'], inplace=True)
    T.reset_index(drop=True, inplace=True)

    return (T)
