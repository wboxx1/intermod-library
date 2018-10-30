# -*- coding: utf-8 -*-
"""
Intermod Tools

@author: wboxx
"""


import numpy as np
import pandas as pd
import helpers.helper_functions

def intermod_table(signals, order):
    """Calculates intermodulation products between the given signals to the specified order

    Will calculate all intermodulation products that could be potentially
    created between the given signals.  Must specify the highest order of
    intermods.

    :param signals: list of signals to calculate intermod products on
    :param order: highest order of intermod products to calculate
    :type signals: list[float]
    :type order: integer
    :returns: pandas dataframe containing the calculated intermod products
    :Example:

    >>> import intermod_library.intermod_tools as it
    >>> signals = [1000, 2000]
    >>> order = 3
    >>> table = it.intermod_table(signals, order)
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
    N = order + 1

    A = np.arange(0, N)

    coefmat = np.zeros([N**M, M])

    ind = 0

    for i in range(M, 0, -1):
        m = N**(M-i)
        B = np.ones([N**(i-1), m])
        C = np.kron(B, A)
        coefmat[:, ind] = C.T.ravel()
        ind += 1

    B = np.ones(2**M)
    coefmat = np.reshape(np.kron(B, coefmat), [-1, M])

    #% make sign array
    A = np.array([1, -1])
    signmat = np.zeros([2**M, M])
    ind = 0

    for i in range(M, 0, -1):
        m = 2**(M-i)
        B = np.ones([2**(i - 1), m])
        C = np.kron(B, A)
        signmat[:, ind] = C.T.ravel()
        ind += 1

    j = np.size(coefmat)/M
    firstblock = signmat

    for i in range(1, int(j / 2**M)):
        signmat = np.vstack([signmat, firstblock])

    finalmat = coefmat * signmat

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

    return (T)
