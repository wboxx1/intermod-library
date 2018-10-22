# -*- coding: utf-8 -*-
"""
Intermod Tools

@author: wboxx
"""


import numpy as np
import pandas as pd


def simple_fft(signal, scan_rate):
    """perform an Fast Fourier Transform of a real-valued input signal
    
    perform an FFT of a real-valued input signal, and generate the output
    in amplitude and phase, scaled to the same units as the input.
    
    :param signal: the time series signal to transform
    :param scan_rate: the sampling frequency (in Hertz)
    :type signal: list[float]
    :type scan_rate: float
    :returns: frequency, amplitude, and phase vectors
    :rtype: tuple(list[float], list[float], list[float])
    :returns frq: a vector of frequency points (in Hertz)
    :returns amp: a vector of amplitudes (same units as the input signal)
    :returns phase: a vector of phases (in radians)
    :Example:


    """

    L = np.size(signal)
    NFFT = 2**np.ceil(np.log2(L))

    z = np.fft.fft(signal, n=int(NFFT)) / L

    deltaf = 1 / (NFFT / ScanRate)

    frq = np.arange(-NFFT / 2, NFFT / 2) * deltaf

    amp = np.squeeze(np.abs(np.fft.fftshift(z)))

    amp = amp / np.max(amp)

    phase = np.angle(np.fft.fftshift(z))

    return frq, amp, phase


def intermod_table(signals, order):
"""Calculates intermods from the given signals to the specified order

"""
#    signals = [0.25,7.6,9.2,12]
#    order = 3  # Number of harmonics
#    order = 3  # Number of harmonics
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

    # np.tile(signmat, (int(j/ 2**M))

    finalmat = coefmat * signmat

    intermods = abs(np.dot(finalmat, signals))

    Order = np.sort(np.sum(abs(finalmat), 1))
    ind = np.argsort(np.sum(abs(finalmat), 1))

    finalmat = finalmat[ind]
    intermods = intermods[ind]
    intermods, ia = np.unique(intermods, return_index=True)
    Signs = finalmat[ia] + 0
    Order = Order[ia]
    final = np.column_stack((intermods, Signs, Order))
    mask = final[:, -1] < N
    final = final[mask]

    header = ['Frequency']
    for i in np.arange(M):
        header = header + ['Signal ' + str(i+1)]

    header = header + ['Order']

    T = pd.DataFrame(final, columns=header)

    return (T[T.Order > 1])
