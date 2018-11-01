import numpy as np

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

    >>> import helpers.helper_functions as helpers
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from functools import reduce

    >>> pi = np.pi
    >>> freqs = [1000, 2000]
    >>> fs = 5*2*np.max(freqs)
    >>> t = np.arange(0, 2.0**10) / fs

    >>> y = 0
    >>> for f in freqs:
            y = y + np.sin(2 * pi * f * t)

    >>> h = np.hamming(np.size(y))

    >>> y_w = y * h

    >>> frq, amp, phase = helpers.simple_fft(y_w, fs)

    >>> fig, ax = plt.subplots()
    >>> ax.plot(frq, amp)
    >>> fig.show()
    """

    L = np.size(signal)
    NFFT = 2**np.ceil(np.log2(L))

    z = np.fft.fft(signal, n=int(NFFT)) / L

    deltaf = 1 / (NFFT / scan_rate)

    frq = np.arange(-NFFT / 2, NFFT / 2) * deltaf

    amp = np.squeeze(np.abs(np.fft.fftshift(z)))

    amp_norm = amp / np.max(amp)

    phase = np.angle(np.fft.fftshift(z))

    return frq, amp_norm, phase
