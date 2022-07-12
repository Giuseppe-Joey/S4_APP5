

# File Name :               APP.py

# GitHub : 	                https://github.com/Giuseppe-Joey/S4_APP5
# Author's Name : 	        Giuseppe Lomonaco - lomg2301 && Lucas Corrales - corl0701
# Author's Team :           N/A
# Author's College :        University of Sherbrooke
# Author's Study :          Electrical Engineering
# Author's Intern # :       N/A
# Season / Session  :       Summer 2022
# For : 	                APP - testing before integrating to APP_Final.py






import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

import soundfile as sf





# 1 - appliquer un fenetrage
# 2 - faire une fft sur le signal
# 3 - extraire phase, amplitude et frequence des sinusoides principales
# 4 - redresser le signal (valeur absolue)
# 5 - faire un filtre passe bas RIF a coeficients egaux (gain DC \ 0dB)
# 6 - trouver lenveloppe temporelle
# 7 - trouver lordre N du signal




# 5 - utiliser le passe-bas et passer h[0] pour avoir passe bande
# 6 -










def APP_calculating_omega(Fs, f):
    """
    This function calculates
    :param Fs: int: sample frequency
    :param f: int: frequency
    :return: omega:
    """
    omega = 2 * np.pi * f / Fs
    print("Fs       :{} Sample/secs".format(Fs))
    print("f        :{} Hz".format(f))
    print("Omega    :{} rad/echantillons".format('{:,.2f}'.format(omega)))

    return omega










def APP_fft_show_real_imag(signal, Fs):
    """
    This function print the real and the imaginary part of a signal

    :param signal: the original  signal
    :param Fs: the sample rate frequency
    :return: return the fft signal
    """

    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1. / Fs)

    plt.figure("Filtering a signal", figsize=(12, 6))
    plt.subplot(121)
    plt.stem(freq, np.real(fft_spectrum), 'b', markerfmt=" ", basefmt="-b")
    plt.title('Partie Reelle')
    plt.xlim(0, 3000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')

    plt.subplot(122)
    plt.stem(freq, np.imag(fft_spectrum), 'b', markerfmt=" ", basefmt="-b")
    plt.title('Partie Imaginaire')
    plt.xlim(0, 3000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.tight_layout()
    plt.show()


    plt.show()

    return fft_spectrum









# cela va donner un passe bas mais pas centre a 0
def APP_filtre_RIF(N, K):
    """
    This function makes a filter from an order N and a factor K

    :param N: int: the order of the filter
    :param K: int: the coeficient factor...
    :return: h: int array: the array containing all int values
    """

    n = np.arange(1, N)
    h = np.zeros(N)

    h[0] = K / N
    h[n] = np.sin(np.pi * n * K / N) / (N * np.sin(np.pi * n / N))

    return h











def amplitude_example(signal, fs):
    # Number of sample points
    N = 1000

    # Sample spacing
    T = 1.0 / 800.0  # f = 800 Hz

    # Create a signal
    x = np.linspace(0.0, N * T, N)
    t0 = np.pi / 6  # non-zero phase of the second sine
    y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(200.0 * 2.0 * np.pi * x + t0)
    yf = np.fft.fft(y)  # to normalize use norm='ortho' as an additional argument

    # Where is a 200 Hz frequency in the results?
    freq = np.fft.fftfreq(x.size, d=T)
    index, = np.where(np.isclose(freq, 200, atol=1 / (T * N)))

    # Get magnitude and phase
    magnitude = np.abs(yf[index[0]])
    phase = np.angle(yf[index[0]])
    print("Magnitude:", magnitude, ", phase:", phase)

    # Plot a spectrum
    plt.plot(freq[0:N // 2], 2 / N * np.abs(yf[0:N // 2]), label='amplitude spectrum')  # in a conventional form
    plt.plot(freq[0:N // 2], np.angle(yf[0:N // 2]), label='phase spectrum')
    plt.legend()
    plt.grid()
    plt.show()


















def APP_find_phase_ampl_freq(signal):

    fft_spectrum = np.fft.fft(signal)
    phase = np.angle(fft_spectrum, deg=True)

    fft_spectrum = np.abs(fft_spectrum)
    peaks, _ = find_peaks(fft_spectrum, prominence=(32, None))
    plt.plot(fft_spectrum)
    plt.plot(peaks, fft_spectrum[peaks], "x")
    # plt.plot(np.zeros_like(fft_spectrum), "--", color="gray")
    plt.show()




























