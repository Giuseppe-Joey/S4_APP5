# File Name :               APP_Final.py
# GitHub : 	                https://github.com/Giuseppe-Joey/S4_APP5
# Author's Name : 	        Giuseppe Lomonaco - lomg2301
#                           Lucas Corrales - corl0701
# Author's Team :           N/A
# Author's College :        University of Sherbrooke
# Author's Study :          Electrical Engineering
# Author's Intern # :       N/A
# Season / Session  :       Summer 2022
# For : 	                APP5











import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import sys
import soundfile as sf
import math







def notes_frequency_dictionnary():
    """
    This function is a dictionnary containing all the k index,
    factor and frequency for every notes

    :return: dictionnary: dict: a dictionnary containing all musicals notes and values
    """
    dictionnary = {'DO':    [-9, 0.595, 261.6],
                   'DO#':   [-8, 0.630, 277.2],
                   'RE':    [-7, 0.667, 293.7],
                   'RE#':   [-6, 0.707, 311.1],
                   'MI':    [-5, 0.749, 329.6],
                   'FA':    [-4, 0.794, 349.2],
                   'FA#':   [-3, 0.841, 370.0],
                   'SOL':   [-2, 0.891, 392.0],
                   'SOL#':  [-1, 0.944, 415.3],
                   'LA':    [0, 1.000, 440.0],
                   'LA#':   [1, 1.060, 466.2],
                   'SI':    [2, 1.123, 493.9]}
    return dictionnary








def sound_processing_filter_1000Hz(filename_input, filename_output):
    """
    This function import the signal, plot the signal before filtering 1000Hz and after filtering

    :param filename_input: the file to extract data from
    :param filename_output: the file path to be written
    :return: None
    """

    # matplotlib inline
    plt.style.use('seaborn-poster')

    # some magic to see better quality graphic
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (9, 7)

    # reading the wav file (wavfile.read() function reads 16 or 32 bits wav file, 24 bits are not supported)
    signal, sampFreq = sf.read(filename_input)

    # getting the duration in seconds from frequency
    length_in_secs = signal.shape[0] / sampFreq

    # creating an array containing the time for the x_axis
    time = np.arange(signal.shape[0]) / signal.shape[0] * length_in_secs

    # keeping only the real numbers and not the complex part
    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1. / sampFreq)

    #obtaining Amplitude vs Frequency spectrum we find absolute value of fourier transform
    fft_spectrum_abs = np.abs(fft_spectrum)

     # plot the FFT amplitude BEFORE
    plt.figure("Filtering a signal", figsize=(12, 6))
    plt.subplot(121)
    plt.stem(freq, fft_spectrum_abs, 'b', markerfmt=" ", basefmt="-b")
    plt.title('Before filtering')
    plt.xlim(0, 3000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')

    # filtering the 1000 Hz
    for i, f in enumerate(freq):
        if f < 1020 and f > 980:  # (1)
            fft_spectrum_abs[i] = 0.0
            fft_spectrum[i] = 0.0

    # plot the FFT amplitude AFTER
    plt.subplot(122)
    plt.stem(freq, fft_spectrum_abs, 'b', markerfmt=" ", basefmt="-b")
    plt.title('After filtering')
    plt.xlim(0, 3000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.tight_layout()
    plt.show()

    irfft_spectrum = np.fft.irfft(fft_spectrum)

    # writing back the signal into .wav file
    sf.write(filename_output, irfft_spectrum, samplerate=sampFreq)










def magnitude_to_dB(amplitude):
    """
    This function takes 1 argument(amplitude) and convert it into dB
    and print the input and output.

    :param magnitude: float: a signal amplitude to convert into dB
    :return: decibel: float: the argument passed converted to decibel
    """

    decibel = 20 * np.log10(amplitude)

    return decibel









def sound_details(signal, Fs):
    """
    This function takes 1 argument (sound file in .wav format) and takes the
    signal and print all key parameters of this file

    :param sound_file: file: the file in .wav format to open
    :param Fs: file: the Sample Frequency
    :return: signal: the signal extracted from the sound file
    """

    print("----------------------------------------")
    print("----- Sound processing function --------")
    print("----------------------------------------")

    # getting the duration in seconds from frequency
    length_in_secs = signal.shape[0] / Fs

    # creating an array containing the time for the x_axis
    time = np.arange(signal.shape[0]) / signal.shape[0] * length_in_secs

    # print the sound type and the sample frequency
    print("Sound dtype is                       : {}".format(signal.dtype))
    print("Sound sample frequency (Fs) is       : {}".format(Fs))
    print("Sound shape is (left(mono), right)   : {}".format(signal.shape))
    print("Sound length in secs is              : {}".format('{:,.3f}'.format(length_in_secs)))
    #print("Sound time (what is this))          : {}".format(time))
    print("Sound size is                        : {}".format(signal.size))
    print("----------------------------------------\n\n")











def window_and_fft(signal):
    """
    This function create a Hann Window from the signal size, multiply the window on the
    signal first and than apply FFT Shift on the fft and convert the magnitude into decibel.

    :param signal: the data extracted from the .wav file
    :return: None
    """
    window = np.hanning(signal.size)
    signal_window = window * signal

    fft_spectrum = np.fft.fft(signal)
    fft_spectrum_abs = np.abs(fft_spectrum)
    fft_spectrum_shift = np.fft.fftshift(magnitude_to_dB(fft_spectrum_abs))



    plt.subplot(3, 2, 1)
    plt.title("Signal de base")
    plt.xlabel('Frequency Echantillons')
    plt.ylabel('FFT Amplitude')
    plt.plot(signal)

    plt.subplot(3, 2, 2)
    plt.title("Fenetre de Hann")
    plt.xlabel('Frequency Echantillons')
    plt.ylabel('FFT Amplitude')
    plt.plot(window)

    plt.subplot(3, 2, 5)
    plt.title("Signal FFT en Absolue")
    plt.plot(fft_spectrum_abs)
    plt.xlabel('Frequency Echantillon')
    plt.ylabel('FFT Amplitude')

    plt.subplot(3, 2, 4)
    plt.title("Fenetre de Hann fois Signal")
    plt.plot(signal_window)
    plt.xlabel('Frequency Echantillon')
    plt.ylabel('FFT Amplitude')

    plt.subplot(3, 2, 3)
    plt.title("Signal FFT")
    plt.plot(fft_spectrum)
    plt.xlabel('Frequency Echantillon')
    plt.ylabel('FFT Amplitude')

    plt.subplot(3, 2, 6)
    plt.title("Avec Fenetre de Hann et Shift")
    plt.plot(fft_spectrum_shift)
    plt.xlabel('Frequency Echantillon')
    plt.ylabel('FFT Amplitude (dB)')

    plt.tight_layout()

    plt.show()










def play_music(signal, sampFreq, dictionnary):
    """
    This function import the signal, plot the signal before filtering 1000Hz and after filtering

    :param filename_input:
    :param filename_output:
    :return:
    """

    # matplotlib inline
    plt.style.use('seaborn-poster')

    # some magic to see better quality graphic
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (9, 7)

    # getting the duration in seconds from frequency
    length_in_secs = signal.shape[0] / sampFreq

    # creating an array containing the time for the x_axis
    time = np.arange(signal.shape[0]) / signal.shape[0] * length_in_secs

    # keeping only the real numbers and not the complex part
    fft_spectrum = np.fft.fft(signal)
    fft_spectrum_magnitude = magnitude_to_dB(np.abs(fft_spectrum))

    freq = np.fft.fftfreq(fft_spectrum_magnitude.size, d=1. / sampFreq)
    plt.plot(fft_spectrum_magnitude)
    plt.show()

    freq_fond = freq[1691]
    print("Fondamentale: {}".format(freq_fond))



    for key, val in dictionnary.items():
        compteur = 0
        if key == 'SOL' and compteur < 3:
            f = val[2]
            somme = np.sin(2 * np.pi * freq_fond * length_in_secs)
            compteur += 1














def find_phase_ampl_freq(signal):

    fft_spectrum = np.fft.fft(signal)
    phase = np.angle(fft_spectrum, deg=True)

    fft_spectrum = np.abs(fft_spectrum)
    peaks, _ = find_peaks(fft_spectrum, prominence=(32, None))
    plt.plot(fft_spectrum)
    plt.plot(peaks, fft_spectrum[peaks], "x")
    # plt.plot(np.zeros_like(fft_spectrum), "--", color="gray")
    plt.show()














def env_temp(signal, k):

    sigabs = np.abs(signal)
    h = np.ones(k)/k

    # filter the data using convolution
    sigh = np.convolve(h, sigabs)

    plt.subplot(311)
    plt.title("signal de base")
    plt.plot(signal)
    plt.xlabel("fréquence échantillion")
    plt.ylabel("amplitude")

    plt.subplot(312)
    plt.title("signal redressé")
    plt.plot(sigabs)
    plt.xlabel("fréquence échantillion")
    plt.ylabel("amplitude")

    plt.subplot(313)
    plt.title("enveloppe temporelle")
    plt.plot(sigh)
    plt.xlabel("fréquence échantillion")
    plt.ylabel("amplitude")

    plt.tight_layout()
    plt.show()

    return sigh












def find_k():
    """
    This function find the k coeficients of a signal at -3dB for a rad/ech. (w)

    :return: k: int: the frequency
    """

    k = 0
    while k < 1000:
        k = k + 0.01
        x = ((np.sin((np.pi / 1000) * k / 2)) / (k*(np.sin((np.pi / 1000) / 2)))) * np.sqrt(2)
        if (x > 0.99999) and (x < 1.00001):
            k = math.trunc(k)
            print("k value is: {}".format(k))
            return k








