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









def coupe_bande(signal, fs):
    """
    This function is a low pass filter and a cut band filter

    :param signal: sound data: this is the extracted data from the sound
    :param fs: sample rate frequency of the extracted sound data
    :return: sig_lp: the signal convoluated twice with the cut band filter 1st and the the low pass filter
    """

    N = 1024
    n = np.arange(1, N)
    n1 = np.arange(0, N)
    fc = 20

    #1
    m = fc * N / fs
    k = round((m * 2)+1)

    if (k % 2) == 0:
        k = k + 1
        print("K value is: {}".format(k))


    else:
        print("WTF!!!")
        print("K value is: {}".format(k))

    #2
    h_lp = np.zeros(N)
    h_lp[n] = (np.sin(np.pi * n * k / N)) / (N * np.sin(np.pi * n / N))
    h_lp[0] = k / N

    #3
    w1 = np.pi * ((k - 1) / N)
    w0 = 2 * np.pi * 1000 / 44100
    print("w0 value is: {}".format(w0))

    #4 dirac
    dirac = np.zeros(N)
    dirac[0] = 1

    # sig_lp = np.convolve(h_lp, signal)


    #5
    h_cb = dirac - h_lp * 2 * np.cos(w0 * n1)

    #6 Black magic!!!
    sig_cb = np.convolve(h_cb, signal)
    sig_lp = np.convolve(h_lp, sig_cb)



    plt.subplot(3, 1, 1)
    plt.plot(signal)
    plt.title("signal")

    plt.subplot(3, 1, 2)
    plt.plot(h_cb)
    plt.title("coupe-bande")

    plt.subplot(3, 1, 3)
    sig_lp *= 10
    plt.plot(sig_lp)
    plt.title("signal coupe")

    plt.tight_layout() #bien mettre les titres
    plt.show()

    return sig_lp











def play_music(signal, Fs, dictionnary):
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
    length_in_secs = signal.shape[0] / Fs

    # creating an array containing the time for the x_axis
    time = np.arange(signal.shape[0]) / signal.shape[0] * length_in_secs

    # keeping only the real numbers and not the complex part
    fft_spectrum = np.fft.fft(signal)
    fft_spectrum_magnitude = magnitude_to_dB(np.abs(fft_spectrum))

    freq = np.fft.fftfreq(fft_spectrum_magnitude.size, d=1. / Fs)
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



























def env_temp(signal, k):
    """
    This function finds the temporal enveloppe of a signal

    :param signal: sound data: this is the extracted data from the sound
    :param k: int: the number of coeficients found
    :return: sigh: the convoluated signal with k
    """

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

    :return: k: int: the k coeficients found
    """

    k = 0
    while k < 1000:
        k = k + 0.01
        x = ((np.sin((np.pi / 1000) * k / 2)) / (k*(np.sin((np.pi / 1000) / 2)))) * np.sqrt(2)
        if (x > 0.99999) and (x < 1.00001):
            k = math.trunc(k)
            print("k value is: {}".format(k))
            return k

















def find_freq_peaks(signal, Fs):
    """
    This function is used to find the peaks of a signal and print them on a graph

    :param signal: sound data: this is the extracted data from the sound
    :param fs: this is the sample frequency of the extracted signal
    :return: None
    """

    #obtaining Amplitude vs Frequency spectrum we find absolute value of fourier transform
    fft_spectrum = np.fft.fft(signal)
    fft_spectrum_abs = np.abs(fft_spectrum)
    peaks, _ = find_peaks(fft_spectrum_abs, prominence=(32, None), height=345)
    print("the number of peaks is: {}".format(len(peaks)))


     # plot the FFT amplitude BEFORE
    plt.figure("Finding peaks harmonics", figsize=(12, 6))
    plt.subplot(121)
    plt.plot(fft_spectrum_abs)
    plt.plot(peaks, fft_spectrum_abs[peaks], "x")
    plt.title('Harmonics')
    plt.xlabel('Sample Frequency')
    plt.ylabel('FFT Amplitude')




    fft_spectrum_dB = magnitude_to_dB(np.abs(fft_spectrum))


     # plot the FFT amplitude BEFORE
    plt.subplot(122)
    plt.plot(fft_spectrum_dB)
    plt.plot(peaks, fft_spectrum_dB[peaks], "x")
    plt.title('Harmonics')
    plt.xlabel('Sample Frequency')
    plt.ylabel('FFT Amplitude (dB)')
    plt.tight_layout()
    plt.show()







def find_amp_and_phase(signal, fs):
    """
    This function find the magnitude and the phase of a signal

    :param signal: sound data: this is the extracted data from the sound
    :param fs: this is the sample frequency of the extracted signal
    :return: None
    """

    # creeation de la fennetre
    window = np.hanning(signal.size)
    signal_window = signal * window


    fft_spectrum = np.fft.fft(signal_window)
    magnitude_fft = np.abs(fft_spectrum)
    print("Magnitude: {}".format(magnitude_fft))

    magnitude_dB = magnitude_to_dB(magnitude_fft)
    print("Magnitude in (dB): {}".format(magnitude_dB))

    phase = np.angle(fft_spectrum)
    print("Phase in rad/ech.: {}".format(phase))