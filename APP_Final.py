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




# 1 - appliquer un fenetrage
# 2 - faire une fft sur le signal
# 3 - extraire phase, amplitude et frequence des sinusoides principales
# 4 - redresser le signal (valeur absolue)
# 5 - faire un filtre passe bas RIF a coeficients egaux (gain DC \ 0dB)
# 6 - trouver lenveloppe temporelle
# 7 - trouver lordre N du signal




# 5 - utiliser le passe-bas et passer h[0] pour avoir passe bande
# 6 -








import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import sys
import soundfile as sf







def APP_notes_frequency_dictionnary():
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







def APP_magnitude_to_dB(amplitude):
    """
    This function takes 1 argument(amplitude) and convert it into dB
    and print the input and output.

    :param magnitude: float: a signal amplitude to convert into dB
    :return: decibel: float: the argument passed converted to decibel
    """

    decibel = 20 * np.log10(amplitude)
    print("Converted {} into {} dB".format(amplitude, decibel))

    return decibel









def APP_sound_details(signal, Fs):
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

    # afficher le graphique du signal de base
    plt.title('Signal de Depart')
    plt.xlabel('Frequence Echantillonnage')
    plt.ylabel('Signal Amplitude')
    plt.plot(signal)
    plt.show()










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









def APP_enveloppe_temporelle(signal, N):
    """
    This function is used to print on a graph the temporal envelop

    :param signal: array: the signal to get the temporal envelop from
    :param N: int: the order of the filter
    """

    window = np.hanning(N)
    signal_abs = np.abs(signal)
    hann = 2 * window / window.sum()

    convolution = np.convolve(hann, signal_abs, mode='valid')

    plt.plot(convolution)
    plt.show()






# declaring a dictionnary
dictionnary = APP_notes_frequency_dictionnary()
# for key, array in dictionnary.items():
#     print(key, array[0])

# f = 1000
# omega = calculating_omega(Fs, f)

#fft_show_real_imag(signal, Fs)





# DEBUT DU PROBLEMATIQUE
# ouvrir le fichier et afficher les details et le graph
sound_data = './sounds/note_guitare_LAd.wav'
signal, Fs = sf.read(sound_data)
APP_sound_details(signal, Fs)

# filter section
N = 1024
K = 3
h = APP_filtre_RIF(N, K)
plt.plot(h)
plt.show()




def APP_window_and_fft(signal):
    window = np.hanning(signal.size)
    signal_window = window * signal
    fft_spectrum = np.fft.fft(signal_window)
    fft_spectrum_shift = np.fft.fftshift(APP_magnitude_to_dB(np.abs(fft_spectrum)))

    plt.subplot(3, 1, 1)
    plt.title("Fenetre de Hann")
    #plt.xlabel('Frequency Echantillons')
    plt.ylabel('FFT Amplitude')
    plt.plot(window)

    plt.subplot(3, 1, 2)
    plt.title("Signal de base")
    #plt.xlabel('Frequency Echantillons')
    plt.ylabel('FFT Amplitude')
    plt.plot(signal)

    plt.subplot(3, 1, 3)
    plt.title("Avec Fenetre de Hann et Shift")
    plt.plot(fft_spectrum_shift)
    plt.xlabel('Frequency Echantillon')
    plt.ylabel('FFT Amplitude (dB)')
    plt.tight_layout()

    plt.show()




APP_window_and_fft(signal)


APP_enveloppe_temporelle(signal, signal.size)










